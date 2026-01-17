"""
Text-prompt SAM3 multiview body demo for HoCap/ExoEgo datasets.

This module provides multiview 3D body mesh fitting across exocentric cameras:
- Uses SAM3 video inference for person detection/tracking.
- Runs SAM3DBodyEstimator on each camera to produce per-view body meshes.
- **Phase 2**: Uses multiview optimization to produce a single fused mesh.
- Logs results to Rerun for interactive 3D visualization.

Assumptions:
- Single person in the multiview capture (multi-person support in future).

Usage:
    pixi run python tool/demo_mv_body.py hocap --root-directory data/sample

See Also:
    - demo_mv_video.py: Multiview SAM3 segmentation (no body fitting)
    - demo.py: Single-image 3D body estimation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import time

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from jaxtyping import Bool, Float32, Int, UInt8
from numpy import ndarray
from torch import Tensor
from simplecv.configs.exoego_dataset_configs import AnnotatedExoEgoDatasetUnion
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence, ExoEgoSample
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole, log_video
from tqdm import tqdm
from transformers import Sam3VideoConfig, Sam3VideoModel, Sam3VideoProcessor

from sam3d_body.api.visualization import (
    BOX_PALETTE,
    SEG_CLASS_OFFSET,
    SEG_OVERLAY_ALPHA,
    compute_vertex_normals,
)
from sam3d_body.build_models import load_sam_3d_body_hf
from sam3d_body.metadata.mhr70 import MHR70_ID2NAME, MHR70_IDS, MHR70_LINKS
from sam3d_body.models.meta_arch import SAM3DBody
from sam3d_body.ops import MultiviewBodyOptimizer, MultiviewOptimizerConfig, validate_reprojection
from sam3d_body.sam_3d_body_estimator import FinalPosePrediction, SAM3DBodyEstimator

# ──────────────────────────────────────────────────────────────────────────────
# Configuration Dataclasses
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class Sam3MVBodyModelConfig:
    """Settings for the SAM3 video checkpoint and compute placement."""

    checkpoint: str = "facebook/sam3"
    """Model identifier passed to ``Sam3VideoModel.from_pretrained``."""
    device: Literal["auto", "cpu", "cuda"] = "auto"
    """Compute device selection; ``auto`` prefers CUDA when available."""
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    """Torch dtype used for model weights and inference."""
    processing_device: Literal["cpu", "cuda"] = "cpu"
    """Device used by the processor for per-frame preprocessing."""
    video_storage_device: Literal["cpu", "cuda"] = "cpu"
    """Device that caches video tensors within the inference session."""
    score_threshold_detection: float = 0.85
    """Minimum detection score before tracking."""
    new_det_thresh: float | None = None
    """Optional stricter threshold for new detections."""


@dataclass(slots=True)
class Sam3MVBodyDemoConfig:
    """CLI options for multiview SAM3 body mesh fitting with Rerun."""

    dataset: AnnotatedExoEgoDatasetUnion
    """ExoEgo dataset configuration (Tyro-resolvable)."""
    prompt: str = "person"
    """Text concept to segment across all cameras."""
    max_frames: int | None = None
    """Optional cap on the number of frames to process."""
    rr_config: RerunTyroConfig = field(default_factory=RerunTyroConfig)
    """Viewer/runtime options for Rerun (window layout, recording, etc.)."""
    model_config: Sam3MVBodyModelConfig = field(default_factory=Sam3MVBodyModelConfig)
    """Checkpoint, device, and dtype settings for the SAM3 video model."""


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────


def _to_numpy(array_like: torch.Tensor | np.ndarray | list) -> np.ndarray:
    """Convert tensors/lists to numpy for downstream processing."""
    if isinstance(array_like, torch.Tensor):
        return array_like.detach().cpu().numpy()
    return np.asarray(array_like)


def _build_projection_matrix(
    K_33: Float32[ndarray, "3 3"],
    world_T_cam: Float32[ndarray, "4 4"],
) -> Float32[ndarray, "3 4"]:
    """Build projection matrix P = K @ [R|t] from intrinsics and extrinsics.

    Args:
        K_33: Camera intrinsics (3x3).
        world_T_cam: Transformation from camera to world coordinates.

    Returns:
        3x4 projection matrix for projecting world points to image coordinates.
    """
    # cam_T_world is the inverse of world_T_cam
    cam_T_world: Float32[ndarray, "4 4"] = np.linalg.inv(world_T_cam).astype(np.float32)
    # Extract [R|t] (3x4)
    Rt: Float32[ndarray, "3 4"] = cam_T_world[:3, :]
    # P = K @ [R|t]
    P: Float32[ndarray, "3 4"] = K_33 @ Rt
    return P


# ──────────────────────────────────────────────────────────────────────────────
# Rerun Visualization Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_blueprint(exo_cam_names: list[str], max_exo_views: Literal[4, 8] = 8) -> rrb.Blueprint:
    """
    Create a viewer layout with 3D body view and per-camera 2D tabs.

    Args:
        exo_cam_names: List of exocentric camera names.
        max_exo_views: Maximum number of camera views to show.

    Returns:
        Blueprint configured to show 3D view, text prompt, and camera tabs.
    """
    main_view: rrb.ContainerLike = rrb.Spatial3DView(
        origin="/",
        name="3D Body View",
        eye_controls=rrb.EyeControls3D(
            kind="Orbital", position=[0.0, 2.0, 2.0], look_target=[0, 0, 0.5], spin_speed=0.5
        ),
        line_grid=rrb.LineGrid3D(visible=False),
    )

    if exo_cam_names:
        exo_tabs_row: list[rrb.ContainerLike] = []
        for name in exo_cam_names[:max_exo_views]:
            origin_base: str = f"/world/exo/{name}/pinhole"
            exo_tabs_row.append(
                rrb.Tabs(
                    contents=[
                        rrb.Spatial2DView(
                            origin=origin_base,
                            name=f"exo/{name}",
                            contents=[
                                f"{origin_base}/video",
                                f"{origin_base}/boxes",
                                f"{origin_base}/keypoints_2d",
                                f"{origin_base}/segmentation_ids",
                            ],
                        )
                    ]
                )
            )

        prompt_view: rrb.ContainerLike = rrb.TextDocumentView()
        exo_view: rrb.ContainerLike = rrb.Horizontal(contents=exo_tabs_row)
        main_view = rrb.Horizontal(contents=[prompt_view, main_view], column_shares=[1, 10])
        main_view = rrb.Vertical(contents=[main_view, exo_view], row_shares=[4, 1])

    return rrb.Blueprint(
        rrb.Horizontal(contents=[main_view], column_shares=[4, 1]),
        collapse_panels=True,
    )


def _log_annotation_context() -> None:
    """Register annotation context with MHR70 skeleton and segmentation colors."""
    # Person class for keypoints
    person_class = rr.ClassDescription(
        info=rr.AnnotationInfo(id=0, label="Person", color=(0, 0, 255)),
        keypoint_annotations=[rr.AnnotationInfo(id=idx, label=name) for idx, name in MHR70_ID2NAME.items()],
        keypoint_connections=MHR70_LINKS,
    )

    # Segmentation classes
    seg_classes: list[rr.ClassDescription] = [
        rr.ClassDescription(info=rr.AnnotationInfo(id=SEG_CLASS_OFFSET, label="Background", color=(64, 64, 64, 0))),
    ]
    num_colors: int = BOX_PALETTE.shape[0]
    for idx in range(1, num_colors + 1):
        rgb: UInt8[ndarray, "3"] = BOX_PALETTE[(idx - 1) % num_colors, :3]
        color_rgba: tuple[int, int, int, int] = (int(rgb[0]), int(rgb[1]), int(rgb[2]), SEG_OVERLAY_ALPHA)
        seg_classes.append(
            rr.ClassDescription(info=rr.AnnotationInfo(id=SEG_CLASS_OFFSET + idx, label=f"Person-{idx}", color=color_rgba))
        )

    # Debug keypoint classes for validation visualization
    debug_classes: list[rr.ClassDescription] = [
        rr.ClassDescription(info=rr.AnnotationInfo(id=1000, label="detected_kpts", color=(0, 255, 0))),  # Green
        rr.ClassDescription(info=rr.AnnotationInfo(id=1001, label="projected_triangulated", color=(255, 128, 0))),  # Orange
        rr.ClassDescription(info=rr.AnnotationInfo(id=1002, label="projected_mhr", color=(0, 128, 255))),  # Blue
    ]

    rr.log("/", rr.AnnotationContext([person_class, *seg_classes, *debug_classes]), static=True)


def _log_cameras(exoego_sequence: BaseExoEgoSequence) -> None:
    """Log camera intrinsics/extrinsics following the schema."""
    if exoego_sequence.exo_sequence is not None:
        for cam in exoego_sequence.exo_sequence.exo_cam_list:
            if cam is None:
                continue
            cam_log_path: Path = Path("/world") / "exo" / cam.name
            log_pinhole(
                cam,
                cam_log_path=cam_log_path,
                image_plane_distance=exoego_sequence.exo_sequence.image_plane_distance,
                static=True,
            )


# ──────────────────────────────────────────────────────────────────────────────
# Segmentation Processing
# ──────────────────────────────────────────────────────────────────────────────


def _prepare_segmentation_assets(
    frame_rgb: UInt8[ndarray, "h w 3"],
    processed_outputs: dict,
) -> tuple[
    UInt8[ndarray, "h w"],
    Float32[ndarray, "n 4"] | None,
    Float32[ndarray, "n"] | None,
    Float32[ndarray, "n h w"] | None,
]:
    """
    Convert SAM3 outputs into segmentation IDs, boxes, and masks.

    Returns:
        Tuple of (seg_map, boxes, scores, masks_np) where:
        - seg_map contains class IDs using SEG_CLASS_OFFSET
        - boxes are in XYXY format
        - scores are detection confidences
        - masks_np are the raw probability masks [N, H, W]
    """
    raw_masks = processed_outputs.get("masks")
    raw_boxes = processed_outputs.get("boxes")
    raw_scores = processed_outputs.get("scores")

    if raw_masks is None or len(raw_masks) == 0:
        h: int = int(frame_rgb.shape[0])
        w: int = int(frame_rgb.shape[1])
        return (np.full((h, w), SEG_CLASS_OFFSET, dtype=np.uint8), None, None, None)

    masks_np: Float32[ndarray, "n h w"] = _to_numpy(raw_masks).astype(np.float32, copy=False)
    if masks_np.ndim == 2:
        masks_np = masks_np[None, ...]

    h = int(frame_rgb.shape[0])
    w = int(frame_rgb.shape[1])
    seg_map: UInt8[ndarray, "h w"] = np.full((h, w), SEG_CLASS_OFFSET, dtype=np.uint8)
    num_instances: int = int(masks_np.shape[0])

    for idx in range(num_instances):
        mask: Float32[ndarray, "h w"] = np.asarray(masks_np[idx], dtype=np.float32)
        mask_bool: Bool[ndarray, "h w"] = mask >= 0.5
        class_id: int = SEG_CLASS_OFFSET + idx + 1
        seg_map = np.where(mask_bool, np.uint8(class_id), seg_map)

    boxes_np: Float32[ndarray, "n 4"] | None = None
    scores_np: Float32[ndarray, "n"] | None = None
    if raw_boxes is not None:
        boxes_np = _to_numpy(raw_boxes).astype(np.float32, copy=False)
        if boxes_np.ndim == 1:
            boxes_np = boxes_np[None, :]
    if raw_scores is not None:
        scores_np = _to_numpy(raw_scores).astype(np.float32, copy=False).reshape(-1)

    return seg_map, boxes_np, scores_np, masks_np


def _log_camera_outputs(
    *,
    pinhole_path: str,
    frame_idx: int,
    seg_map: UInt8[ndarray, "h w"],
    boxes: Float32[ndarray, "n 4"] | None,
    scores: Float32[ndarray, "n"] | None,
    timestamp_ns: int | None = None,
) -> None:
    """Log per-camera segmentation and boxes to Rerun (video is logged as static asset)."""
    # Sync with video_time timeline if timestamp available
    if timestamp_ns is not None:
        rr.set_time("video_time", duration=1e-9 * timestamp_ns)
    rr.set_time("frame_idx", sequence=frame_idx)
    # Note: Video is logged once as AssetVideo, no per-frame image logging needed
    rr.log(f"{pinhole_path}/segmentation_ids", rr.SegmentationImage(seg_map))

    if boxes is None:
        return

    class_ids: Int[ndarray, "n"] = np.arange(1, boxes.shape[0] + 1, dtype=np.int32)
    labels: list[str] | None = None
    if scores is not None:
        labels = [f"{float(score):.2f}" for score in scores.tolist()]

    colors: UInt8[ndarray, "n 3"] = np.asarray(
        [BOX_PALETTE[idx % BOX_PALETTE.shape[0], :3] for idx in range(boxes.shape[0])],
        dtype=np.uint8,
    )
    rr.log(
        f"{pinhole_path}/boxes",
        rr.Boxes2D(
            array=boxes,
            array_format=rr.Box2DFormat.XYXY,
            class_ids=class_ids,
            labels=labels,
            colors=colors,
            show_labels=True,
        ),
    )


def _log_body_predictions(
    *,
    pred_list: list[FinalPosePrediction],
    faces: Int[ndarray, "n_faces 3"],
    world_T_cam: Float32[ndarray, "4 4"],
    cam_name: str,
    frame_idx: int,
    pinhole_path: str,
) -> None:
    """
    Log 3D body meshes and keypoints in world coordinates.

    Args:
        pred_list: List of per-person body predictions from SAM3DBodyEstimator.
        faces: Triangle indices for the body mesh.
        world_T_cam: 4x4 transformation matrix from camera to world coordinates.
        cam_name: Camera name for logging path organization.
        frame_idx: Current frame index for timeline.
        pinhole_path: Base path for 2D logging.
    """
    rr.set_time("frame_idx", sequence=frame_idx)
    mesh_root_path: Path = Path("/world/pred") / cam_name
    faces_int: Int[ndarray, "n_faces 3"] = np.ascontiguousarray(faces, dtype=np.int32)

    for i, output in enumerate(pred_list):
        box_color: UInt8[ndarray, "1 4"] = BOX_PALETTE[i % len(BOX_PALETTE)].reshape(1, 4)

        # 3D keypoints: transform from camera to world coordinates
        kpts_cam: Float32[ndarray, "n_kpts 3"] = np.ascontiguousarray(output.pred_keypoints_3d, dtype=np.float32)
        cam_t: Float32[ndarray, "3"] = np.ascontiguousarray(output.pred_cam_t, dtype=np.float32)
        kpts_cam_translated: Float32[ndarray, "n_kpts 3"] = kpts_cam + cam_t

        # Transform to world coordinates
        kpts_hom: Float32[ndarray, "n_kpts 4"] = np.concatenate(
            [kpts_cam_translated, np.ones((kpts_cam_translated.shape[0], 1), dtype=np.float32)], axis=1
        )
        kpts_world: Float32[ndarray, "n_kpts 3"] = (kpts_hom @ world_T_cam.T)[:, :3]

        rr.log(
            str(mesh_root_path / f"kpts3d_{i}"),
            rr.Points3D(
                positions=kpts_world,
                keypoint_ids=MHR70_IDS,
                class_ids=0,
                colors=(0, 255, 0),
            ),
        )

        # 2D keypoints for the 2D view
        kpts_uv: Float32[ndarray, "n_kpts 2"] = np.ascontiguousarray(output.pred_keypoints_2d, dtype=np.float32)
        rr.log(
            f"{pinhole_path}/keypoints_2d",
            rr.Points2D(
                positions=kpts_uv,
                keypoint_ids=MHR70_IDS,
                class_ids=0,
                colors=(0, 255, 0),
            ),
        )

        # Body mesh: transform vertices to world coordinates
        verts_cam: Float32[ndarray, "n_verts 3"] = np.ascontiguousarray(output.pred_vertices, dtype=np.float32)
        verts_cam_translated: Float32[ndarray, "n_verts 3"] = verts_cam + cam_t
        verts_hom: Float32[ndarray, "n_verts 4"] = np.concatenate(
            [verts_cam_translated, np.ones((verts_cam_translated.shape[0], 1), dtype=np.float32)], axis=1
        )
        verts_world: Float32[ndarray, "n_verts 3"] = (verts_hom @ world_T_cam.T)[:, :3]

        vertex_normals: Float32[ndarray, "n_verts 3"] = compute_vertex_normals(verts_world, faces_int)

        rr.log(
            str(mesh_root_path / f"mesh_{i}"),
            rr.Mesh3D(
                vertex_positions=verts_world,
                triangle_indices=faces_int,
                vertex_normals=vertex_normals,
                albedo_factor=(
                    float(box_color[0, 0]) / 255.0,
                    float(box_color[0, 1]) / 255.0,
                    float(box_color[0, 2]) / 255.0,
                    0.6,
                ),
            ),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────────────────────────────


def main(cfg: Sam3MVBodyDemoConfig) -> None:
    """
    Run multiview SAM3 body mesh fitting across exocentric cameras.

    This function:
    1. Loads the ExoEgo dataset and creates a shared SAM3 video session.
    2. Initializes the SAM3DBodyEstimator for per-camera body fitting.
    3. For each timestep, processes all exo cameras through detection + body estimation.
    4. Transforms meshes to world coordinates and logs visualizations to Rerun.

    Args:
        cfg: Demo configuration with dataset, prompt, and model settings.

    Raises:
        RuntimeError: If dataset yields zero frames.
    """
    sequence: BaseExoEgoSequence = cfg.dataset.setup()

    if len(sequence) == 0:
        raise RuntimeError("Dataset yielded zero canonical frames.")

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map: dict[str, torch.dtype] = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype: torch.dtype = dtype_map[cfg.model_config.dtype] if device.type == "cuda" else torch.float32

    # Load SAM3 video model and processor
    config_kwargs: dict = {
        "score_threshold_detection": cfg.model_config.score_threshold_detection,
    }
    if cfg.model_config.new_det_thresh is not None:
        config_kwargs["new_det_thresh"] = cfg.model_config.new_det_thresh
    else:
        config_kwargs["new_det_thresh"] = cfg.model_config.score_threshold_detection

    model_cfg = Sam3VideoConfig.from_pretrained(cfg.model_config.checkpoint, **config_kwargs)
    model = Sam3VideoModel.from_pretrained(cfg.model_config.checkpoint, config=model_cfg).to(device=device, dtype=dtype)
    processor = Sam3VideoProcessor.from_pretrained(cfg.model_config.checkpoint)

    # Initialize a single video inference session for all cameras
    inference_session = processor.init_video_session(
        video=None,
        inference_device=device,
        processing_device=cfg.model_config.processing_device,
        video_storage_device=cfg.model_config.video_storage_device,
        dtype=dtype,
    )
    processor.add_text_prompt(inference_session=inference_session, text=cfg.prompt)

    # Load SAM3DBody model for body mesh fitting
    load_output = load_sam_3d_body_hf(repo_id="facebook/sam-3d-body-dinov3")
    sam3d_body_model: SAM3DBody = load_output[0]
    body_estimator = SAM3DBodyEstimator(sam_3d_body_model=sam3d_body_model)

    parent_log_path: Path = Path("world")
    exo_cam_names: list[str] = (
        [cam.name for cam in sequence.exo_sequence.exo_cam_list if cam is not None] if sequence.exo_sequence else []
    )

    # Log static data
    rr.log("text-prompt", rr.TextDocument(f"## Prompt\n\n{cfg.prompt}", media_type="text/markdown"), static=True)
    rr.send_blueprint(_make_blueprint(exo_cam_names))
    rr.log("/", sequence.world_coordinate_system, static=True)
    _log_annotation_context()
    _log_cameras(sequence)

    # Log video assets once (much more efficient than per-frame images)
    # Pattern from simplecv/apis/view_exoego.py
    video_timestamps: dict[str, Int[ndarray, "n_frames"]] = {}
    timeline: str = "video_time"
    if sequence.exo_sequence is not None:
        exo_video_names: list[str] = sequence.exo_sequence.exo_video_names
        exo_video_paths: list[Path] = sequence.exo_sequence.exo_video_paths
        # Get video blobs if available (RRD sequences), otherwise use file paths (HoCap)
        exo_video_blobs: dict[str, bytes] | None = getattr(sequence.exo_sequence, "_video_blobs", None)
        for stream_name, video_file in zip(exo_video_names, exo_video_paths, strict=True):
            video_log_path: Path = parent_log_path / "exo" / stream_name / "pinhole" / "video"
            # Use blob if available (RRD), otherwise use file path (HoCap)
            video_source: bytes | Path = (
                exo_video_blobs[stream_name]
                if exo_video_blobs and stream_name in exo_video_blobs
                else video_file
            )
            # Skip if file path doesn't exist (shouldn't happen with blobs)
            if isinstance(video_source, Path) and not video_source.exists():
                continue
            timestamps_ns: Int[ndarray, "n_frames"] = log_video(
                video_source, video_log_path, timeline=timeline
            )
            video_timestamps[stream_name] = timestamps_ns

    total_frames: int = len(sequence)
    if cfg.max_frames is not None:
        total_frames = min(total_frames, cfg.max_frames)

    # Build projection matrices for all valid cameras (once, since cameras are static)
    projection_matrices_list: list[Float32[ndarray, "3 4"]] = []
    valid_cam_indices: list[int] = []
    valid_cam_names: list[str] = []
    world_T_cam_list: list[Float32[ndarray, "4 4"]] = []

    first_sample: ExoEgoSample = sequence[0]
    if first_sample.exo_cam_params_list is not None:
        for cam_idx, cam_params in enumerate(first_sample.exo_cam_params_list):
            if cam_params is None:
                continue
            k_raw = cam_params.intrinsics.k_matrix
            if k_raw is None:
                continue
            K_33: Float32[ndarray, "3 3"] = k_raw.astype(np.float32)
            world_T_cam: Float32[ndarray, "4 4"] = cam_params.extrinsics.world_T_cam.astype(np.float32)
            P: Float32[ndarray, "3 4"] = _build_projection_matrix(K_33, world_T_cam)
            projection_matrices_list.append(P)
            valid_cam_indices.append(cam_idx)
            valid_cam_names.append(cam_params.name)
            world_T_cam_list.append(world_T_cam)

    if len(projection_matrices_list) < 2:
        print("[warn] Need at least 2 valid cameras for multiview optimization. Falling back to per-camera mode.")

    projection_matrices: Float32[ndarray, "n_views 3 4"] = np.stack(projection_matrices_list, axis=0) if projection_matrices_list else np.zeros((0, 3, 4), dtype=np.float32)
    n_views: int = len(valid_cam_indices)

    # Initialize multiview optimizer (reused across frames for temporal smoothing)
    optimizer: MultiviewBodyOptimizer | None = None
    first_world_T_cam: Float32[ndarray, "4 4"] | None = None
    if n_views >= 2:
        first_world_T_cam = world_T_cam_list[0]
        # Get MHR head from body_estimator for differentiable forward kinematics
        mhr_head = body_estimator.model.head_pose
        optimizer = MultiviewBodyOptimizer(
            projection_matrices=projection_matrices,
            first_world_T_cam=first_world_T_cam,
            mhr_head=mhr_head,
            config=MultiviewOptimizerConfig(num_iterations=30, learning_rate=0.01),
            device=str(device),
        )

    # Use no_grad for inference (allows enable_grad override for optimization)
    with torch.no_grad():
        # Timing accumulators
        time_inference: float = 0.0
        time_triangulation: float = 0.0
        time_optimization: float = 0.0
        time_validation: float = 0.0
        time_logging: float = 0.0

        for frame_idx in tqdm(range(total_frames), desc="Processing frames"):
            sample: ExoEgoSample = sequence[frame_idx]

            if sample.exo_bgr_list is None or sample.exo_cam_params_list is None:
                continue

            # ===== Phase 1: Collect data from all cameras =====
            t_phase1_start: float = time.perf_counter()
            per_view_predictions: list[FinalPosePrediction] = []
            per_view_keypoints_2d: list[Float32[ndarray, "n_kpts 3"]] = []  # [u, v, conf]

            for _view_idx, cam_idx in enumerate(valid_cam_indices):
                bgr = sample.exo_bgr_list[cam_idx]
                cam_params = sample.exo_cam_params_list[cam_idx]
                if cam_params is None:
                    continue

                frame_bgr: UInt8[ndarray, "h w 3"] = bgr
                frame_rgb: UInt8[ndarray, "h w 3"] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # Run SAM3 detection
                inputs = processor(images=frame_rgb, device=device, return_tensors="pt")  # type: ignore[misc]
                model_outputs = model(
                    inference_session=inference_session,
                    frame=inputs.pixel_values[0],
                    reverse=False,
                )
                processed_outputs = processor.postprocess_outputs(
                    inference_session,
                    model_outputs,
                    original_sizes=inputs.original_sizes,
                )

                seg_map, boxes, scores, masks_np = _prepare_segmentation_assets(frame_rgb, processed_outputs)
                pinhole_path: Path = parent_log_path / "exo" / cam_params.name / "pinhole"

                # Log 2D outputs for each camera (video is logged as asset, just overlays here)
                cam_timestamp_ns: int | None = None
                if cam_params.name in video_timestamps and frame_idx < len(video_timestamps[cam_params.name]):
                    cam_timestamp_ns = int(video_timestamps[cam_params.name][frame_idx])
                _log_camera_outputs(
                    pinhole_path=str(pinhole_path),
                    frame_idx=frame_idx,
                    seg_map=seg_map,
                    boxes=boxes,
                    scores=scores,
                    timestamp_ns=cam_timestamp_ns,
                )

                # Run body estimation (single person assumption: take first detection)
                if boxes is not None and masks_np is not None and len(boxes) > 0:
                    k_raw_body = cam_params.intrinsics.k_matrix
                    if k_raw_body is not None:
                        K_33: Float32[ndarray, "3 3"] = k_raw_body.astype(np.float32)
                        pred_list: list[FinalPosePrediction] = body_estimator.process_one_image(
                            rgb_hw3=frame_rgb,
                            xyxy=boxes[:1],  # Take only first person
                            masks=masks_np[:1],
                            masks_score=scores[:1] if scores is not None else None,
                            K_33=K_33,
                        )

                        if pred_list:
                            pred: FinalPosePrediction = pred_list[0]
                            per_view_predictions.append(pred)

                            # Collect 2D keypoints with confidence=1 for detected keypoints
                            kpts_2d: Float32[ndarray, "n_kpts 2"] = pred.pred_keypoints_2d.astype(np.float32)
                            n_kpts: int = kpts_2d.shape[0]
                            conf: Float32[ndarray, "n_kpts 1"] = np.ones((n_kpts, 1), dtype=np.float32)
                            kpts_2d_with_conf: Float32[ndarray, "n_kpts 3"] = np.concatenate([kpts_2d, conf], axis=-1)
                            per_view_keypoints_2d.append(kpts_2d_with_conf)

                            # Also log 2D keypoints
                            rr.set_time("frame_idx", sequence=frame_idx)
                            rr.log(
                                f"{pinhole_path}/keypoints_2d",
                                rr.Points2D(
                                    positions=kpts_2d,
                                    keypoint_ids=MHR70_IDS,
                                    class_ids=0,
                                    colors=(0, 255, 0),
                                ),
                            )

            # ===== Phase 2: Multiview optimization =====
            t_phase1_end: float = time.perf_counter()
            time_inference += t_phase1_end - t_phase1_start

            if optimizer is not None and len(per_view_predictions) >= 2:
                # Stack 2D keypoints: [n_views, n_kpts, 3]
                keypoints_2d_stack: Float32[ndarray, "n_views n_kpts 3"] = np.stack(per_view_keypoints_2d, axis=0)

                # Triangulation timing
                t_triang_start: float = time.perf_counter()

                # Run optimizer (need to enable grad for optimization inside inference_mode context)
                t_opt_start: float = time.perf_counter()
                with torch.enable_grad():
                    result = optimizer.optimize_frame(
                        initial_predictions=per_view_predictions,
                        keypoints_2d_per_view=keypoints_2d_stack,
                    )
                t_opt_end: float = time.perf_counter()
                time_triangulation += t_opt_start - t_triang_start  # includes stacking
                time_optimization += t_opt_end - t_opt_start

                # Get fused mesh using first prediction as base + optimized translation
                base_pred: FinalPosePrediction = per_view_predictions[0]

                # Transform mesh to world coordinates using first camera's transform
                assert first_world_T_cam is not None  # We are in multiview mode

                # Get mesh vertices from mhr_forward (same FK as validation keypoints)
                mhr_head = body_estimator.model.head_pose
                with torch.no_grad():
                    # Prepare optimized params
                    opt_global_rot_mesh: Float32[Tensor, "1 3"] = torch.from_numpy(
                        result.global_rot[np.newaxis, :].astype(np.float32)
                    ).to(device)
                    opt_body_pose_mesh: Float32[Tensor, "1 133"] = torch.from_numpy(
                        result.body_pose[np.newaxis, :].astype(np.float32)
                    ).to(device)
                    shape_params_mesh: Float32[Tensor, "1 45"] = torch.from_numpy(
                        base_pred.shape_params[np.newaxis, :].astype(np.float32)
                    ).to(device)
                    scale_params_mesh: Float32[Tensor, "1 28"] = torch.from_numpy(
                        base_pred.scale_params[np.newaxis, :].astype(np.float32)
                    ).to(device)
                    hand_params_mesh: Float32[Tensor, "1 108"] = torch.from_numpy(
                        base_pred.hand_pose_params[np.newaxis, :].astype(np.float32)
                    ).to(device)
                    expr_params_mesh: Float32[Tensor, "1 72"] = torch.from_numpy(
                        base_pred.expr_params[np.newaxis, :].astype(np.float32)
                    ).to(device)
                    zero_trans_mesh: Float32[Tensor, "1 3"] = torch.zeros(1, 3, device=device, dtype=torch.float32)

                    # Run mhr_forward for vertices
                    mhr_mesh_output = mhr_head.mhr_forward(
                        global_trans=zero_trans_mesh,
                        global_rot=opt_global_rot_mesh,
                        body_pose_params=opt_body_pose_mesh,
                        hand_pose_params=hand_params_mesh,
                        scale_params=scale_params_mesh,
                        shape_params=shape_params_mesh,
                        expr_params=expr_params_mesh,
                        return_keypoints=False,  # Just vertices
                    )
                    verts_fk: Float32[Tensor, "1 n_verts 3"] = mhr_mesh_output

                    # Apply Y/Z flip (same as keypoints)
                    verts_fk = verts_fk.clone()
                    verts_fk[..., 1] *= -1
                    verts_fk[..., 2] *= -1

                    # Add translation after FK
                    opt_trans_mesh: Float32[Tensor, "3"] = torch.from_numpy(
                        result.translation.astype(np.float32)
                    ).to(device)
                    verts_cam_t: Float32[Tensor, "n_verts 3"] = verts_fk.squeeze(0) + opt_trans_mesh

                # Convert to numpy and transform to world
                verts_cam_np: Float32[ndarray, "n_verts 3"] = verts_cam_t.cpu().numpy()
                verts_hom: Float32[ndarray, "n_verts 4"] = np.concatenate(
                    [verts_cam_np, np.ones((verts_cam_np.shape[0], 1), dtype=np.float32)], axis=1
                )
                verts_world: Float32[ndarray, "n_verts 3"] = (verts_hom @ first_world_T_cam.T)[:, :3]

                # Log fused mesh
                rr.set_time("frame_idx", sequence=frame_idx)
                faces_int: Int[ndarray, "n_faces 3"] = np.ascontiguousarray(body_estimator.faces, dtype=np.int32)
                vertex_normals: Float32[ndarray, "n_verts 3"] = compute_vertex_normals(verts_world, faces_int)

                rr.log(
                    "/world/pred/fused/mesh",
                    rr.Mesh3D(
                        vertex_positions=verts_world,
                        triangle_indices=faces_int,
                        vertex_normals=vertex_normals,
                        albedo_factor=(0.2, 0.6, 1.0, 0.8),  # Blue for fused mesh
                    ),
                )

                # Log triangulated 3D keypoints
                triangulated_kpts: Float32[ndarray, "n_kpts 3"] = result.triangulated_kpts_3d[:, :3]
                rr.log(
                    "/world/pred/fused/kpts3d",
                    rr.Points3D(
                        positions=triangulated_kpts,
                        keypoint_ids=MHR70_IDS,
                        class_ids=0,
                        colors=(255, 128, 0),  # Orange for triangulated
                        radii=0.015,
                    ),
                )

                # ===== Phase 3: Validation =====
                # Project triangulated keypoints back to 2D and compute errors
                triangulated_conf: Float32[ndarray, "n_kpts"] = result.triangulated_kpts_3d[:, 3]

                # Build homogeneous coordinates for triangulated keypoints
                ones_kpts: Float32[ndarray, "n_kpts 1"] = np.ones((triangulated_kpts.shape[0], 1), dtype=np.float32)
                triangulated_hom: Float32[ndarray, "n_kpts 4"] = np.concatenate([triangulated_kpts, ones_kpts], axis=-1)

                # Project to each view: P @ X_hom -> uv_hom
                n_kpts: int = triangulated_kpts.shape[0]
                projected_triangulated: Float32[ndarray, "n_views n_kpts 2"] = np.zeros(
                    (n_views, n_kpts, 2), dtype=np.float32
                )
                for view_idx in range(n_views):
                    P_view: Float32[ndarray, "3 4"] = projection_matrices[view_idx]
                    uv_hom: Float32[ndarray, "n_kpts 3"] = (P_view @ triangulated_hom.T).T
                    uv: Float32[ndarray, "n_kpts 2"] = uv_hom[:, :2] / (uv_hom[:, 2:3] + 1e-8)
                    projected_triangulated[view_idx] = uv

                # Compare against detected 2D keypoints
                target_2d: Float32[ndarray, "n_views n_kpts 2"] = keypoints_2d_stack[:, :, :2]
                confidence_2d: Float32[ndarray, "n_views n_kpts"] = keypoints_2d_stack[:, :, 2]

                # Expand triangulated confidence to all views
                conf_expanded: Float32[ndarray, "n_views n_kpts"] = np.broadcast_to(
                    triangulated_conf[np.newaxis, :], (n_views, n_kpts)
                ).astype(np.float32)
                combined_conf: Float32[ndarray, "n_views n_kpts"] = conf_expanded * confidence_2d

                # Validate triangulated keypoints
                val_triangulated = validate_reprojection(
                    projected_2d=projected_triangulated,
                    target_2d=target_2d,
                    confidence=combined_conf,
                )

                # Log validation metrics
                rr.log("/metrics/reprojection/triangulated/mean_error", rr.Scalars(val_triangulated.mean_error_px))
                rr.log("/metrics/reprojection/triangulated/p90_error", rr.Scalars(val_triangulated.p90_error_px))

                # ===== Validate MHR mesh keypoints (per-view self-projection) =====
                # Each view's MHR prediction is independent. To validate:
                # Project each view's pred_keypoints_3d + pred_cam_t with K to get projected 2D
                # Compare to that view's pred_keypoints_2d
                projected_mhr: Float32[ndarray, "n_views n_kpts 2"] = np.zeros(
                    (n_views, base_pred.pred_keypoints_3d.shape[0], 2), dtype=np.float32
                )

                for view_idx, pred in enumerate(per_view_predictions):
                    # Get 3D keypoints + cam_t for this view
                    kpts_3d: Float32[ndarray, "n_kpts 3"] = pred.pred_keypoints_3d.astype(np.float32)
                    cam_t: Float32[ndarray, "3"] = pred.pred_cam_t.astype(np.float32)
                    kpts_cam: Float32[ndarray, "n_kpts 3"] = kpts_3d + cam_t

                    # Get K for this view
                    cam_idx_for_view: int = valid_cam_indices[view_idx]
                    cam_params_for_view = sample.exo_cam_params_list[cam_idx_for_view]
                    if cam_params_for_view is not None and cam_params_for_view.intrinsics.k_matrix is not None:
                        K_view: Float32[ndarray, "3 3"] = cam_params_for_view.intrinsics.k_matrix.astype(np.float32)

                        # Project with K: uv = K @ xyz / z
                        for k in range(kpts_cam.shape[0]):
                            xyz: Float32[ndarray, "3"] = kpts_cam[k]
                            uv_hom: Float32[ndarray, "3"] = K_view @ xyz
                            if abs(uv_hom[2]) > 1e-6:
                                projected_mhr[view_idx, k] = uv_hom[:2] / uv_hom[2]
                            else:
                                projected_mhr[view_idx, k] = np.array([np.nan, np.nan], dtype=np.float32)

                # target_2d is each view's pred_keypoints_2d (already collected)
                # Validate MHR keypoints (per-view)
                val_mhr = validate_reprojection(
                    projected_2d=projected_mhr,
                    target_2d=target_2d,
                    confidence=confidence_2d,
                )

                rr.log("/metrics/reprojection/mhr_perview/mean_error", rr.Scalars(val_mhr.mean_error_px))

                # ===== Validate MHR multiview consistency (world-space projection) =====
                # Use mhr_forward with optimized parameters to get keypoints
                # This ensures consistency with the optimizer's FK
                assert first_world_T_cam is not None  # We are in multiview mode

                mhr_head = body_estimator.model.head_pose
                with torch.no_grad():
                    # Prepare optimized params as batched tensors
                    opt_global_rot: Float32[Tensor, "1 3"] = torch.from_numpy(
                        result.global_rot[np.newaxis, :].astype(np.float32)
                    ).to(device)
                    opt_translation: Float32[Tensor, "1 3"] = torch.from_numpy(
                        result.translation[np.newaxis, :].astype(np.float32)
                    ).to(device)
                    opt_body_pose: Float32[Tensor, "1 133"] = torch.from_numpy(
                        result.body_pose[np.newaxis, :].astype(np.float32)
                    ).to(device)

                    # Fixed params from base_pred
                    shape_params_v: Float32[Tensor, "1 45"] = torch.from_numpy(
                        base_pred.shape_params[np.newaxis, :].astype(np.float32)
                    ).to(device)
                    scale_params_v: Float32[Tensor, "1 28"] = torch.from_numpy(
                        base_pred.scale_params[np.newaxis, :].astype(np.float32)
                    ).to(device)
                    hand_params_v: Float32[Tensor, "1 108"] = torch.from_numpy(
                        base_pred.hand_pose_params[np.newaxis, :].astype(np.float32)
                    ).to(device)
                    expr_params_v: Float32[Tensor, "1 72"] = torch.from_numpy(
                        base_pred.expr_params[np.newaxis, :].astype(np.float32)
                    ).to(device)

                    # Run mhr_forward with optimized params
                    # NOTE: mhr_forward uses global_trans=0 internally, pred_cam_t applied post-FK
                    zero_trans_v: Float32[Tensor, "1 3"] = torch.zeros(1, 3, device=device, dtype=torch.float32)
                    mhr_output_v = mhr_head.mhr_forward(
                        global_trans=zero_trans_v,  # Always zero
                        global_rot=opt_global_rot,
                        body_pose_params=opt_body_pose,
                        hand_pose_params=hand_params_v,
                        scale_params=scale_params_v,
                        shape_params=shape_params_v,
                        expr_params=expr_params_v,
                        return_keypoints=True,
                    )
                    # mhr_output_v = (vertices, keypoints_308)
                    keypoints_308_v: Float32[Tensor, "1 308 3"] = mhr_output_v[1]
                    keypoints_70: Float32[Tensor, "1 70 3"] = keypoints_308_v[:, :70, :]

                    # Apply camera flip
                    keypoints_cam: Float32[Tensor, "70 3"] = keypoints_70.squeeze(0).clone()
                    keypoints_cam[..., 1] *= -1
                    keypoints_cam[..., 2] *= -1

                    # Apply camera-space translation AFTER FK (same as network forward)
                    keypoints_cam = keypoints_cam + opt_translation.squeeze(0)

                # Convert to numpy
                kpts_cam_val: Float32[ndarray, "70 3"] = keypoints_cam.cpu().numpy()

                # Transform to world using first camera's extrinsics
                ones_kpts_mhr: Float32[ndarray, "70 1"] = np.ones((70, 1), dtype=np.float32)
                kpts_cam_hom: Float32[ndarray, "70 4"] = np.concatenate([kpts_cam_val, ones_kpts_mhr], axis=-1)
                mhr_world: Float32[ndarray, "70 3"] = (kpts_cam_hom @ first_world_T_cam.T)[:, :3]

                # Project to each view using full P matrix (world -> 2D)
                mhr_world_hom: Float32[ndarray, "n_kpts 4"] = np.concatenate(
                    [mhr_world, np.ones((mhr_world.shape[0], 1), dtype=np.float32)], axis=-1
                )
                projected_mhr_world: Float32[ndarray, "n_views n_kpts 2"] = np.zeros(
                    (n_views, mhr_world.shape[0], 2), dtype=np.float32
                )
                for view_idx in range(n_views):
                    P_view: Float32[ndarray, "3 4"] = projection_matrices[view_idx]
                    uv_hom_world: Float32[ndarray, "n_kpts 3"] = (P_view @ mhr_world_hom.T).T
                    uv_world: Float32[ndarray, "n_kpts 2"] = uv_hom_world[:, :2] / (uv_hom_world[:, 2:3] + 1e-8)
                    projected_mhr_world[view_idx] = uv_world

                # Validate multiview consistency
                val_mhr_world = validate_reprojection(
                    projected_2d=projected_mhr_world,
                    target_2d=target_2d,
                    confidence=confidence_2d,
                )

                rr.log("/metrics/reprojection/mhr_world/mean_error", rr.Scalars(val_mhr_world.mean_error_px))

                # Compute 3D distance between MHR and triangulated keypoints
                # This tells us directly how far the FK output is from triangulated ground truth
                dist_3d: Float32[ndarray, "n_kpts"] = np.linalg.norm(mhr_world - triangulated_kpts, axis=1)
                mean_3d_dist: float = float(np.mean(dist_3d))
                rr.log("/metrics/3d_distance/mhr_vs_triangulated", rr.Scalars(mean_3d_dist))

                # ===== Debug visualization: log keypoints to each 2D view =====
                # Log MHR 3D keypoints from mhr_forward (cyan, to compare with triangulated orange)
                rr.log(
                    "/world/pred/fused/mhr_kpts3d",
                    rr.Points3D(
                        positions=mhr_world,
                        keypoint_ids=MHR70_IDS,
                        class_ids=0,
                        colors=(0, 255, 255),  # Cyan for MHR FK
                        radii=0.012,
                    ),
                )

                for view_idx, cam_name in enumerate(valid_cam_names):
                    debug_path: str = f"/world/exo/{cam_name}/pinhole/debug"

                    # Detected 2D keypoints (green, class_id=1000)
                    detected_uv: Float32[ndarray, "n_kpts 2"] = target_2d[view_idx]
                    rr.log(
                        f"{debug_path}/detected",
                        rr.Points2D(positions=detected_uv, class_ids=1000, radii=4),
                    )

                    # Projected triangulated (orange, class_id=1001)
                    triang_uv: Float32[ndarray, "n_kpts 2"] = projected_triangulated[view_idx]
                    rr.log(
                        f"{debug_path}/triangulated",
                        rr.Points2D(positions=triang_uv, class_ids=1001, radii=3),
                    )

                    # Projected MHR per-view (blue, class_id=1002)
                    mhr_uv: Float32[ndarray, "n_kpts 2"] = projected_mhr[view_idx]
                    rr.log(
                        f"{debug_path}/mhr_perview",
                        rr.Points2D(positions=mhr_uv, class_ids=1002, radii=3),
                    )

                    # Projected MHR world-optimized (cyan, class_id=1003)
                    mhr_world_uv: Float32[ndarray, "n_kpts 2"] = projected_mhr_world[view_idx]
                    rr.log(
                        f"{debug_path}/mhr_world",
                        rr.Points2D(positions=mhr_world_uv, class_ids=1003, radii=3),
                    )

                # Print summary for this frame
                if frame_idx % 10 == 0 or frame_idx == total_frames - 1:
                    print(f"  Frame {frame_idx}: triangulated   {val_triangulated}")
                    print(f"  Frame {frame_idx}: MHR per-view   {val_mhr}")
                    print(f"  Frame {frame_idx}: MHR world      {val_mhr_world}")
                    print(f"  Frame {frame_idx}: 3D dist (MHR vs triang) = {mean_3d_dist:.3f}m")

                    # Per-view triangulated error breakdown
                    view_errors: list[str] = []
                    for v_idx in range(n_views):
                        view_err: float = val_triangulated.per_view_mean_errors[v_idx] if v_idx < len(val_triangulated.per_view_mean_errors) else float("nan")
                        view_errors.append(f"v{v_idx}={view_err:.1f}")
                    print(f"  [DEBUG] triangulated per-view: {', '.join(view_errors)}")

                    # MHR world per-view error breakdown
                    mhr_world_errors: list[str] = []
                    for v_idx in range(n_views):
                        mhr_err: float = val_mhr_world.per_view_mean_errors[v_idx] if v_idx < len(val_mhr_world.per_view_mean_errors) else float("nan")
                        mhr_world_errors.append(f"v{v_idx}={mhr_err:.1f}")
                    print(f"  [DEBUG] MHR world per-view: {', '.join(mhr_world_errors)}")

                    # Compute filtered metrics (exclude views with error > 100px)
                    OUTLIER_THRESHOLD: float = 100.0
                    good_view_indices: list[int] = [
                        v_idx for v_idx in range(n_views)
                        if val_mhr_world.per_view_mean_errors[v_idx] < OUTLIER_THRESHOLD
                    ]
                    if good_view_indices:
                        good_errors: list[float] = [val_mhr_world.per_view_mean_errors[v_idx] for v_idx in good_view_indices]
                        filtered_mean: float = float(np.mean(good_errors))
                        print(f"  Frame {frame_idx}: MHR world (filtered, {len(good_view_indices)} views) mean={filtered_mean:.1f}px")

                    # Diagnostic: check MHR self-consistency for first view
                    # pred.pred_keypoints_3d + pred_cam_t projected with K should match pred.pred_keypoints_2d
                    if len(per_view_predictions) > 0:
                        test_pred: FinalPosePrediction = per_view_predictions[0]
                        test_kpts_3d: Float32[ndarray, "n_kpts 3"] = test_pred.pred_keypoints_3d.astype(np.float32)
                        test_cam_t: Float32[ndarray, "3"] = test_pred.pred_cam_t.astype(np.float32)
                        test_kpts_cam: Float32[ndarray, "n_kpts 3"] = test_kpts_3d + test_cam_t

                        # Get K for first view from first sample's cam params
                        first_cam_idx: int = valid_cam_indices[0]
                        first_cam_params = sample.exo_cam_params_list[first_cam_idx]
                        if first_cam_params is not None and first_cam_params.intrinsics.k_matrix is not None:
                            test_K: Float32[ndarray, "3 3"] = first_cam_params.intrinsics.k_matrix.astype(np.float32)

                            # Project: uv = K @ xyz / z
                            projected_test: Float32[ndarray, "n_kpts 2"] = np.zeros((test_kpts_cam.shape[0], 2), dtype=np.float32)
                            for k in range(test_kpts_cam.shape[0]):
                                xyz: Float32[ndarray, "3"] = test_kpts_cam[k]
                                uv_hom: Float32[ndarray, "3"] = test_K @ xyz
                                if abs(uv_hom[2]) > 1e-6:
                                    projected_test[k] = uv_hom[:2] / uv_hom[2]
                                else:
                                    projected_test[k] = np.array([np.nan, np.nan], dtype=np.float32)

                            # Compare to MHR's own 2D keypoints
                            mhr_2d: Float32[ndarray, "n_kpts 2"] = test_pred.pred_keypoints_2d.astype(np.float32)
                            self_error: Float32[ndarray, "n_kpts"] = np.linalg.norm(projected_test - mhr_2d, axis=-1).astype(np.float32)
                            mean_self_err: float = float(np.nanmean(self_error))
                            print(f"  [DEBUG] MHR self-proj error (view 0): mean={mean_self_err:.1f}px")

            elif len(per_view_predictions) > 0:
                # Fallback: log first camera's prediction if not enough views
                pred: FinalPosePrediction = per_view_predictions[0]
                world_T_cam = world_T_cam_list[0] if world_T_cam_list else np.eye(4, dtype=np.float32)

                verts_cam: Float32[ndarray, "n_verts 3"] = pred.pred_vertices.astype(np.float32)
                cam_t: Float32[ndarray, "3"] = pred.pred_cam_t.astype(np.float32)
                verts_cam_translated: Float32[ndarray, "n_verts 3"] = verts_cam + cam_t
                verts_hom: Float32[ndarray, "n_verts 4"] = np.concatenate(
                    [verts_cam_translated, np.ones((verts_cam_translated.shape[0], 1), dtype=np.float32)], axis=1
                )
                verts_world: Float32[ndarray, "n_verts 3"] = (verts_hom @ world_T_cam.T)[:, :3]

                rr.set_time("frame_idx", sequence=frame_idx)
                faces_int: Int[ndarray, "n_faces 3"] = np.ascontiguousarray(body_estimator.faces, dtype=np.int32)
                vertex_normals: Float32[ndarray, "n_verts 3"] = compute_vertex_normals(verts_world, faces_int)

                rr.log(
                    "/world/pred/fallback/mesh",
                    rr.Mesh3D(
                        vertex_positions=verts_world,
                        triangle_indices=faces_int,
                        vertex_normals=vertex_normals,
                        albedo_factor=(0.8, 0.4, 0.2, 0.6),  # Orange for fallback
                    ),
                )

    # Print timing summary
    total_time: float = time_inference + time_triangulation + time_optimization + time_validation + time_logging
    print(f"\n[timing] Total: {total_time:.2f}s for {total_frames} frames ({total_frames/total_time:.1f} FPS)")
    print(f"  Inference:     {time_inference:.2f}s ({time_inference/total_time*100:.1f}%)")
    print(f"  Triangulation: {time_triangulation:.2f}s ({time_triangulation/total_time*100:.1f}%)")
    print(f"  Optimization:  {time_optimization:.2f}s ({time_optimization/total_time*100:.1f}%)")
    print(f"  Validation:    {time_validation:.2f}s ({time_validation/total_time*100:.1f}%)")
    print(f"  Logging:       {time_logging:.2f}s ({time_logging/total_time*100:.1f}%)")
    print(f"[done] Processed {total_frames} frames with multiview optimization across {n_views} cameras.")


__all__ = [
    "Sam3MVBodyModelConfig",
    "Sam3MVBodyDemoConfig",
    "main",
]
