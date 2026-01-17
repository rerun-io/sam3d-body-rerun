"""
Text-prompt SAM3 multiview video demo for HoCap/ExoEgo datasets.

This module provides multiview video segmentation across exocentric cameras:
- Uses a single SAM3 video inference session shared across all cameras.
- For each timestamp, processes all exo cameras sequentially.
- Fuses depth from all cameras into a TSDF volume for 3D mesh reconstruction.
- Logs results to Rerun for interactive visualization.

Usage:
    pixi run python tool/demo_mv_video.py --dataset.exocap.sequence_name <name> --prompt "person"

See Also:
    - demo_chunk_video.py: Single-camera chunked video processing
    - demo_mv_image.py: Multiview image (single frame) demo
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import open3d as o3d
import rerun as rr
import rerun.blueprint as rrb
import torch
from jaxtyping import Bool, Float32, Int, UInt8, UInt16
from numpy import ndarray
from simplecv.configs.exoego_dataset_configs import AnnotatedExoEgoDatasetUnion
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence, ExoEgoSample
from simplecv.ops.tsdf_depth_fuser import Open3DFuser
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole
from tqdm import tqdm
from transformers import Sam3VideoConfig, Sam3VideoModel, Sam3VideoProcessor

from sam3d_body.api.visualization import BOX_PALETTE, SEG_OVERLAY_ALPHA

# ──────────────────────────────────────────────────────────────────────────────
# Configuration Dataclasses
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class Sam3MVVideoModelConfig:
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
    """Optional stricter threshold for new detections (falls back to detection threshold if None)."""


@dataclass(slots=True)
class Sam3MVVideoDemoConfig:
    """CLI options for multiview SAM3 video segmentation with Rerun."""

    dataset: AnnotatedExoEgoDatasetUnion
    """ExoEgo dataset configuration (Tyro-resolvable)."""
    prompt: str = "person"
    """Text concept to segment across all cameras."""
    max_frames: int | None = None
    """Optional cap on the number of frames to process."""
    rr_config: RerunTyroConfig = field(default_factory=RerunTyroConfig)
    """Viewer/runtime options for Rerun (window layout, recording, etc.)."""
    model_config: Sam3MVVideoModelConfig = field(default_factory=Sam3MVVideoModelConfig)
    """Checkpoint, device, and dtype settings for the SAM3 video model."""


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────


def _to_numpy(array_like: torch.Tensor | np.ndarray | list) -> np.ndarray:
    """Convert tensors/lists to numpy for downstream processing."""
    if isinstance(array_like, torch.Tensor):
        return array_like.detach().cpu().numpy()
    return np.asarray(array_like)


# ──────────────────────────────────────────────────────────────────────────────
# Rerun Visualization Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_blueprint(exo_cam_names: list[str], max_exo_views: Literal[4, 8] = 8) -> rrb.Blueprint:
    """
    Create a viewer layout with per-camera 2D views, stacked horizontally.

    Args:
        exo_cam_names: List of exocentric camera names.
        max_exo_views: Maximum number of camera views to show.

    Returns:
        Blueprint configured to show 3D view, text prompt, and camera tabs.
    """
    main_view: rrb.ContainerLike = rrb.Spatial3DView(
        origin="/",
        name="3D View",
        eye_controls=rrb.EyeControls3D(
            kind="Orbital", position=[0.0, 1.5, 1.5], look_target=[0, 0, 0.3], spin_speed=0.5
        ),
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
                                f"{origin_base}/image",
                                f"{origin_base}/segmentation_ids",
                                f"{origin_base}/boxes",
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
    """
    Register annotation context with background + instance color palette.

    This sets up class IDs for segmentation masks and bounding boxes,
    using colors from ``BOX_PALETTE`` with ``SEG_OVERLAY_ALPHA`` opacity.
    """
    class_descriptions: list[rr.ClassDescription] = [
        rr.ClassDescription(info=rr.AnnotationInfo(id=0, label="Background", color=(64, 64, 64, 0)))
    ]
    num_colors: int = BOX_PALETTE.shape[0]
    for idx in range(1, num_colors + 1):
        rgb: UInt8[ndarray, "3"] = BOX_PALETTE[(idx - 1) % num_colors, :3]
        color_rgba: tuple[int, int, int, int] = (
            int(rgb[0]),
            int(rgb[1]),
            int(rgb[2]),
            SEG_OVERLAY_ALPHA,
        )
        class_descriptions.append(
            rr.ClassDescription(info=rr.AnnotationInfo(id=idx, label=f"Object-{idx}", color=color_rgba))
        )
    rr.log("/", rr.AnnotationContext(class_descriptions), static=True)


def _log_cameras(exoego_sequence: BaseExoEgoSequence) -> None:
    """
    Log camera intrinsics/extrinsics following the schema.

    Args:
        exoego_sequence: The ExoEgo sequence containing camera metadata.
    """
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
    UInt16[ndarray, "h w"],
    Float32[ndarray, "n 4"] | None,
    Float32[ndarray, "n"] | None,
]:
    """
    Convert SAM3 outputs into segmentation IDs and boxes.

    Args:
        frame_rgb: Input RGB image for shape reference.
        processed_outputs: Dictionary from processor.postprocess_outputs.

    Returns:
        Tuple of (seg_map, boxes, scores) where seg_map contains class IDs,
        boxes are in XYXY format, and scores are detection confidences.
    """
    raw_masks = processed_outputs.get("masks")
    raw_boxes = processed_outputs.get("boxes")
    raw_scores = processed_outputs.get("scores")

    if raw_masks is None or len(raw_masks) == 0:
        h: int = int(frame_rgb.shape[0])
        w: int = int(frame_rgb.shape[1])
        return (np.zeros((h, w), dtype=np.uint16), None, None)

    masks_np: Float32[ndarray, "n h w"] = _to_numpy(raw_masks).astype(np.float32, copy=False)
    if masks_np.ndim == 2:
        masks_np = masks_np[None, ...]

    h = int(frame_rgb.shape[0])
    w = int(frame_rgb.shape[1])
    seg_map: UInt16[ndarray, "h w"] = np.zeros((h, w), dtype=np.uint16)
    num_instances: int = int(masks_np.shape[0])

    for idx in range(num_instances):
        mask: Float32[ndarray, "h w"] = np.asarray(masks_np[idx], dtype=np.float32)
        mask_bool: Bool[ndarray, "h w"] = mask >= 0.5
        class_id: int = idx + 1
        seg_map = np.where(mask_bool, np.uint16(class_id), seg_map)

    boxes_np: Float32[ndarray, "n 4"] | None = None
    scores_np: Float32[ndarray, "n"] | None = None
    if raw_boxes is not None:
        boxes_np = _to_numpy(raw_boxes).astype(np.float32, copy=False)
        if boxes_np.ndim == 1:
            boxes_np = boxes_np[None, :]
    if raw_scores is not None:
        scores_np = _to_numpy(raw_scores).astype(np.float32, copy=False).reshape(-1)

    return seg_map, boxes_np, scores_np


def _colorize_segmentation(
    seg_map: UInt16[ndarray, "h w"],
    base_rgb: UInt8[ndarray, "h w 3"],
) -> UInt8[ndarray, "h w 3"]:
    """
    Recolor the RGB image using segmentation IDs for TSDF fusion.

    Args:
        seg_map: Segmentation map with class IDs (0 = background).
        base_rgb: Original RGB image.

    Returns:
        RGB image with segmented regions recolored using palette colors.
    """
    fusion_rgb: UInt8[ndarray, "h w 3"] = np.asarray(base_rgb, copy=True)
    unique_ids = np.unique(seg_map)
    for class_id in unique_ids:
        if class_id == 0:
            continue
        mask = seg_map == class_id
        color = BOX_PALETTE[(class_id - 1) % BOX_PALETTE.shape[0], :3]
        fusion_rgb[mask] = color[:3]
    return fusion_rgb


def _log_camera_outputs(
    *,
    pinhole_path: str,
    frame_idx: int,
    frame_bgr: UInt8[ndarray, "h w 3"],
    seg_map: UInt16[ndarray, "h w"],
    boxes: Float32[ndarray, "n 4"] | None,
    scores: Float32[ndarray, "n"] | None,
) -> None:
    """
    Log per-camera image, segmentation, and boxes to Rerun.

    Args:
        pinhole_path: Entity path for the camera's pinhole transform.
        frame_idx: Current frame index for timeline.
        frame_bgr: BGR image from OpenCV.
        seg_map: Segmentation map with class IDs.
        boxes: Optional bounding boxes in XYXY format.
        scores: Optional detection scores.
    """
    rr.set_time("frame_idx", sequence=frame_idx)
    rr.log(f"{pinhole_path}/image", rr.Image(frame_bgr, color_model=rr.ColorModel.BGR).compress(jpeg_quality=85))
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


# ──────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────────────────────────────


def main(cfg: Sam3MVVideoDemoConfig) -> None:
    """
    Run text-prompt SAM3 video segmentation across multiview cameras.

    This function:
    1. Loads the ExoEgo dataset and creates a shared SAM3 video session.
    2. For each timestep, processes all exo cameras through the model.
    3. Fuses depth into a TSDF volume and logs visualizations to Rerun.

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

    # Initialize a single inference session for all cameras
    inference_session = processor.init_video_session(
        video=None,
        inference_device=device,
        processing_device=cfg.model_config.processing_device,
        video_storage_device=cfg.model_config.video_storage_device,
        dtype=dtype,
    )
    processor.add_text_prompt(inference_session=inference_session, text=cfg.prompt)

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

    total_frames: int = len(sequence)
    if cfg.max_frames is not None:
        total_frames = min(total_frames, cfg.max_frames)

    with torch.inference_mode():
        for frame_idx in tqdm(range(total_frames), desc="Processing frames"):
            sample: ExoEgoSample = sequence[frame_idx]

            if sample.exo_bgr_list is None or sample.exo_cam_params_list is None:
                continue

            exo_depth_list = sample.exo_depth_list
            exo_cam_param_list = sample.exo_cam_params_list

            # Create a fresh fuser for this timestep
            fuser = Open3DFuser(fusion_resolution=0.01)

            # Process each camera for this timestamp
            for cam_idx, (bgr, cam_params) in enumerate(
                zip(sample.exo_bgr_list, exo_cam_param_list, strict=True)
            ):
                if cam_params is None:
                    continue
                frame_bgr: UInt8[ndarray, "h w 3"] = bgr
                frame_rgb: UInt8[ndarray, "h w 3"] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # Run inference through the shared session
                inputs = processor(images=frame_rgb, device=device, return_tensors="pt")  # type: ignore[misc]  # transformers stub issue
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

                seg_map, boxes, scores = _prepare_segmentation_assets(frame_rgb, processed_outputs)
                pinhole_path: Path = parent_log_path / "exo" / cam_params.name / "pinhole"

                _log_camera_outputs(
                    pinhole_path=str(pinhole_path),
                    frame_idx=frame_idx,
                    frame_bgr=frame_bgr,
                    seg_map=seg_map,
                    boxes=boxes,
                    scores=scores,
                )

                # Fuse this camera's depth into the TSDF volume
                if exo_depth_list is not None:
                    fusion_rgb: UInt8[ndarray, "h w 3"] = _colorize_segmentation(seg_map, frame_rgb)
                    K_33: Float32[ndarray, "3 3"] | None = cam_params.intrinsics.k_matrix
                    if K_33 is not None:
                        fuser.fuse_frames(
                            depth_hw=exo_depth_list[cam_idx],
                            K_33=K_33,
                            cam_T_world_44=cam_params.extrinsics.cam_T_world,
                            rgb_hw3=fusion_rgb,
                        )

            # Extract and log the mesh for this timestep
            if exo_depth_list is not None:
                rr.set_time("frame_idx", sequence=frame_idx)
                gt_mesh: o3d.geometry.TriangleMesh = fuser.get_mesh()
                gt_mesh.compute_vertex_normals()

                vertex_positions: Float32[ndarray, "num_vertices 3"] = np.asarray(gt_mesh.vertices, dtype=np.float32)
                triangle_indices: Int[ndarray, "num_faces 3"] = np.asarray(gt_mesh.triangles, dtype=np.int32)
                vertex_normals: Float32[ndarray, "num_vertices 3"] = np.asarray(gt_mesh.vertex_normals, dtype=np.float32)
                vertex_colors: Float32[ndarray, "num_vertices 3"] = np.asarray(gt_mesh.vertex_colors, dtype=np.float32)

                rr.log(
                    str(parent_log_path / "gt" / "env_mesh"),
                    rr.Mesh3D(
                        vertex_positions=vertex_positions,
                        triangle_indices=triangle_indices,
                        vertex_normals=vertex_normals,
                        vertex_colors=vertex_colors,
                    ),
                )

    print(f"[done] Processed {total_frames} frames across {len(exo_cam_names)} exo cameras (prompt='{cfg.prompt}').")


__all__ = [
    "Sam3MVVideoModelConfig",
    "Sam3MVVideoDemoConfig",
    "main",
]
