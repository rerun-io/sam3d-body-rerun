"""
Text-prompt SAM3 multiview image demo for HoCap/ExoEgo datasets.

This module provides multiview image segmentation on the first frame:
- Loads a single multiview sample from an ExoEgo sequence.
- Runs SAM3 instance segmentation on each exocentric camera.
- Fuses depth from all cameras into a TSDF volume for 3D mesh reconstruction.
- Logs results to Rerun for interactive visualization.

Usage:
    pixi run python tool/demo_mv_image.py --dataset.exocap.sequence_name <name> --prompt "person"

See Also:
    - demo_mv_video.py: Multiview video (temporal) demo
    - demo.py: Single-image 3D pose estimation demo
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
from simplecv.camera_parameters import Fisheye62Parameters, PinholeParameters
from simplecv.configs.exoego_dataset_configs import AnnotatedExoEgoDatasetUnion
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence, ExoEgoSample
from simplecv.ops.tsdf_depth_fuser import Open3DFuser
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole
from transformers import Sam3Model, Sam3Processor

from sam3d_body.api.visualization import BOX_PALETTE, SEG_OVERLAY_ALPHA

# ──────────────────────────────────────────────────────────────────────────────
# Configuration Dataclasses
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class Sam3MVImageModelConfig:
    """Settings for the SAM3 checkpoint and compute placement."""

    checkpoint: str = "facebook/sam3"
    """Model identifier passed to ``Sam3Model.from_pretrained``."""
    device: Literal["auto", "cpu", "cuda"] = "auto"
    """Compute device selection; ``auto`` prefers CUDA when available."""
    dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    """Torch dtype used for model weights and inference."""


@dataclass(slots=True)
class Sam3MVImageDemoConfig:
    """CLI options for multiview SAM3 image segmentation with Rerun."""

    dataset: AnnotatedExoEgoDatasetUnion
    """ExoEgo dataset configuration (Tyro-resolvable)."""
    prompt: str = "person"
    """Text concept to segment across all cameras."""
    score_threshold: float = 0.75
    """Minimum detection score to keep an instance."""
    rr_config: RerunTyroConfig = field(default_factory=RerunTyroConfig)
    """Viewer/runtime options for Rerun (window layout, recording, etc.)."""
    model_config: Sam3MVImageModelConfig = field(default_factory=Sam3MVImageModelConfig)
    """Checkpoint, device, and dtype settings for the SAM3 model."""

# ──────────────────────────────────────────────────────────────────────────────
# SAM3 Image Predictor
# ──────────────────────────────────────────────────────────────────────────────


class _Sam3ImagePredictor:
    """Lightweight wrapper around the SAM3 model for single-image inference."""

    def __init__(self, cfg: Sam3MVImageModelConfig) -> None:
        """Initialize the predictor with model and processor."""
        self.cfg: Sam3MVImageModelConfig = cfg
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype_map: dict[str, torch.dtype] = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        dtype: torch.dtype = dtype_map[cfg.dtype] if device.type == "cuda" else torch.float32
        self._inference_device: torch.device = device  # type: ignore[read-only]  # pyrefly false positive
        self.model = Sam3Model.from_pretrained(cfg.checkpoint).to(device=device, dtype=dtype)
        self.processor = Sam3Processor.from_pretrained(cfg.checkpoint)

    def predict(
        self,
        rgb_hw3: UInt8[ndarray, "h w 3"],
        *,
        text: str,
        score_threshold: float,
    ) -> dict:
        """
        Run SAM3 instance segmentation on one RGB image.

        Args:
            rgb_hw3: Input RGB image (H, W, 3).
            text: Text prompt for the segmentation target.
            score_threshold: Minimum confidence to keep an instance.

        Returns:
            Dictionary with 'masks', 'boxes', 'scores' keys.
        """
        inputs = self.processor(images=rgb_hw3, text=text, return_tensors="pt")  # type: ignore[misc]  # transformers stub issue
        inputs = {k: v.to(self._inference_device) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.model.dtype)

        with torch.no_grad():
            outputs = self.model(**inputs)

        original_sizes = inputs.get("original_sizes")
        target_sizes = original_sizes.tolist() if original_sizes is not None else None  # type: ignore[union-attr]
        processed = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=score_threshold,
            mask_threshold=0.5,
            target_sizes=target_sizes,
        )[0]
        return processed


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
    results: dict,
) -> tuple[
    UInt16[ndarray, "h w"],
    Float32[ndarray, "n 4"] | None,
    Float32[ndarray, "n"] | None,
]:
    """
    Convert SAM3 outputs into segmentation IDs and boxes.

    Args:
        frame_rgb: Input RGB image for shape reference.
        results: Dictionary from predictor.predict() with masks/boxes/scores.

    Returns:
        Tuple of (seg_map, boxes, scores) where seg_map contains class IDs,
        boxes are in XYXY format, and scores are detection confidences.
    """
    raw_masks = results.get("masks")
    raw_boxes = results.get("boxes")
    raw_scores = results.get("scores")

    # Convert tensors to numpy if needed
    if raw_masks is not None and hasattr(raw_masks, "detach"):
        raw_masks = raw_masks.detach().cpu().numpy()
    if raw_boxes is not None and hasattr(raw_boxes, "detach"):
        raw_boxes = raw_boxes.detach().cpu().numpy()
    if raw_scores is not None and hasattr(raw_scores, "detach"):
        raw_scores = raw_scores.detach().cpu().numpy()

    if raw_masks is None or len(raw_masks) == 0:
        h: int = int(frame_rgb.shape[0])
        w: int = int(frame_rgb.shape[1])
        return (np.zeros((h, w), dtype=np.uint16), None, None)

    masks_np: Float32[ndarray, "n h w"] = np.asarray(raw_masks, dtype=np.float32)
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
        boxes_np = np.asarray(raw_boxes, dtype=np.float32)
        if boxes_np.ndim == 1:
            boxes_np = boxes_np[None, :]
    if raw_scores is not None:
        scores_np = np.asarray(raw_scores, dtype=np.float32).reshape(-1)

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
    frame_bgr: UInt8[ndarray, "h w 3"],
    seg_map: UInt16[ndarray, "h w"],
    boxes: Float32[ndarray, "n 4"] | None,
    scores: Float32[ndarray, "n"] | None,
) -> None:
    """
    Log per-camera image, segmentation, and boxes to Rerun.

    Args:
        pinhole_path: Entity path for the camera's pinhole transform.
        frame_bgr: BGR image from OpenCV.
        seg_map: Segmentation map with class IDs.
        boxes: Optional bounding boxes in XYXY format.
        scores: Optional detection scores.
    """
    rr.set_time("frame_idx", sequence=0)
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


def main(cfg: Sam3MVImageDemoConfig) -> None:
    """
    Run text-prompt SAM3 segmentation on the first multiview frame.

    This function:
    1. Loads the first frame from the ExoEgo dataset.
    2. Runs SAM3 instance segmentation on each exo camera.
    3. Fuses depth into a TSDF volume and logs visualizations to Rerun.

    Args:
        cfg: Demo configuration with dataset, prompt, and model settings.

    Raises:
        RuntimeError: If dataset yields zero frames.
        AssertionError: If depth or camera parameters are missing.
    """
    sequence: BaseExoEgoSequence = cfg.dataset.setup()

    if len(sequence) == 0:
        raise RuntimeError("Dataset yielded zero canonical frames.")

    sample: ExoEgoSample = sequence[0]
    predictor = _Sam3ImagePredictor(cfg.model_config)

    parent_log_path: Path = Path("world")

    rr.log("text-prompt", rr.TextDocument(f"## Prompt\n\n{cfg.prompt}", media_type="text/markdown"), static=True)

    exo_cam_names: list[str] = (
        [cam.name for cam in sequence.exo_sequence.exo_cam_list if cam is not None] if sequence.exo_sequence else []
    )
    exo_depth_list: list[UInt16[ndarray, "h w"]] | None = sample.exo_depth_list
    exo_cam_param_list: list[Fisheye62Parameters | PinholeParameters | None] | None = sample.exo_cam_params_list
    assert exo_depth_list is not None, "Exo depth images are required for this demo."
    assert exo_cam_param_list is not None, "Exo camera parameters are required for this demo."

    # Log static data
    rr.send_blueprint(_make_blueprint(exo_cam_names))
    rr.log("/", sequence.world_coordinate_system, static=True)
    _log_annotation_context()
    _log_cameras(sequence)
    fuser = Open3DFuser(fusion_resolution=0.004)

    if sample.exo_bgr_list is not None and sequence.exo_sequence is not None:
        for idx, (bgr, cam_params) in enumerate(zip(sample.exo_bgr_list, exo_cam_param_list, strict=True)):
            if cam_params is None:
                continue
            frame_bgr: UInt8[ndarray, "h w 3"] = bgr
            frame_rgb: UInt8[ndarray, "h w 3"] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            outputs = predictor.predict(frame_rgb, text=cfg.prompt, score_threshold=cfg.score_threshold)
            seg_map, boxes, scores = _prepare_segmentation_assets(frame_rgb, outputs)
            fusion_rgb: UInt8[ndarray, "h w 3"] = _colorize_segmentation(seg_map, frame_rgb)
            pinhole_path: Path = parent_log_path / "exo" / cam_params.name / "pinhole"

            _log_camera_outputs(
                pinhole_path=str(pinhole_path),
                frame_bgr=frame_bgr,
                seg_map=seg_map,
                boxes=boxes,
                scores=scores,
            )

            K_33: Float32[ndarray, "3 3"] | None = cam_params.intrinsics.k_matrix
            assert K_33 is not None, "Pinhole camera intrinsics are required for depth fusion."
            fuser.fuse_frames(
                depth_hw=exo_depth_list[idx],
                K_33=K_33,
                cam_T_world_44=cam_params.extrinsics.cam_T_world,
                rgb_hw3=fusion_rgb,
            )

        # Extract and log the mesh
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

    print(f"[done] Logged first multiview frame with prompt='{cfg.prompt}', {len(exo_cam_names)} exo cams.")


__all__ = [
    "Sam3MVImageModelConfig",
    "Sam3MVImageDemoConfig",
    "main",
]
