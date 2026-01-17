"""
Text-prompt SAM3 video demo with batch-mode inference.

This module provides video segmentation using SAM3's propagate_in_video_iterator:
- Loads entire video into memory for batch processing.
- Uses SAM3's text-prompt interface for zero-shot object detection and tracking.
- Logs results to Rerun for interactive visualization.

For memory-constrained scenarios with long videos, use demo_stream.py instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from jaxtyping import Bool, Float32, Int, UInt8, UInt16
from numpy import ndarray
from simplecv.rerun_log_utils import RerunTyroConfig, log_video
from tqdm import tqdm
from transformers import Sam3VideoConfig, Sam3VideoModel, Sam3VideoProcessor

from sam3d_body.api.visualization import BOX_PALETTE, SEG_OVERLAY_ALPHA

if TYPE_CHECKING:
    from transformers.models.sam3_video.processing_sam3_video import Sam3VideoInferenceSession

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

BOX_DRAW_ORDER: int = 6
"""Rerun draw order for bounding boxes (above segmentation)."""

MAX_VIDEO_MEMORY_GB: float = 4.0
"""Maximum estimated raw video memory (GB) for batch loading. Larger videos should use demo_stream.py."""


# ──────────────────────────────────────────────────────────────────────────────
# Video Checks
# ──────────────────────────────────────────────────────────────────────────────


def check_video_constraints(video_path: Path, max_frames: int | None = None) -> None:
    """
    Check if video is too large for batch loading based on estimated memory usage.

    Memory is estimated as: frames × width × height × 3 bytes (RGB uint8).

    Args:
        video_path: Path to the video file.
        max_frames: Optional cap on frame count (uses this for memory estimate if set).

    Raises:
        RuntimeError: If video cannot be opened.
        MemoryError: If estimated video memory exceeds MAX_VIDEO_MEMORY_GB.
    """
    cap: cv2.VideoCapture = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps: float = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Use max_frames if specified
    effective_frames: int = min(total_frames, max_frames) if max_frames else total_frames
    duration_seconds: float = total_frames / fps

    # Estimate raw memory: frames × width × height × 3 (RGB uint8)
    bytes_per_frame: int = width * height * 3
    estimated_bytes: int = bytes_per_frame * effective_frames
    estimated_gb: float = estimated_bytes / (1024**3)

    if estimated_gb > MAX_VIDEO_MEMORY_GB:
        raise MemoryError(
            f"Video too large for batch loading.\n"
            f"  Resolution:  {width}x{height}\n"
            f"  Duration:    {duration_seconds:.1f}s ({effective_frames} frames)\n"
            f"  Est. memory: {estimated_gb:.1f} GB (max: {MAX_VIDEO_MEMORY_GB:.0f} GB)\n"
            f"  Suggestion:  Use demo_stream.py for streaming inference (constant memory),\n"
            f"               or use --max-frames to limit frames loaded."
        )

    print(f"[Video check] {width}x{height}, {duration_seconds:.1f}s, {effective_frames} frames (~{estimated_gb:.1f} GB) ✓")


# ──────────────────────────────────────────────────────────────────────────────
# Configuration Dataclasses
# ──────────────────────────────────────────────────────────────────────────────

DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}
"""Mapping from string dtype names to torch.dtype."""


def safe_dtype(dtype_pref: str, device: torch.device) -> torch.dtype:
    """Return dtype, falling back to float32 on CPU for half-precision types."""
    dtype: torch.dtype = DTYPE_MAP[dtype_pref]
    if device.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


@dataclass(slots=True)
class Sam3VideoModelConfig:
    """
    Checkpoint, device, and dtype settings for the SAM3 video model.

    Controls compute placement and detection thresholds for batch inference.
    """

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
class Sam3VideoDemoConfig:
    """
    CLI options for running text-prompt SAM3 video segmentation with batch inference.

    This is the top-level configuration passed to ``main()``.
    """

    video_path: Path = Path()
    """Path to the input video file (any format supported by transformers ``load_video``)."""

    prompt: str = "person"
    """Text concept to detect and track across the video."""

    max_frames: int | None = None
    """Optional cap on the number of frames to load and propagate."""

    rr_config: RerunTyroConfig = field(default_factory=RerunTyroConfig)
    """Viewer/runtime options for Rerun (window layout, recording, etc.)."""

    model_config: Sam3VideoModelConfig = field(default_factory=Sam3VideoModelConfig)
    """Checkpoint, device, and dtype settings for the SAM3 video model."""


# ──────────────────────────────────────────────────────────────────────────────
# Rerun Visualization Helpers
# ──────────────────────────────────────────────────────────────────────────────


def log_annotation_context() -> None:
    """
    Register annotation context with background + instance color palette.

    This sets up class IDs for segmentation masks and bounding boxes,
    using colors from ``BOX_PALETTE``.
    """
    class_descriptions: list[rr.ClassDescription] = [
        rr.ClassDescription(
            info=rr.AnnotationInfo(id=0, label="Background", color=(64, 64, 64, 0))
        )
    ]
    for idx, color_rgb in enumerate(BOX_PALETTE[:, :3].tolist(), start=1):
        color_rgba: tuple[int, int, int, int] = (
            int(color_rgb[0]),
            int(color_rgb[1]),
            int(color_rgb[2]),
            SEG_OVERLAY_ALPHA,
        )
        class_descriptions.append(
            rr.ClassDescription(info=rr.AnnotationInfo(id=idx, label=f"Object-{idx}", color=color_rgba))
        )
    rr.log("/", rr.AnnotationContext(class_descriptions), static=True)


def build_blueprint() -> rrb.Blueprint:
    """
    Create a Rerun blueprint with a 2D view for video + segmentation overlay.

    Returns:
        Blueprint configured to show video, segmentation, and boxes.
    """
    view: rrb.Spatial2DView = rrb.Spatial2DView(
        name="Video + Segmentation",
        contents=["video/raw", "video/segmentation", "video/boxes"],
    )
    return rrb.Blueprint(view, collapse_panels=True)


def to_numpy(array_like: torch.Tensor | np.ndarray | list) -> np.ndarray:
    """
    Convert tensors/lists to numpy array.

    Args:
        array_like: Input tensor, array, or list.

    Returns:
        Numpy array on CPU.
    """
    if isinstance(array_like, torch.Tensor):
        return array_like.detach().cpu().numpy()
    return np.asarray(array_like)


def log_frame_outputs(
    frame_idx: int,
    frame_timestamps_ns: Int[ndarray, "num_frames"],
    processed_outputs: dict,
    frame_shape: tuple[int, int],
) -> None:
    """
    Log segmentation masks and bounding boxes for a video frame to Rerun.

    Args:
        frame_idx: Current frame index (used as timeline sequence).
        frame_timestamps_ns: Array of frame timestamps in nanoseconds from the video asset.
        processed_outputs: Dict containing 'masks', 'boxes', and optionally 'scores'.
        frame_shape: Tuple of (height, width) for the frame.
    """
    # Set time on video_time timeline to match the video asset
    if frame_idx < len(frame_timestamps_ns):
        rr.set_time("video_time", duration=1e-9 * frame_timestamps_ns[frame_idx])
    rr.set_time("frame_idx", sequence=frame_idx)

    raw_masks = processed_outputs.get("masks")
    raw_boxes = processed_outputs.get("boxes")
    raw_scores = processed_outputs.get("scores")

    if raw_masks is None or len(raw_masks) == 0:
        return

    # Process masks
    masks_np: Float32[ndarray, "n h w"] = to_numpy(raw_masks).astype(np.float32, copy=False)
    if masks_np.ndim == 2:
        masks_np = masks_np[None, ...]

    h: int = frame_shape[0]
    w: int = frame_shape[1]
    num_instances: int = masks_np.shape[0]

    # Build per-instance colors (used for boxes)
    colors: UInt8[ndarray, "k 4"] = np.asarray(
        [BOX_PALETTE[idx % BOX_PALETTE.shape[0]] for idx in range(num_instances)],
        dtype=np.uint8,
    )

    # Create segmentation map (use uint8 if ≤255 objects for 50% storage savings)
    if num_instances <= 255:
        seg_map: UInt8[ndarray, "h w"] | UInt16[ndarray, "h w"] = np.zeros((h, w), dtype=np.uint8)
        for idx in range(num_instances):
            mask: Float32[ndarray, "h w"] = masks_np[idx]
            mask_bool: Bool[ndarray, "h w"] = mask >= 0.5
            class_id: int = idx + 1  # Reserve 0 for background
            seg_map = np.where(mask_bool, np.uint8(class_id), seg_map)
    else:
        seg_map = np.zeros((h, w), dtype=np.uint16)
        for idx in range(num_instances):
            mask = masks_np[idx]
            mask_bool = mask >= 0.5
            class_id = idx + 1
            seg_map = np.where(mask_bool, np.uint16(class_id), seg_map)

    # Log segmentation (video is already logged as static asset)
    rr.log(
        "video/segmentation",
        rr.SegmentationImage(seg_map),
    )

    # Log bounding boxes if present
    if raw_boxes is not None:
        boxes_np: Float32[ndarray, "n 4"] = to_numpy(raw_boxes).astype(np.float32, copy=False)
        if boxes_np.ndim == 1:
            boxes_np = boxes_np[None, :]

        class_ids: Int[ndarray, "n"] = np.arange(1, boxes_np.shape[0] + 1, dtype=np.int32)

        labels: list[str] | None = None
        if raw_scores is not None:
            scores_np: Float32[ndarray, "n"] = to_numpy(raw_scores).astype(np.float32, copy=False).reshape(-1)
            labels = [f"{score:.2f}" for score in scores_np.tolist()]

        rr.log(
            "video/boxes",
            rr.Boxes2D(
                array=boxes_np,
                array_format=rr.Box2DFormat.XYXY,
                class_ids=class_ids,
                labels=labels,
                colors=colors[:, :3],
                show_labels=True,
                draw_order=BOX_DRAW_ORDER,
            ),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────────────────────────────────────


def load_sam3_model(
    config: Sam3VideoModelConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Sam3VideoModel, Sam3VideoProcessor]:
    """
    Load SAM3 video model and processor from checkpoint.

    Args:
        config: Model configuration with checkpoint and thresholds.
        device: Target compute device.
        dtype: Target dtype for model weights.

    Returns:
        Tuple of (model, processor) ready for inference.
    """
    config_kwargs: dict[str, float] = {
        "score_threshold_detection": config.score_threshold_detection,
        "new_det_thresh": config.new_det_thresh or config.score_threshold_detection,
    }

    model_cfg: Sam3VideoConfig = Sam3VideoConfig.from_pretrained(
        config.checkpoint, **config_kwargs
    )
    model: Sam3VideoModel = Sam3VideoModel.from_pretrained(
        config.checkpoint, config=model_cfg
    ).to(device=device, dtype=dtype)

    processor: Sam3VideoProcessor = Sam3VideoProcessor.from_pretrained(config.checkpoint)

    return model, processor


# ──────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────────────────────────────


def main(cfg: Sam3VideoDemoConfig) -> None:
    """
    Run text-prompt SAM3 video segmentation using batch inference.

    This function:
    1. Loads the SAM3 model and the entire video into memory.
    2. Initializes an inference session with the video tensor.
    3. Propagates masks through the video using text prompts.
    4. Logs results to Rerun for visualization.

    For long videos, consider using demo_stream.py which has constant memory usage.

    Args:
        cfg: Full configuration including video path, prompt, and model settings.

    Raises:
        FileNotFoundError: If the video file does not exist.
    """
    if not cfg.video_path.exists():
        raise FileNotFoundError(f"Video not found: {cfg.video_path}")

    # Resolve device and dtype
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype: torch.dtype = safe_dtype(cfg.model_config.dtype, device)

    # Load model and processor
    model, processor = load_sam3_model(cfg.model_config, device, dtype)

    # Load video frames into memory
    from transformers.video_utils import load_video

    # Check video constraints (duration + resolution)
    check_video_constraints(cfg.video_path, cfg.max_frames)

    video_frames_np, video_metadata = load_video(
        str(cfg.video_path),
        num_frames=cfg.max_frames,
    )
    video_frames: UInt8[ndarray, "t h w 3"] = video_frames_np  # type: ignore[assignment]
    total_frames: int = video_frames.shape[0]

    # Initialize inference session with full video
    inference_session: Sam3VideoInferenceSession = processor.init_video_session(
        video=video_frames,
        inference_device=device,
        processing_device=cfg.model_config.processing_device,
        video_storage_device=cfg.model_config.video_storage_device,
        dtype=dtype,
    )
    processor.add_text_prompt(inference_session=inference_session, text=cfg.prompt)

    # Setup Rerun visualization
    rr.send_blueprint(build_blueprint())
    log_annotation_context()

    # Log video asset once (much more efficient than per-frame images)
    frame_timestamps_ns: Int[ndarray, "num_frames"] = log_video(
        video_source=cfg.video_path,
        video_log_path=Path("video/raw"),
        timeline="video_time",
    )

    # Run batch inference using propagate_in_video_iterator
    with torch.inference_mode():
        for model_outputs in tqdm(
            model.propagate_in_video_iterator(
                inference_session=inference_session,
                max_frame_num_to_track=cfg.max_frames,
            ),
            total=total_frames,
            desc="Propagating masks",
        ):
            frame_idx: int = int(model_outputs.frame_idx)
            frame_rgb: UInt8[ndarray, "h w 3"] = video_frames[frame_idx]
            processed_outputs = processor.postprocess_outputs(inference_session, model_outputs)
            log_frame_outputs(
                frame_idx,
                frame_timestamps_ns,
                processed_outputs,
                frame_shape=(frame_rgb.shape[0], frame_rgb.shape[1]),
            )

    # Summary
    fps: float | None = getattr(video_metadata, "fps", None)
    fps_msg: str = f" @ {fps:.2f} fps" if fps is not None else ""
    print(f"[done] Processed {total_frames} frames{fps_msg} (prompt='{cfg.prompt}').")


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

__all__: list[str] = [
    "Sam3VideoModelConfig",
    "Sam3VideoDemoConfig",
    "main",
]
