"""
Text-prompt SAM3 video demo with chunk-based inference.

This module provides memory-efficient video segmentation by processing in chunks:
- Loads video in chunks of ~30 seconds (configurable based on resolution).
- Uses overlapping boundaries to preserve hotstart quality benefits.
- Uses SAM3's propagate_in_video_iterator within each chunk.
- Logs results to Rerun for interactive visualization.

This is a middle ground between:
- demo_video.py (full video in RAM - best quality, highest memory)
- demo_stream.py (frame-by-frame - lowest memory, reduced temporal context)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple

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

DEFAULT_CHUNK_MEMORY_GB: float = 4.0
"""Default target memory per chunk in GB. Actual chunk size adapts to video resolution."""

HOTSTART_OVERLAP_FRAMES: int = 10
"""Number of overlap frames between chunks to preserve hotstart quality benefits."""


# ──────────────────────────────────────────────────────────────────────────────
# Video Metadata
# ──────────────────────────────────────────────────────────────────────────────


class VideoMetadata(NamedTuple):
    """Video file metadata from OpenCV probe."""

    width: int
    height: int
    fps: float
    total_frames: int
    bytes_per_frame: int


def probe_video(video_path: Path) -> VideoMetadata:
    """
    Probe video file for metadata without loading frames.

    Args:
        video_path: Path to the video file.

    Returns:
        VideoMetadata containing resolution, fps, and frame count.

    Raises:
        RuntimeError: If video cannot be opened.
    """
    cap: cv2.VideoCapture = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps: float = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    bytes_per_frame: int = width * height * 3  # RGB uint8
    return VideoMetadata(width, height, fps, total_frames, bytes_per_frame)


def compute_chunk_size(meta: VideoMetadata, target_memory_gb: float) -> int:
    """
    Compute optimal chunk size in frames based on target memory budget.

    Args:
        meta: Video metadata from probe_video.
        target_memory_gb: Target memory budget per chunk in GB.

    Returns:
        Number of frames per chunk.
    """
    target_bytes: int = int(target_memory_gb * (1024**3))
    chunk_frames: int = max(1, target_bytes // meta.bytes_per_frame)
    return chunk_frames


# ──────────────────────────────────────────────────────────────────────────────
# Chunk Loading
# ──────────────────────────────────────────────────────────────────────────────


def load_video_chunk(
    video_path: Path,
    start_frame: int,
    num_frames: int,
) -> UInt8[ndarray, "t h w 3"]:
    """
    Load a specific chunk of frames from a video file.

    Args:
        video_path: Path to the video file.
        start_frame: Starting frame index (0-based).
        num_frames: Number of frames to load.

    Returns:
        Array of shape (num_frames, height, width, 3) in RGB format.

    Raises:
        RuntimeError: If video cannot be opened or seek fails.
    """
    cap: cv2.VideoCapture = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames: list[UInt8[ndarray, "h w 3"]] = []
    for _ in range(num_frames):
        ret: bool
        frame: UInt8[ndarray, "h w 3"]
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame_rgb: UInt8[ndarray, "h w 3"] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames loaded from {video_path} at position {start_frame}")

    return np.stack(frames, axis=0)


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

    Controls compute placement and detection thresholds for chunk-based inference.
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
class Sam3ChunkVideoDemoConfig:
    """
    CLI options for running text-prompt SAM3 video segmentation with chunk-based inference.

    This is the top-level configuration passed to ``main()``.
    """

    video_path: Path = Path()
    """Path to the input video file (any format supported by OpenCV)."""

    prompt: str = "person"
    """Text concept to detect and track across the video."""

    max_frames: int | None = None
    """Optional cap on total frames to process across all chunks."""

    chunk_memory_gb: float = DEFAULT_CHUNK_MEMORY_GB
    """Target memory per chunk in GB. Chunk size in frames is computed from this and video resolution."""

    overlap_frames: int = HOTSTART_OVERLAP_FRAMES
    """Number of overlap frames between chunks for hotstart quality preservation."""

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


def main(cfg: Sam3ChunkVideoDemoConfig) -> None:
    """
    Run text-prompt SAM3 video segmentation using chunk-based inference.

    This function:
    1. Loads the SAM3 model once.
    2. Probes video to compute optimal chunk size based on memory budget.
    3. Processes video in overlapping chunks using propagate_in_video_iterator.
    4. Logs results to Rerun for visualization.

    Memory usage is O(chunk_size) instead of O(video_length).

    Args:
        cfg: Full configuration including video path, prompt, and chunking settings.

    Raises:
        FileNotFoundError: If the video file does not exist.
    """
    if not cfg.video_path.exists():
        raise FileNotFoundError(f"Video not found: {cfg.video_path}")

    # Probe video metadata
    meta: VideoMetadata = probe_video(cfg.video_path)

    # Compute chunk size based on memory budget
    chunk_frames: int = compute_chunk_size(meta, cfg.chunk_memory_gb)
    chunk_duration: float = chunk_frames / meta.fps

    # Apply max_frames limit
    total_frames: int = meta.total_frames
    if cfg.max_frames is not None:
        total_frames = min(total_frames, cfg.max_frames)

    print(f"[Chunk config] {meta.width}x{meta.height} @ {meta.fps:.1f}fps")
    print(f"[Chunk config] Total: {total_frames} frames, Chunk: {chunk_frames} frames (~{chunk_duration:.1f}s)")
    print(f"[Chunk config] Overlap: {cfg.overlap_frames} frames")

    # Resolve device and dtype
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype: torch.dtype = safe_dtype(cfg.model_config.dtype, device)

    # Load model and processor once
    model, processor = load_sam3_model(cfg.model_config, device, dtype)

    # Setup Rerun visualization
    rr.send_blueprint(build_blueprint())
    log_annotation_context()

    # Log video asset once (much more efficient than per-frame images)
    frame_timestamps_ns: Int[ndarray, "num_frames"] = log_video(
        video_source=cfg.video_path,
        video_log_path=Path("video/raw"),
        timeline="video_time",
    )

    # Compute chunk boundaries with overlap
    # Step size is chunk_frames - overlap to create overlapping windows
    step_size: int = max(1, chunk_frames - cfg.overlap_frames)
    chunk_starts: list[int] = list(range(0, total_frames, step_size))

    # Track which frames we've already yielded (to skip overlap frames)
    frames_yielded: set[int] = set()
    total_processed: int = 0

    pbar: tqdm = tqdm(total=total_frames, desc="Processing chunks")

    with torch.inference_mode():
        for _chunk_idx, chunk_start in enumerate(chunk_starts):
            # Calculate actual chunk end
            chunk_end: int = min(chunk_start + chunk_frames, total_frames)
            actual_chunk_size: int = chunk_end - chunk_start

            if actual_chunk_size <= 0:
                break

            # Load chunk frames
            chunk_frames_np: UInt8[ndarray, "t h w 3"] = load_video_chunk(
                cfg.video_path, chunk_start, actual_chunk_size
            )

            # Initialize inference session for this chunk
            inference_session: Sam3VideoInferenceSession = processor.init_video_session(
                video=chunk_frames_np,
                inference_device=device,
                processing_device=cfg.model_config.processing_device,
                video_storage_device=cfg.model_config.video_storage_device,
                dtype=dtype,
            )
            processor.add_text_prompt(inference_session=inference_session, text=cfg.prompt)

            # Process chunk using propagate_in_video_iterator
            for model_outputs in model.propagate_in_video_iterator(
                inference_session=inference_session,
                max_frame_num_to_track=actual_chunk_size,
            ):
                # Convert chunk-local frame index to global frame index
                local_frame_idx: int = int(model_outputs.frame_idx)
                global_frame_idx: int = chunk_start + local_frame_idx

                # Skip frames we've already yielded (overlap from previous chunk)
                if global_frame_idx in frames_yielded:
                    continue

                # Skip if beyond max_frames
                if global_frame_idx >= total_frames:
                    continue

                frames_yielded.add(global_frame_idx)

                # Get frame shape from chunk
                frame_rgb: UInt8[ndarray, "h w 3"] = chunk_frames_np[local_frame_idx]

                # Postprocess and log
                processed_outputs = processor.postprocess_outputs(inference_session, model_outputs)
                log_frame_outputs(
                    global_frame_idx,
                    frame_timestamps_ns,
                    processed_outputs,
                    frame_shape=(frame_rgb.shape[0], frame_rgb.shape[1]),
                )

                pbar.update(1)
                total_processed += 1

            # Free chunk memory
            del chunk_frames_np
            del inference_session

    pbar.close()

    # Summary
    print(f"[done] Processed {total_processed} frames in {len(chunk_starts)} chunks (prompt='{cfg.prompt}').")


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

__all__: list[str] = [
    "Sam3VideoModelConfig",
    "Sam3ChunkVideoDemoConfig",
    "main",
]
