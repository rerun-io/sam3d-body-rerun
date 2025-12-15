"""Text-prompt SAM3 video demo that streams frames to minimize RAM."""

from dataclasses import dataclass, field
from fractions import Fraction
import warnings
from pathlib import Path
from typing import Literal

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
import subprocess

from sam3d_body.api.visualization import BOX_PALETTE


@dataclass(slots=True)
class Sam3VideoModelConfig:
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
class Sam3StreamDemoConfig:
    """CLI options for running text-prompt SAM3 video segmentation in streaming mode."""

    rr_config: RerunTyroConfig = field(default_factory=RerunTyroConfig)
    """Viewer/runtime options for Rerun (window layout, recording, etc.)."""
    model_config: Sam3VideoModelConfig = field(default_factory=Sam3VideoModelConfig)
    """Checkpoint, device, and dtype settings for the SAM3 video model."""
    video_path: Path = Path()
    """Path to the input video file (any format supported by OpenCV)."""
    prompt: str = "person"
    """Text concept to detect and track across the video."""
    max_frames: int | None = None
    """Optional cap on the number of frames to decode and propagate."""


def _resolve_device(device_pref: Literal["auto", "cpu", "cuda"]) -> torch.device:
    """Pick a torch device respecting user preference."""
    if device_pref == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_pref)


def _resolve_dtype(dtype_pref: Literal["bfloat16", "float16", "float32"], device: torch.device) -> torch.dtype:
    """Choose a safe dtype; fall back to float32 on CPU for fp16/bf16."""
    mapping: dict[str, torch.dtype] = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype: torch.dtype = mapping[dtype_pref]
    if device.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _to_numpy(array_like) -> np.ndarray:
    """Convert tensors/lists to numpy for downstream processing."""
    if isinstance(array_like, torch.Tensor):
        return array_like.detach().cpu().numpy()
    return np.asarray(array_like)


def _log_annotation_context() -> None:
    """Register a simple background + instance palette for masks/boxes."""
    class_descriptions: list[rr.ClassDescription] = [
        rr.ClassDescription(info=rr.AnnotationInfo(id=0, label="Background", color=(64, 64, 64)))
    ]
    for idx, color_rgb in enumerate(BOX_PALETTE[:, :3].tolist(), start=1):
        class_descriptions.append(
            rr.ClassDescription(
                info=rr.AnnotationInfo(id=idx, label=f"Object-{idx}", color=tuple(int(c) for c in color_rgb))
            )
        )
    rr.log("/", rr.AnnotationContext(class_descriptions), static=True)


def _build_blueprint() -> rrb.Blueprint:
    """Create a 2D view showing the video frame, overlay, and boxes."""
    view: rrb.Spatial2DView = rrb.Spatial2DView(
        name="Video + Segmentation",
        contents=[
            "video/raw",
            "video/segmentation_overlay",
            "video/boxes",
        ],
    )
    return rrb.Blueprint(view, collapse_panels=True)


def _log_frame_outputs(
    frame_idx: int,
    frame_rgb: UInt8[ndarray, "h w 3"],
    frame_time_ns: int | None,
    processed_outputs: dict,
) -> None:
    """Log the current frame plus segmentation/boxes to Rerun."""
    if frame_time_ns is not None:
        rr.set_time_nanos("video_time", frame_time_ns)
    else:
        rr.set_time("frame_idx", sequence=frame_idx)

    raw_masks = processed_outputs.get("masks")
    raw_boxes = processed_outputs.get("boxes")
    raw_scores = processed_outputs.get("scores")

    if raw_masks is None or len(raw_masks) == 0:
        return

    masks_np: Float32[ndarray, "n h w"] = _to_numpy(raw_masks).astype(np.float32, copy=False)
    if masks_np.ndim == 2:
        masks_np = masks_np[None, ...]

    h: int = int(frame_rgb.shape[0])
    w: int = int(frame_rgb.shape[1])
    seg_map: UInt16[ndarray, "h w"] = np.zeros((h, w), dtype=np.uint16)
    seg_overlay: UInt8[ndarray, "h w 4"] = np.zeros((h, w, 4), dtype=np.uint8)

    num_instances: int = int(masks_np.shape[0])
    colors: UInt8[ndarray, "k 4"] = np.asarray(
        [BOX_PALETTE[idx % BOX_PALETTE.shape[0]] for idx in range(num_instances)],
        dtype=np.uint8,
    )

    for idx in range(num_instances):
        mask: Float32[ndarray, "h w"] = np.asarray(masks_np[idx], dtype=np.float32)
        mask_bool: Bool[ndarray, "h w"] = mask >= 0.5
        class_id: int = idx + 1  # reserve 0 for background

        seg_map = np.where(mask_bool, np.uint16(class_id), seg_map)

        color: UInt8[ndarray, "4"] = colors[idx]
        seg_overlay[mask_bool] = np.array([color[0], color[1], color[2], 120], dtype=np.uint8)

    rr.log("video/segmentation_ids", rr.SegmentationImage(seg_map, draw_order=1))
    rr.log(
        "video/segmentation_overlay",
        rr.Image(seg_overlay, color_model=rr.ColorModel.RGBA, draw_order=2),
    )

    if raw_boxes is not None:
        boxes_np: Float32[ndarray, "n 4"] = _to_numpy(raw_boxes).astype(np.float32, copy=False)
        if boxes_np.ndim == 1:
            boxes_np = boxes_np[None, :]
        class_ids: Int[ndarray, "n"] = np.arange(1, boxes_np.shape[0] + 1, dtype=np.int32)
        labels: list[str] | None = None
        if raw_scores is not None:
            scores_np: Float32[ndarray, "n"] = _to_numpy(raw_scores).astype(np.float32, copy=False).reshape(-1)
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
                draw_order=3,
            ),
        )


def _probe_video(path: Path) -> tuple[int, int, float | None]:
    """Probe video dimensions and fps via ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate",
        "-of",
        "csv=p=0",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    line = proc.stdout.strip()
    width_s, height_s, fps_s = line.split(",")
    width = int(width_s)
    height = int(height_s)
    fps: float | None
    try:
        fps = float(Fraction(fps_s))
    except Exception:
        fps = None
    return width, height, fps


def _iter_video_frames_ffmpeg(path: Path):
    """Yield frames using ffmpeg pipe (supports AV1 and keeps memory constant)."""
    width, height, fps = _probe_video(path)
    bytes_per_frame = width * height * 3
    cmd = [
        "ffmpeg",
        "-i",
        str(path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-vcodec",
        "rawvideo",
        "-loglevel",
        "error",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        frame_idx = 0
        while True:
            buf = proc.stdout.read(bytes_per_frame)
            if not buf or len(buf) < bytes_per_frame:
                break
            frame_rgb: UInt8[ndarray, "h w 3"] = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 3))
            yield frame_idx, frame_rgb, fps, width, height
            frame_idx += 1
    finally:
        if proc.stdout:
            proc.stdout.close()
        proc.wait()


def _iter_video_frames(path: Path):
    """Yield RGB frames from disk without retaining the full video in memory."""
    cap = cv2.VideoCapture(str(path))
    if cap.isOpened():
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or None
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_idx = 0
        ok, frame_bgr = cap.read()
        if ok:
            try:
                while ok:
                    frame_rgb: UInt8[ndarray, "h w 3"] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    yield frame_idx, frame_rgb, fps, width, height
                    frame_idx += 1
                    ok, frame_bgr = cap.read()
                return
            finally:
                cap.release()
        cap.release()
    # Fallback to ffmpeg pipe (handles codecs OpenCV might miss).
    yield from _iter_video_frames_ffmpeg(path)


def main(cfg: Sam3StreamDemoConfig) -> None:
    """Run text-prompt SAM3 video segmentation using streaming inference."""
    if not cfg.video_path.exists():
        raise FileNotFoundError(f"Video not found: {cfg.video_path}")

    device: torch.device = _resolve_device(cfg.model_config.device)
    dtype: torch.dtype = _resolve_dtype(cfg.model_config.dtype, device)

    frame_iter = _iter_video_frames(cfg.video_path)
    try:
        first = next(frame_iter, None)
        if first is None:
            raise RuntimeError(f"Video contains no frames: {cfg.video_path}")
        first_idx, first_frame, fps, width, height = first

        config_kwargs: dict = {
            "score_threshold_detection": cfg.model_config.score_threshold_detection,
        }
        if cfg.model_config.new_det_thresh is not None:
            config_kwargs["new_det_thresh"] = cfg.model_config.new_det_thresh
        else:
            config_kwargs["new_det_thresh"] = cfg.model_config.score_threshold_detection

        model_cfg = Sam3VideoConfig.from_pretrained(cfg.model_config.checkpoint, **config_kwargs)
        model = Sam3VideoModel.from_pretrained(cfg.model_config.checkpoint, config=model_cfg).to(
            device=device, dtype=dtype
        )
        processor = Sam3VideoProcessor.from_pretrained(cfg.model_config.checkpoint)

        frame_timestamps_ns: Int[ndarray, "num_frames"] | None = None
        try:
            frame_timestamps_ns = log_video(
                video_path=cfg.video_path,
                video_log_path=Path("video/raw"),
                timeline="video_time",
            )
        except Exception as exc:  # log_video is best-effort; continue if it fails
            warnings.warn(f"Failed to log video asset via Rerun: {exc}")

        inference_session = processor.init_video_session(
            video=None,
            inference_device=device,
            processing_device=cfg.model_config.processing_device,
            video_storage_device=cfg.model_config.video_storage_device,
            dtype=dtype,
        )
        processor.add_text_prompt(inference_session=inference_session, text=cfg.prompt)

        rr.send_blueprint(_build_blueprint())
        _log_annotation_context()

        total_frames: int = 0

        def _process_frame(frame_idx: int, frame_rgb: UInt8[ndarray, "h w 3"]) -> None:
            nonlocal total_frames
            inputs = processor(images=frame_rgb, device=device, return_tensors="pt")
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
            frame_time_ns: int | None = None
            if frame_timestamps_ns is not None and frame_idx < frame_timestamps_ns.shape[0]:
                frame_time_ns = int(frame_timestamps_ns[frame_idx])
            _log_frame_outputs(
                frame_idx=frame_idx,
                frame_rgb=frame_rgb,
                frame_time_ns=frame_time_ns,
                processed_outputs=processed_outputs,
            )
            total_frames += 1

        with torch.inference_mode():
            progress_total = cfg.max_frames if cfg.max_frames is not None else None
            pbar = tqdm(total=progress_total, desc="Streaming masks")

            _process_frame(first_idx, first_frame)
            pbar.update(1)

            for frame_idx, frame_rgb, *_ in frame_iter:
                if cfg.max_frames is not None and frame_idx >= cfg.max_frames:
                    break
                _process_frame(frame_idx, frame_rgb)
                pbar.update(1)

            pbar.close()

        fps_msg: str = f" @ {fps:.2f} fps" if fps else ""
        print(f"[done] Stream-processed {total_frames} frames{fps_msg} (prompt='{cfg.prompt}').")
    finally:
        if hasattr(frame_iter, "close"):
            frame_iter.close()


__all__ = [
    "Sam3VideoModelConfig",
    "Sam3StreamDemoConfig",
    "main",
]
