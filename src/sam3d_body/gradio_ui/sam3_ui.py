"""
Demonstrates integrating Rerun visualization with Gradio.

Provides example implementations of data streaming, keypoint annotation, and dynamic
visualization across multiple Gradio tabs using Rerun's recording and visualization capabilities.
"""

import uuid
from pathlib import Path
from typing import Final

import gradio as gr
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import spaces
from gradio_rerun import Rerun
from jaxtyping import Bool, Float32, UInt8
from numpy import ndarray

from sam3d_body.api.demo import SAM3Config, SAM3Predictor, SAM3Results
from sam3d_body.api.visualization import SEG_CLASS_OFFSET, SEG_OVERLAY_ALPHA

CFG: SAM3Config = SAM3Config()
MODEL_E2E: SAM3Predictor = SAM3Predictor(config=CFG)
DONE_STATUS: Final[str] = "✅ Ready"
RUNNING_STATUS: Final[str] = "⏳ Running prediction..."
# Absolute path to bundled example data used by Gradio examples.
TEST_INPUT_DIR: Final[Path] = Path(__file__).resolve().parents[3] / "data" / "example-data"
# Palette reused for segmentation overlays; RGBA so we can set alpha when building an overlay.
BOX_PALETTE: UInt8[np.ndarray, "n_colors 4"] = np.array(
    [
        [255, 99, 71, 255],  # tomato
        [65, 105, 225, 255],  # royal blue
        [60, 179, 113, 255],  # medium sea green
        [255, 215, 0, 255],  # gold
        [138, 43, 226, 255],  # blue violet
        [255, 140, 0, 255],  # dark orange
        [220, 20, 60, 255],  # crimson
        [70, 130, 180, 255],  # steel blue
    ],
    dtype=np.uint8,
)

# Allow Gradio to serve and cache files from the bundled test data directory.
gr.set_static_paths([str(TEST_INPUT_DIR)])


def _ensure_uuid(recording_id: uuid.UUID | str | None) -> uuid.UUID:
    """Normalize provided recording id to a uuid.UUID instance."""
    if recording_id is None:
        return uuid.uuid4()
    if isinstance(recording_id, uuid.UUID):
        return recording_id
    return uuid.UUID(str(recording_id))


def get_recording(recording_id: uuid.UUID | str | None) -> rr.RecordingStream:
    normalized_id: uuid.UUID = _ensure_uuid(recording_id)
    return rr.RecordingStream(application_id="rerun_example_gradio", recording_id=normalized_id)


@spaces.GPU()
def sam3d_prediction_fn(
    img: UInt8[ndarray, "h w 3"] | None, text_prompt: str | None, recording_id: uuid.UUID | str | None
):
    if text_prompt is None:
        raise gr.Error("Must provide a text prompt.")
    # Here we get a recording using the provided recording id.
    rec = get_recording(recording_id)
    stream = rec.binary_stream()  # type: ignore

    if img is None:
        raise gr.Error("Must provide an image to blur.")

    blueprint = rrb.Blueprint(
        rrb.Spatial2DView(
            name="Image + Segmentation",
            contents=[
                "image",
                "image/segmentation_ids",
            ],
        ),
        collapse_panels=True,
    )
    sam3_results: SAM3Results = MODEL_E2E.predict_single_image(rgb_hw3=img, text=text_prompt)

    rec.send_blueprint(blueprint)
    rec.set_time("iteration", sequence=0)
    rec.log("image", rr.Image(img))
    yield stream.read(), RUNNING_STATUS

    h: int = int(img.shape[0])
    w: int = int(img.shape[1])
    seg_map: UInt8[np.ndarray, "h w"] = np.full((h, w), SEG_CLASS_OFFSET, dtype=np.uint8)

    # Build a single segmentation image where each instance gets a unique id.
    for idx, segmask in enumerate(sam3_results.masks):
        mask: Float32[np.ndarray, "h w"] = np.asarray(segmask, dtype=np.float32).squeeze()
        mask_bool: Bool[np.ndarray, "h w"] = mask >= 0.5
        class_id: int = SEG_CLASS_OFFSET + idx + 1  # reserve SEG_CLASS_OFFSET for background
        seg_map = np.where(mask_bool, np.uint8(class_id), seg_map)

    class_descriptions: list[rr.ClassDescription] = [
        rr.ClassDescription(info=rr.AnnotationInfo(id=SEG_CLASS_OFFSET, label="Background", color=(64, 64, 64, 0)))
    ]
    for idx, color_rgb in enumerate(BOX_PALETTE[:, :3].tolist(), start=1):
        color_rgba: tuple[int, int, int, int] = (
            int(color_rgb[0]),  # type: ignore[arg-type]  # numpy .tolist() lacks type stubs
            int(color_rgb[1]),  # type: ignore[arg-type]
            int(color_rgb[2]),  # type: ignore[arg-type]
            SEG_OVERLAY_ALPHA,
        )
        class_descriptions.append(
            rr.ClassDescription(
                info=rr.AnnotationInfo(id=SEG_CLASS_OFFSET + idx, label=f"Mask-{idx}", color=color_rgba)
            )
        )

    rec.log("/", rr.AnnotationContext(class_descriptions), static=True)
    rec.log("image/segmentation_ids", rr.SegmentationImage(seg_map))
    yield stream.read(), RUNNING_STATUS


def _switch_to_outputs() -> gr.Tabs:
    return gr.update(selected="outputs")


def main():
    viewer = Rerun(
        streaming=True,
        panel_states={
            "time": "collapsed",
            "blueprint": "hidden",
            "selection": "hidden",
        },
        height=800,
    )

    with gr.Blocks() as demo:
        recording_id = gr.State(str(uuid.uuid4()))

        with gr.Row():
            with gr.Column(scale=1):
                tabs = gr.Tabs(selected="inputs")
                with tabs:
                    with gr.TabItem("Inputs", id="inputs"):
                        img = gr.Image(interactive=True, label="Image", type="numpy", image_mode="RGB")
                        text_prompt = gr.Textbox(label="Text Prompt")
                        create_rrd = gr.Button("Predict Pose")
                    with gr.TabItem("Outputs", id="outputs"):
                        status = gr.Text(DONE_STATUS, label="Status")

                gr.Examples(
                    examples=[
                        [str(TEST_INPUT_DIR / "Planche.jpg")],
                        [str(TEST_INPUT_DIR / "Amir-Khan-Lamont-Peterson_2689582.jpg")],
                        [str(TEST_INPUT_DIR / "BNAAHPYGMYSE26U6C6T7VA6544.jpg")],
                        [str(TEST_INPUT_DIR / "yoga-example.jpg")],
                    ],
                    inputs=[img, text_prompt],
                    outputs=[viewer, status],
                    fn=sam3d_prediction_fn,
                    run_on_click=False,
                    cache_examples=False,
                    examples_per_page=2,
                )
            with gr.Column(scale=5):
                viewer.render()

        create_rrd.click(
            fn=_switch_to_outputs,
            inputs=None,
            outputs=[tabs],
        ).then(
            sam3d_prediction_fn,
            inputs=[img, text_prompt, recording_id],
            outputs=[viewer, status],
        ).then(
            lambda: gr.update(value=DONE_STATUS),
            inputs=None,
            outputs=[status],
        )
    return demo
