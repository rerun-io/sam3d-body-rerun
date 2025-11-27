"""
Demonstrates integrating Rerun visualization with Gradio.

Provides example implementations of data streaming, keypoint annotation, and dynamic
visualization across multiple Gradio tabs using Rerun's recording and visualization capabilities.
"""

import os
import tempfile
from pathlib import Path
from typing import Final

import cv2
import gradio as gr
import rerun as rr
import rerun.blueprint as rrb
import spaces
from gradio_rerun import Rerun
from jaxtyping import Int, UInt8
from monopriors.relative_depth_models import RelativeDepthPrediction
from numpy import ndarray

from sam3d_body.api.demo import SAM3Config, SAM3DBodyE2E, SAM3DBodyE2EConfig, create_view, set_annotation_context
from sam3d_body.api.visualization import visualize_sample
from sam3d_body.sam_3d_body_estimator import FinalPosePrediction

CFG: SAM3DBodyE2EConfig = SAM3DBodyE2EConfig(sam3_config=SAM3Config())
MODEL_E2E: SAM3DBodyE2E = SAM3DBodyE2E(config=CFG)
mesh_faces: Int[ndarray, "n_faces=36874 3"] = MODEL_E2E.sam3d_body_estimator.faces
STATE: Final[str] = "✅ Ready"
# Absolute path to bundled example data used by Gradio examples.
TEST_INPUT_DIR: Final[Path] = Path(__file__).resolve().parents[3] / "data" / "example-data"

# Allow Gradio to serve and cache files from the bundled test data directory.
gr.set_static_paths([str(TEST_INPUT_DIR)])


@spaces.GPU()
@rr.thread_local_stream("sam3d_body_gradio_ui")
def sam3d_prediction_fn(
    rgb_hw3: UInt8[ndarray, "h w 3"],
    log_relative_depth: bool,
    pending_cleanup: list[str] | None = None,
) -> tuple[str, str]:
    # resize rgb so that its largest dimension is 1024
    rgb_hw3 = cv2.resize(
        rgb_hw3,
        dsize=(0, 0),
        fx=1024 / max(rgb_hw3.shape[0], rgb_hw3.shape[1]),
        fy=1024 / max(rgb_hw3.shape[0], rgb_hw3.shape[1]),
        interpolation=cv2.INTER_AREA,
    )
    # We eventually want to clean up the RRD file after it's sent to the viewer, so tracking
    # any pending files to be cleaned up when the state is deleted.
    temp = tempfile.NamedTemporaryFile(prefix="cube_", suffix=".rrd", delete=False)

    if pending_cleanup is not None:
        pending_cleanup.append(temp.name)

    view: rrb.ContainerLike = create_view()
    blueprint = rrb.Blueprint(view, collapse_panels=True)
    rr.save(path=temp.name, default_blueprint=blueprint)
    set_annotation_context()
    parent_log_path = Path("/world")
    rr.log("/", rr.ViewCoordinates.RDF, static=True)

    outputs: tuple[list[FinalPosePrediction], RelativeDepthPrediction] = MODEL_E2E.predict_single_image(rgb_hw3=rgb_hw3)
    pred_list: list[FinalPosePrediction] = outputs[0]
    relative_pred: RelativeDepthPrediction = outputs[1]
    rr.set_time(timeline="image_sequence", sequence=0)
    visualize_sample(
        pred_list=pred_list,
        rgb_hw3=rgb_hw3,
        parent_log_path=parent_log_path,
        faces=mesh_faces,
        relative_depth_pred=relative_pred if log_relative_depth else None,
    )

    return temp.name, STATE


def cleanup_rrds(pending_cleanup: list[str]) -> None:
    for f in pending_cleanup:
        os.unlink(f)


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

    with gr.Blocks() as demo, gr.Tab("SAM3D Body Estimation"):
        pending_cleanup = gr.State([], time_to_live=10, delete_callback=cleanup_rrds)
        with gr.Row():
            with gr.Column(scale=1):
                img = gr.Image(interactive=True, label="Image", type="numpy", image_mode="RGB")
                depth_checkbox = gr.Checkbox(label="Log relative depth", value=False)
                create_rrd = gr.Button("Predict Pose")
                status = gr.Text(STATE, label="Status")
                gr.Examples(
                    examples=[
                        [str(TEST_INPUT_DIR / "Amir-Khan-Lamont-Peterson_2689582.jpg"), False],
                        [str(TEST_INPUT_DIR / "Planche.jpg"), True],
                        [str(TEST_INPUT_DIR / "BNAAHPYGMYSE26U6C6T7VA6544.jpg"), False],
                        [str(TEST_INPUT_DIR / "yoga-example.jpg"), True],
                    ],
                    inputs=[img, depth_checkbox],
                    outputs=[viewer, status],
                    fn=sam3d_prediction_fn,
                    run_on_click=True,
                    cache_examples=False,
                    examples_per_page=2,
                )

            with gr.Column(scale=5):
                viewer.render()
        create_rrd.click(
            sam3d_prediction_fn,
            inputs=[img, depth_checkbox, pending_cleanup],
            outputs=[viewer, status],
        )
    return demo
