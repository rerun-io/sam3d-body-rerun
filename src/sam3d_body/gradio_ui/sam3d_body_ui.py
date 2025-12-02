"""
Demonstrates integrating Rerun visualization with Gradio.

Provides example implementations of data streaming, keypoint annotation, and dynamic
visualization across multiple Gradio tabs using Rerun's recording and visualization capabilities.
"""

import os
import shutil
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
from sam3d_body.api.visualization import export_meshes_to_glb, visualize_sample
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
    rgb_hw3,
    log_relative_depth,
    export_glb,
    center_glb,
    pending_cleanup=None,
) -> tuple[str, str, list[str]]:
    # resize rgb so that its largest dimension is 1024
    rgb_hw3: UInt8[ndarray, "h w 3"] = cv2.resize(
        rgb_hw3,  # type: ignore[arg-type]
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

    view: rrb.ContainerLike = create_view(log_relative_depth)
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

    glb_files: list[str] = []
    if export_glb and len(pred_list) > 0:
        glb_dir: Path = Path(tempfile.mkdtemp(prefix="sam3d_glb_"))
        glb_paths = export_meshes_to_glb(
            pred_list=pred_list,
            faces=mesh_faces,
            output_dir=glb_dir,
            center_mesh=center_glb,
        )
        glb_files = [str(p) for p in glb_paths]
        if pending_cleanup is not None:
            pending_cleanup.extend(glb_files)
            pending_cleanup.append(str(glb_dir))

    return temp.name, STATE, glb_files


def cleanup_rrds(pending_cleanup: list[str]) -> None:
    for f in pending_cleanup:
        if os.path.isdir(f):
            shutil.rmtree(f, ignore_errors=True)
        elif os.path.isfile(f):
            os.unlink(f)


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

    with gr.Blocks() as demo, gr.Tab("SAM3D Body Estimation"):
        pending_cleanup = gr.State([], time_to_live=10, delete_callback=cleanup_rrds)
        gr.Markdown(
            """
# SAM3D Body with Rerun
An unofficial playground for Meta's SAM3D Body (DINOv3) with promptable SAM3 masks and live Rerun visualization. Uses **Rerun** for 3D inspection, **Gradio** for the UI, and **Pixi** for one-command setup.

When "Log relative depth" is enabled, the 3D panel switches to tabbed views for full, background-only, and people-only depth point clouds.

<div style="display:flex; gap:8px; justify-content:center; align-items:center; flex-wrap:wrap; margin:12px 0;">
  <a title="Rerun" href="https://rerun.io" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/Rerun-0.27%2B-0b82f9" alt="Rerun badge">
  </a>
  <a title="Pixi" href="https://pixi.sh/latest/" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/Install%20with-Pixi-16A34A" alt="Pixi badge">
  </a>
  <a title="CUDA" href="https://developer.nvidia.com/cuda-toolkit" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/CUDA-12.9%2B-76b900" alt="CUDA badge">
  </a>
  <a title="GitHub" href="https://github.com/rerun-io/sam3d-body-rerun" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/github/stars/rerun-io/sam3d-body-rerun?label=GitHub%20%E2%98%85&logo=github&color=C8C" alt="GitHub stars">
  </a>
</div>
            """
        )
        gr.HTML(
            """
<style>
.sam3d-viewer-col { margin-top: 20px; }
</style>
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                tabs = gr.Tabs(selected="inputs")
                with tabs:
                    with gr.TabItem("Inputs", id="inputs"):
                        img = gr.Image(interactive=True, label="Image", type="numpy", image_mode="RGB")
                        depth_checkbox = gr.Checkbox(label="Log relative depth", value=False)
                        with gr.Row():
                            export_checkbox = gr.Checkbox(label="Export GLB meshes", value=False)
                            center_checkbox = gr.Checkbox(label="Center GLB at origin", value=True)
                        create_rrd = gr.Button("Predict Pose")
                    with gr.TabItem("Outputs", id="outputs"):
                        status = gr.Text(STATE, label="Status")
                        mesh_files = gr.Files(label="GLB meshes", file_count="multiple")
                gr.Examples(
                    examples=[
                        [str(TEST_INPUT_DIR / "Planche.jpg"), True, False, True],
                        [str(TEST_INPUT_DIR / "Amir-Khan-Lamont-Peterson_2689582.jpg"), False, False, True],
                        [str(TEST_INPUT_DIR / "BNAAHPYGMYSE26U6C6T7VA6544.jpg"), False, True, True],
                        [str(TEST_INPUT_DIR / "yoga-example.jpg"), True, True, False],
                    ],
                    inputs=[img, depth_checkbox, export_checkbox, center_checkbox],
                    outputs=[viewer, status, mesh_files],
                    fn=sam3d_prediction_fn,
                    run_on_click=True,
                    cache_examples=False,
                    examples_per_page=2,
                )
            with gr.Column(scale=5, elem_classes=["sam3d-viewer-col"]):
                viewer.render()

        create_rrd.click(
            fn=_switch_to_outputs,
            inputs=None,
            outputs=[tabs],
        ).then(
            sam3d_prediction_fn,
            inputs=[img, depth_checkbox, export_checkbox, center_checkbox, pending_cleanup],
            outputs=[viewer, status, mesh_files],
        )
    return demo
