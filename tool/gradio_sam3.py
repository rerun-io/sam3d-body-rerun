import warnings

import gradio as gr
import numpy as np
import requests

# import spaces
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

warnings.filterwarnings("ignore")

# Global model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Sam3Model.from_pretrained(
    "facebook/sam3", torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")


# @spaces.GPU()
def segment(image: Image.Image, text: str, threshold: float, mask_threshold: float):
    """
    Perform promptable concept segmentation using SAM3.
    Returns format compatible with gr.AnnotatedImage: (image, [(mask, label), ...])
    """
    if image is None:
        return None, "❌ Please upload an image."

    if not text.strip():
        return (image, []), "❌ Please enter a text prompt."

    try:
        inputs = processor(images=image, text=text.strip(), return_tensors="pt").to(device)

        for key in inputs:
            if inputs[key].dtype == torch.float32:
                inputs[key] = inputs[key].to(model.dtype)

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        n_masks = len(results["masks"])
        if n_masks == 0:
            return (image, []), f"❌ No objects found matching '{text}' (try adjusting thresholds)."

        # Format for AnnotatedImage: list of (mask, label) tuples
        # mask should be numpy array with values 0-1 (float) matching image dimensions
        annotations = []
        for i, (mask, score) in enumerate(zip(results["masks"], results["scores"])):
            # Convert binary mask to float numpy array (0-1 range)
            mask_np = mask.cpu().numpy().astype(np.float32)
            label = f"{text} #{i + 1} ({score:.2f})"
            annotations.append((mask_np, label))

        scores_text = ", ".join([f"{s:.2f}" for s in results["scores"].cpu().numpy()[:5]])
        info = f"✅ Found **{n_masks}** objects matching **'{text}'**\nConfidence scores: {scores_text}{'...' if n_masks > 5 else ''}"

        # Return tuple: (base_image, list_of_annotations)
        return (image, annotations), info

    except Exception as e:
        return (image, []), f"❌ Error during segmentation: {str(e)}"


def clear_all():
    """Clear all inputs and outputs"""
    return None, "", None, 0.5, 0.5, "📝 Enter a prompt and click **Segment** to start."


def segment_example(image_path: str, prompt: str):
    """Handle example clicks"""
    if image_path.startswith("http"):
        image = Image.open(requests.get(image_path, stream=True).raw).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")
    return segment(image, prompt, 0.5, 0.5)


# Gradio Interface
with gr.Blocks(
    theme=gr.themes.Soft(),
    title="SAM3 - Promptable Concept Segmentation",
    css=".gradio-container {max-width: 1400px !important;}",
) as demo:
    gr.Markdown(
        """
        # SAM3 - Promptable Concept Segmentation (PCS)
        
        **SAM3** performs zero-shot instance segmentation using natural language prompts.
        Upload an image, enter a text prompt (e.g., "person", "car", "dog"), and get segmentation masks.
        
        Built with [anycoder](https://huggingface.co/spaces/akhaliq/anycoder)
        """
    )

    gr.Markdown("### Inputs")
    with gr.Row(variant="panel"):
        image_input = gr.Image(
            label="Input Image",
            type="pil",
            height=400,
        )
        # AnnotatedImage expects: (base_image, [(mask, label), ...])
        image_output = gr.AnnotatedImage(
            label="Output (Segmented Image)",
            height=400,
            show_legend=True,
        )

    with gr.Row():
        text_input = gr.Textbox(label="Text Prompt", placeholder="e.g., person, ear, cat, bicycle...", scale=3)
        clear_btn = gr.Button("🔍 Clear", size="sm", variant="secondary")

    with gr.Row():
        thresh_slider = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.5,
            step=0.01,
            label="Detection Threshold",
            info="Higher = fewer detections",
        )
        mask_thresh_slider = gr.Slider(
            minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="Mask Threshold", info="Higher = sharper masks"
        )

    info_output = gr.Markdown(value="📝 Enter a prompt and click **Segment** to start.", label="Info / Results")

    segment_btn = gr.Button("🎯 Segment", variant="primary", size="lg")

    gr.Examples(
        examples=[
            ["http://images.cocodataset.org/val2017/000000077595.jpg", "cat"],
        ],
        inputs=[image_input, text_input],
        outputs=[image_output, info_output],
        fn=segment_example,
        cache_examples=False,
    )

    clear_btn.click(
        fn=clear_all, outputs=[image_input, text_input, image_output, thresh_slider, mask_thresh_slider, info_output]
    )

    segment_btn.click(
        fn=segment,
        inputs=[image_input, text_input, thresh_slider, mask_thresh_slider],
        outputs=[image_output, info_output],
    )

    gr.Markdown(
        """
        ### Notes
        - **Model**: [facebook/sam3](https://huggingface.co/facebook/sam3)
        - Click on segments in the output to see labels
        - GPU recommended for faster inference
        """
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, debug=True)
