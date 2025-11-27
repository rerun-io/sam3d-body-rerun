from sam3d_body.gradio_ui.sam3d_body_ui import TEST_INPUT_DIR, main

if __name__ == "__main__":
    demo = main()
    demo.launch(ssr_mode=False, allowed_paths=[str(TEST_INPUT_DIR)])
