"""Thin CLI wrapper to launch the streaming SAM3 video demo inside the package."""

import tyro

from sam3d_body.api.demo_stream import Sam3StreamDemoConfig, main

if __name__ == "__main__":
    main(tyro.cli(Sam3StreamDemoConfig))
