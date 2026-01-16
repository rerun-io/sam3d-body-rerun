"""Thin CLI wrapper to launch the SAM3 video demo inside the package."""

import tyro

from sam3d_body.api.demo_video import Sam3VideoDemoConfig, main

if __name__ == "__main__":
    main(tyro.cli(Sam3VideoDemoConfig))
