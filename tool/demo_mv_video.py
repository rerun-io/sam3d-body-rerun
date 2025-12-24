"""Thin CLI wrapper to launch the SAM3 multiview video demo inside the package."""

import tyro

from sam3d_body.api.demo_mv_video import Sam3MVVideoDemoConfig, main

if __name__ == "__main__":
    main(tyro.cli(Sam3MVVideoDemoConfig))
