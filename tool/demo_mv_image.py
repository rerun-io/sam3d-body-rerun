"""Thin CLI wrapper to launch the SAM3 multiview image demo inside the package."""

import tyro

from sam3d_body.api.demo_mv_image import Sam3MVImageDemoConfig, main

if __name__ == "__main__":
    main(tyro.cli(Sam3MVImageDemoConfig))
