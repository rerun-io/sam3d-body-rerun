"""Thin CLI wrapper to launch the SAM3 multiview body demo inside the package."""

import tyro

from sam3d_body.api.demo_mv_body import Sam3MVBodyDemoConfig, main

if __name__ == "__main__":
    main(tyro.cli(Sam3MVBodyDemoConfig))
