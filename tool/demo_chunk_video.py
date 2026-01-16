"""Thin CLI wrapper to launch the SAM3 chunk-based video demo inside the package."""

import tyro

from sam3d_body.api.demo_chunk_video import Sam3ChunkVideoDemoConfig, main

if __name__ == "__main__":
    main(tyro.cli(Sam3ChunkVideoDemoConfig))
