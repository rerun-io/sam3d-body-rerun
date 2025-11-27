import tyro

from sam3d_body.api.demo import Sam3DBodyDemoConfig, main

if __name__ == "__main__":
    main(tyro.cli(Sam3DBodyDemoConfig))
