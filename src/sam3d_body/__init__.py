import os

# Only enable beartype when running in the 'dev' environment
# Check the PIXI_ENVIRONMENT_NAME environment variable set by pixi
if os.environ.get("PIXI_ENVIRONMENT_NAME") == "dev":
    try:
        from beartype.claw import beartype_this_package

        beartype_this_package()
    except ImportError:
        # beartype not available even in dev environment
        pass
