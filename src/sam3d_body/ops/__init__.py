"""Operations module for multiview optimization.

Re-exports triangulation from simplecv and provides loss functions and optimizer.
"""

# Re-export triangulation from simplecv (already a dependency)
from simplecv.ops.triangulate import (
    batch_triangulate,
    proj_3d_vectorized,
)

from sam3d_body.ops.losses import (
    LossWeights,
    MultiviewLossOutput,
    MultiviewOptimizationLoss,
    MultiviewReprojectionLoss,
    TemporalSmoothnessLoss,
    Triangulated3DLoss,
)
from sam3d_body.ops.optimizer import (
    MultiviewBodyOptimizer,
    MultiviewOptimizerConfig,
    OptimizationResult,
)
from sam3d_body.ops.validation import (
    ValidationResult,
    compute_reprojection_errors,
    validate_reprojection,
)

__all__ = [
    # Triangulation (from simplecv)
    "batch_triangulate",
    "proj_3d_vectorized",
    # Loss functions
    "LossWeights",
    "MultiviewLossOutput",
    "MultiviewOptimizationLoss",
    "MultiviewReprojectionLoss",
    "TemporalSmoothnessLoss",
    "Triangulated3DLoss",
    # Optimizer
    "MultiviewBodyOptimizer",
    "MultiviewOptimizerConfig",
    "OptimizationResult",
    # Validation
    "ValidationResult",
    "compute_reprojection_errors",
    "validate_reprojection",
]

