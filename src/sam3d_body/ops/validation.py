"""Validation utilities for multiview optimization quality assessment.

Computes 2D reprojection errors to validate triangulation and optimization
against detected keypoints (pseudo ground truth).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Float, Float32
from numpy import ndarray


@dataclass(slots=True)
class ValidationResult:
    """Results from reprojection error validation."""

    mean_error_px: float
    """Mean 2D error in pixels across all keypoints and views."""
    p90_error_px: float
    """90th percentile error in pixels."""
    max_error_px: float
    """Maximum per-keypoint error in pixels."""
    per_view_mean_errors: list[float]
    """Mean error per camera view."""
    passed: bool
    """True if all metrics within thresholds."""

    def __str__(self) -> str:
        """Format as summary string."""
        status: str = "✓" if self.passed else "✗"
        return f"{status} mean={self.mean_error_px:.1f}px, p90={self.p90_error_px:.1f}px, max={self.max_error_px:.1f}px"


# Default thresholds
DEFAULT_MEAN_THRESHOLD_PX: float = 10.0
DEFAULT_P90_THRESHOLD_PX: float = 15.0
DEFAULT_MAX_THRESHOLD_PX: float = 50.0


def compute_reprojection_errors(
    projected_2d: Float[ndarray, "n_views n_kpts 2"],
    target_2d: Float[ndarray, "n_views n_kpts 2"],
    confidence: Float[ndarray, "n_views n_kpts"] | None = None,
) -> Float32[ndarray, "n_views n_kpts"]:
    """Compute per-keypoint 2D reprojection errors.

    Args:
        projected_2d: Projected keypoints per view.
        target_2d: Target (detected) keypoints per view.
        confidence: Optional confidence mask. Errors for zero-confidence
            keypoints are set to NaN.

    Returns:
        Euclidean distance in pixels for each keypoint/view.
    """
    diff: Float[ndarray, "n_views n_kpts 2"] = projected_2d - target_2d
    errors: Float32[ndarray, "n_views n_kpts"] = np.linalg.norm(diff, axis=-1).astype(np.float32)

    if confidence is not None:
        # Mask out low-confidence keypoints
        mask: Bool[ndarray, "n_views n_kpts"] = confidence > 0
        errors = np.where(mask, errors, np.nan)

    return errors


def validate_reprojection(
    projected_2d: Float[ndarray, "n_views n_kpts 2"],
    target_2d: Float[ndarray, "n_views n_kpts 2"],
    confidence: Float[ndarray, "n_views n_kpts"] | None = None,
    mean_threshold: float = DEFAULT_MEAN_THRESHOLD_PX,
    p90_threshold: float = DEFAULT_P90_THRESHOLD_PX,
    max_threshold: float = DEFAULT_MAX_THRESHOLD_PX,
) -> ValidationResult:
    """Validate reprojection quality against thresholds.

    Args:
        projected_2d: Projected keypoints per view ``[n_views, n_kpts, 2]``.
        target_2d: Target keypoints per view ``[n_views, n_kpts, 2]``.
        confidence: Optional per-keypoint confidence ``[n_views, n_kpts]``.
        mean_threshold: Maximum allowed mean error (pixels).
        p90_threshold: Maximum allowed 90th percentile error (pixels).
        max_threshold: Maximum allowed per-keypoint error (pixels).

    Returns:
        ValidationResult with computed metrics and pass/fail status.
    """
    errors: Float32[ndarray, "n_views n_kpts"] = compute_reprojection_errors(
        projected_2d, target_2d, confidence
    )

    # Flatten and remove NaN for aggregate stats
    valid_errors: Float32[ndarray, "n_valid"] = errors[~np.isnan(errors)]

    if len(valid_errors) == 0:
        return ValidationResult(
            mean_error_px=float("nan"),
            p90_error_px=float("nan"),
            max_error_px=float("nan"),
            per_view_mean_errors=[],
            passed=False,
        )

    mean_err: float = float(np.mean(valid_errors))
    p90_err: float = float(np.percentile(valid_errors, 90))
    max_err: float = float(np.max(valid_errors))

    # Per-view mean errors
    n_views: int = errors.shape[0]
    per_view_means: list[float] = []
    for view_idx in range(n_views):
        view_errors: Float32[ndarray, "n_kpts"] = errors[view_idx]
        view_valid: Float32[ndarray, "n_valid"] = view_errors[~np.isnan(view_errors)]
        if len(view_valid) > 0:
            per_view_means.append(float(np.mean(view_valid)))
        else:
            per_view_means.append(float("nan"))

    passed: bool = (
        mean_err <= mean_threshold
        and p90_err <= p90_threshold
        and max_err <= max_threshold
    )

    return ValidationResult(
        mean_error_px=mean_err,
        p90_error_px=p90_err,
        max_error_px=max_err,
        per_view_mean_errors=per_view_means,
        passed=passed,
    )


__all__ = [
    "ValidationResult",
    "compute_reprojection_errors",
    "validate_reprojection",
    "DEFAULT_MEAN_THRESHOLD_PX",
    "DEFAULT_P90_THRESHOLD_PX",
    "DEFAULT_MAX_THRESHOLD_PX",
]
