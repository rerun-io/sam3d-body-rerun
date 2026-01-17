"""Multiview optimization loss functions for MHR body mesh fitting.

Implements the L_multi loss function:
    L_multi = λ_2d * L_2d + λ_3d * L_3d + λ_temp * L_temp

Where:
- L_2d: Reprojection error across all camera views
- L_3d: Distance to triangulated 3D keypoints
- L_temp: Temporal smoothness penalty
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn
from jaxtyping import Float32
from torch import Tensor


class LossWeights(NamedTuple):
    """Weights for multiview optimization loss terms."""

    reprojection_2d: float = 1.0
    """Weight for 2D reprojection loss (L1 in pixels)."""
    keypoint_3d: float = 100.0
    """Weight for 3D keypoint supervision (L1 in meters, ~100:1 m→px ratio)."""
    temporal: float = 0.1
    """Weight for temporal smoothness penalty."""


@dataclass(slots=True)
class MultiviewLossOutput:
    """Structured output from multiview loss computation."""

    total: Tensor
    """Combined weighted loss value."""
    reprojection_2d: Tensor
    """2D reprojection component."""
    keypoint_3d: Tensor
    """3D keypoint component."""
    temporal: Tensor
    """Temporal smoothness component."""


class MultiviewReprojectionLoss(nn.Module):
    """2D reprojection loss across multiple camera views.

    Computes the mean squared error between projected mesh keypoints
    and detected 2D keypoints across all camera views.
    """

    def __init__(self, reduction: str = "mean") -> None:
        """Initialize the reprojection loss.

        Args:
            reduction: Reduction mode ('mean' or 'sum').
        """
        super().__init__()
        self.reduction: str = reduction

    def forward(
        self,
        projected_2d: Float32[Tensor, "n_views n_kpts 2"],
        target_2d: Float32[Tensor, "n_views n_kpts 2"],
        confidence: Float32[Tensor, "n_views n_kpts"] | None = None,
    ) -> Tensor:
        """Compute reprojection loss.

        Args:
            projected_2d: Projected keypoints from mesh ``[n_views, n_kpts, 2]``.
            target_2d: Detected 2D keypoints ``[n_views, n_kpts, 2]``.
            confidence: Optional per-keypoint confidence ``[n_views, n_kpts]``.
                Used to down-weight noisy detections.

        Returns:
            Scalar loss value.
        """
        # L1 loss (mean absolute distance in pixels) for consistent scale
        diff: Float32[Tensor, "n_views n_kpts 2"] = projected_2d - target_2d
        l1_error: Float32[Tensor, "n_views n_kpts"] = diff.abs().sum(dim=-1)  # L1 per kpt

        if confidence is not None:
            # Weight by confidence, normalize by sum of confidences
            weighted_error: Float32[Tensor, "n_views n_kpts"] = l1_error * confidence
            conf_sum: Tensor = confidence.sum().clamp(min=1e-8)
            return weighted_error.sum() / conf_sum

        if self.reduction == "mean":
            return l1_error.mean()
        return l1_error.sum()


class Triangulated3DLoss(nn.Module):
    """3D keypoint loss using triangulated supervision.

    Computes the mean squared error between mesh keypoints and
    triangulated 3D keypoints from multi-view geometry.
    """

    def __init__(self, reduction: str = "mean") -> None:
        """Initialize the 3D keypoint loss.

        Args:
            reduction: Reduction mode ('mean' or 'sum').
        """
        super().__init__()
        self.reduction: str = reduction

    def forward(
        self,
        mesh_kpts_3d: Float32[Tensor, "n_kpts 3"],
        triangulated_3d: Float32[Tensor, "n_kpts 3"],
        confidence: Float32[Tensor, "n_kpts"] | None = None,
    ) -> Tensor:
        """Compute 3D keypoint loss.

        Args:
            mesh_kpts_3d: 3D keypoints from mesh model ``[n_kpts, 3]``.
            triangulated_3d: Triangulated keypoints ``[n_kpts, 3]``.
            confidence: Optional per-keypoint confidence ``[n_kpts]``.

        Returns:
            Scalar loss value.
        """
        # L1 loss (mean absolute distance in meters) for better scaling
        diff: Float32[Tensor, "n_kpts 3"] = mesh_kpts_3d - triangulated_3d
        l1_error: Float32[Tensor, "n_kpts"] = diff.abs().sum(dim=-1)  # Per-keypoint L1

        if confidence is not None:
            # Only consider keypoints with non-zero confidence
            valid_mask: Float32[Tensor, "n_kpts"] = (confidence > 0).float()
            weighted_error: Float32[Tensor, "n_kpts"] = l1_error * valid_mask * confidence
            mask_sum: Tensor = (valid_mask * confidence).sum().clamp(min=1e-8)
            return weighted_error.sum() / mask_sum

        if self.reduction == "mean":
            return l1_error.mean()
        return l1_error.sum()


class TemporalSmoothnessLoss(nn.Module):
    """Temporal smoothness loss for video sequences.

    Penalizes abrupt changes in pose parameters and translation
    between consecutive frames.
    """

    def __init__(self, reduction: str = "mean") -> None:
        """Initialize temporal smoothness loss.

        Args:
            reduction: Reduction mode ('mean' or 'sum').
        """
        super().__init__()
        self.reduction: str = reduction

    def forward(
        self,
        pose_curr: Float32[Tensor, "n_params"],
        pose_prev: Float32[Tensor, "n_params"],
        trans_curr: Float32[Tensor, "3"] | None = None,
        trans_prev: Float32[Tensor, "3"] | None = None,
    ) -> Tensor:
        """Compute temporal smoothness loss.

        Args:
            pose_curr: Current frame pose parameters.
            pose_prev: Previous frame pose parameters.
            trans_curr: Optional current translation.
            trans_prev: Optional previous translation.

        Returns:
            Scalar loss value.
        """
        pose_diff: Float32[Tensor, "n_params"] = pose_curr - pose_prev
        pose_loss: Tensor = (pose_diff**2).mean() if self.reduction == "mean" else (pose_diff**2).sum()

        if trans_curr is not None and trans_prev is not None:
            trans_diff: Float32[Tensor, "3"] = trans_curr - trans_prev
            trans_loss: Tensor = (trans_diff**2).mean() if self.reduction == "mean" else (trans_diff**2).sum()
            return pose_loss + trans_loss

        return pose_loss


class MultiviewOptimizationLoss(nn.Module):
    """Combined multiview optimization loss (L_multi).

    Combines 2D reprojection, 3D keypoint, and temporal smoothness
    losses with configurable weights.
    """

    def __init__(self, weights: LossWeights | None = None) -> None:
        """Initialize the combined loss.

        Args:
            weights: Loss term weights. Uses defaults if None.
        """
        super().__init__()
        self.weights: LossWeights = weights or LossWeights()
        self.reprojection_loss = MultiviewReprojectionLoss()
        self.keypoint_3d_loss = Triangulated3DLoss()
        self.temporal_loss = TemporalSmoothnessLoss()

    def forward(
        self,
        *,
        projected_2d: Float32[Tensor, "n_views n_kpts 2"],
        target_2d: Float32[Tensor, "n_views n_kpts 2"],
        mesh_kpts_3d: Float32[Tensor, "n_kpts 3"],
        triangulated_3d: Float32[Tensor, "n_kpts 3"],
        confidence_2d: Float32[Tensor, "n_views n_kpts"] | None = None,
        confidence_3d: Float32[Tensor, "n_kpts"] | None = None,
        pose_curr: Float32[Tensor, "n_params"] | None = None,
        pose_prev: Float32[Tensor, "n_params"] | None = None,
        trans_curr: Float32[Tensor, "3"] | None = None,
        trans_prev: Float32[Tensor, "3"] | None = None,
    ) -> MultiviewLossOutput:
        """Compute combined multiview loss.

        Args:
            projected_2d: Projected mesh keypoints per view.
            target_2d: Detected 2D keypoints per view.
            mesh_kpts_3d: 3D mesh keypoints.
            triangulated_3d: Triangulated 3D supervision.
            confidence_2d: Optional 2D detection confidence.
            confidence_3d: Optional triangulation confidence.
            pose_curr: Current pose for temporal loss.
            pose_prev: Previous pose for temporal loss.
            trans_curr: Current translation for temporal loss.
            trans_prev: Previous translation for temporal loss.

        Returns:
            MultiviewLossOutput with total and component losses.
        """
        # 2D reprojection loss
        loss_2d: Tensor = self.reprojection_loss(projected_2d, target_2d, confidence_2d)

        # 3D keypoint loss
        loss_3d: Tensor = self.keypoint_3d_loss(mesh_kpts_3d, triangulated_3d, confidence_3d)

        # Temporal smoothness loss
        loss_temp: Tensor
        if pose_curr is not None and pose_prev is not None:
            loss_temp = self.temporal_loss(pose_curr, pose_prev, trans_curr, trans_prev)
        else:
            loss_temp = torch.tensor(0.0, device=projected_2d.device)

        # Combine with weights
        total: Tensor = (
            self.weights.reprojection_2d * loss_2d
            + self.weights.keypoint_3d * loss_3d
            + self.weights.temporal * loss_temp
        )

        return MultiviewLossOutput(
            total=total,
            reprojection_2d=loss_2d,
            keypoint_3d=loss_3d,
            temporal=loss_temp,
        )


__all__ = [
    "LossWeights",
    "MultiviewLossOutput",
    "MultiviewReprojectionLoss",
    "Triangulated3DLoss",
    "TemporalSmoothnessLoss",
    "MultiviewOptimizationLoss",
]
