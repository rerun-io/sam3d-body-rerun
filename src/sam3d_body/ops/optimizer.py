"""PyTorch optimizer for multiview MHR body mesh fitting.

Implements staged Adam optimization with:
- 2D reprojection loss across all camera views
- 3D keypoint loss from triangulated supervision
- Temporal smoothness loss for video sequences
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from jaxtyping import Float, Float32
from numpy import ndarray
from simplecv.ops.triangulate import batch_triangulate
from torch import Tensor

from sam3d_body.ops.losses import (
    LossWeights,
    MultiviewLossOutput,
    MultiviewOptimizationLoss,
)
from sam3d_body.sam_3d_body_estimator import FinalPosePrediction


def axis_angle_to_matrix(axis_angle: Float32[Tensor, "3"]) -> Float32[Tensor, "3 3"]:
    """Convert axis-angle representation to rotation matrix (Rodrigues formula).

    Differentiable for optimization.

    Args:
        axis_angle: Axis-angle vector [3], where direction = axis, magnitude = angle.

    Returns:
        3x3 rotation matrix.
    """
    angle: Tensor = torch.norm(axis_angle)
    if angle < 1e-8:
        # No rotation - return identity
        return torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)

    axis: Float32[Tensor, "3"] = axis_angle / angle

    # Rodrigues formula: R = I + sin(θ)K + (1-cos(θ))K²
    # where K is the skew-symmetric matrix of axis
    K: Float32[Tensor, "3 3"] = torch.zeros(3, 3, device=axis_angle.device, dtype=axis_angle.dtype)
    K[0, 1] = -axis[2]
    K[0, 2] = axis[1]
    K[1, 0] = axis[2]
    K[1, 2] = -axis[0]
    K[2, 0] = -axis[1]
    K[2, 1] = axis[0]

    R: Float32[Tensor, "3 3"] = (
        torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
        + torch.sin(angle) * K
        + (1 - torch.cos(angle)) * (K @ K)
    )
    return R


@dataclass(slots=True)
class MultiviewOptimizerConfig:
    """Configuration for multiview mesh optimization."""

    num_iterations: int = 100
    """Number of Adam iterations per frame."""
    learning_rate: float = 0.01
    """Learning rate for Adam optimizer."""
    loss_weights: LossWeights = field(default_factory=LossWeights)
    """Weights for loss components (2D, 3D, temporal)."""
    optimize_translation: bool = True
    """Whether to optimize translation parameters."""
    optimize_global_rot: bool = True
    """Whether to optimize global rotation."""
    optimize_body_pose: bool = True
    """Whether to optimize body pose parameters."""
    min_views_for_triangulation: int = 2
    """Minimum camera views required for triangulation."""


@dataclass(slots=True)
class OptimizationResult:
    """Result from multiview optimization for a single frame."""

    global_rot: Float32[ndarray, "3"]
    """Optimized global rotation (radians)."""
    body_pose: Float32[ndarray, "133"]
    """Optimized body pose parameters."""
    translation: Float32[ndarray, "3"]
    """Optimized translation."""
    triangulated_kpts_3d: Float32[ndarray, "n_kpts 4"]
    """Triangulated 3D keypoints with confidence."""
    loss_history: list[float]
    """Loss values per iteration."""
    final_loss: MultiviewLossOutput
    """Detailed final loss breakdown."""


class MultiviewBodyOptimizer:
    """Multiview optimizer for MHR body mesh fitting.

    Takes per-camera body predictions and 2D keypoints, triangulates
    3D keypoints, and optimizes a single fused mesh using gradient descent.
    Uses MHR forward kinematics to compute keypoints from pose parameters.
    """

    def __init__(
        self,
        projection_matrices: Float[ndarray, "n_views 3 4"],
        first_world_T_cam: Float[ndarray, "4 4"],
        mhr_head: "torch.nn.Module",  # MHRHead from models.heads
        config: MultiviewOptimizerConfig | None = None,
        device: str = "cuda",
    ) -> None:
        """Initialize the multiview optimizer.

        Args:
            projection_matrices: Camera projection matrices P = K @ [R|t]
                with shape ``[n_views, 3, 4]``.
            first_world_T_cam: Transform from first camera to world coordinates.
                Used to convert camera-space keypoints to world before projection.
            mhr_head: MHRHead module for differentiable forward kinematics.
                Used to compute vertices/keypoints from pose parameters.
            config: Optimization configuration. Uses defaults if None.
            device: PyTorch device for optimization.
        """
        self.config: MultiviewOptimizerConfig = config or MultiviewOptimizerConfig()
        self.device: str = device
        self.n_views: int = projection_matrices.shape[0]

        # Store projection matrices as torch tensor
        self.P: Float32[Tensor, "n_views 3 4"] = torch.from_numpy(
            projection_matrices.astype(np.float32)
        ).to(device)

        # Store world_T_cam for first camera (to transform cam->world before projection)
        self.first_world_T_cam: Float32[Tensor, "4 4"] = torch.from_numpy(
            first_world_T_cam.astype(np.float32)
        ).to(device)

        # Store MHR head for forward kinematics
        self.mhr_head = mhr_head

        # Loss function
        self.loss_fn = MultiviewOptimizationLoss(weights=self.config.loss_weights)

        # Previous frame parameters for temporal smoothing
        self._prev_global_rot: Float32[Tensor, "3"] | None = None
        self._prev_body_pose: Float32[Tensor, "133"] | None = None
        self._prev_translation: Float32[Tensor, "3"] | None = None

    def reset_temporal_state(self) -> None:
        """Reset previous frame state (call at start of new sequence)."""
        self._prev_global_rot = None
        self._prev_body_pose = None
        self._prev_translation = None

    def _triangulate_keypoints(
        self,
        keypoints_2d: Float[ndarray, "n_views n_kpts 3"],
    ) -> Float32[ndarray, "n_kpts 4"]:
        """Triangulate 3D keypoints from multi-view 2D observations.

        Args:
            keypoints_2d: 2D keypoints per view with confidence in last channel.
                Shape ``[n_views, n_kpts, 3]`` where last dim is [u, v, conf].

        Returns:
            Triangulated 3D keypoints with confidence ``[n_kpts, 4]``.
        """
        P_np: Float[ndarray, "n_views 3 4"] = self.P.cpu().numpy()
        result = batch_triangulate(
            keypoints_2d,
            P_np,
            min_views=self.config.min_views_for_triangulation,
        )
        return result.astype(np.float32)

    def _project_keypoints(
        self,
        keypoints_cam: Float32[Tensor, "n_kpts 3"],
    ) -> Float32[Tensor, "n_views n_kpts 2"]:
        """Project 3D keypoints to all camera views (differentiable).

        Uses pure PyTorch operations to maintain gradient flow for optimization.
        Keypoints are assumed to be in the first camera's coordinate system and
        are transformed to world coordinates before projection.

        Args:
            keypoints_cam: 3D keypoints in first camera's coordinate space ``[n_kpts, 3]``.

        Returns:
            2D projections per view ``[n_views, n_kpts, 2]``.
        """
        n_kpts: int = keypoints_cam.shape[0]

        # Add homogeneous coordinate: [n_kpts, 3] -> [n_kpts, 4]
        ones: Float32[Tensor, "n_kpts 1"] = torch.ones(n_kpts, 1, device=self.device, dtype=keypoints_cam.dtype)
        kpts_cam_hom: Float32[Tensor, "n_kpts 4"] = torch.cat([keypoints_cam, ones], dim=-1)

        # Transform from camera to world coordinates: [n_kpts, 4] @ [4, 4].T -> [n_kpts, 4]
        kpts_world_hom: Float32[Tensor, "n_kpts 4"] = kpts_cam_hom @ self.first_world_T_cam.T

        # Reshape for batched matmul: [1, 4, n_kpts]
        kpts_world_batched: Float32[Tensor, "1 4 n_kpts"] = kpts_world_hom.T.unsqueeze(0)

        # P: [n_views, 3, 4] @ [1, 4, n_kpts] -> [n_views, 3, n_kpts]
        uv_hom: Float32[Tensor, "n_views 3 n_kpts"] = self.P @ kpts_world_batched

        # Transpose to [n_views, n_kpts, 3]
        uv_hom = uv_hom.permute(0, 2, 1)

        # Convert from homogeneous: divide by w (with numerical stability)
        w: Float32[Tensor, "n_views n_kpts 1"] = uv_hom[..., 2:3]
        eps: float = 1e-8
        w_safe: Float32[Tensor, "n_views n_kpts 1"] = torch.where(
            torch.abs(w) < eps,
            torch.sign(w) * eps,
            w,
        )
        uv: Float32[Tensor, "n_views n_kpts 2"] = uv_hom[..., :2] / w_safe

        return uv

    def optimize_frame(
        self,
        initial_predictions: list[FinalPosePrediction],
        keypoints_2d_per_view: Float[ndarray, "n_views n_kpts 3"],
    ) -> OptimizationResult:
        """Optimize mesh parameters for a single frame.

        Takes initial per-camera predictions and 2D keypoint detections,
        then refines parameters to minimize multiview consistency loss.

        Args:
            initial_predictions: Initial body predictions from SAM3DBody per camera.
                Uses the first valid prediction for initialization.
            keypoints_2d_per_view: 2D keypoints from all cameras with confidence.
                Shape ``[n_views, n_kpts, 3]`` where last dim is [u, v, conf].

        Returns:
            OptimizationResult with optimized parameters and loss history.
        """
        # Get initial parameters from first valid prediction
        init_pred: FinalPosePrediction = initial_predictions[0]

        # Initialize optimizable parameters from network prediction
        # Network provides good initialization, we refine with multiview constraints
        global_rot: Float32[Tensor, "3"] = torch.from_numpy(
            init_pred.global_rot.astype(np.float32)
        ).to(self.device).requires_grad_(self.config.optimize_global_rot)

        body_pose: Float32[Tensor, "133"] = torch.from_numpy(
            init_pred.body_pose_params.astype(np.float32)
        ).to(self.device).requires_grad_(self.config.optimize_body_pose)

        translation: Float32[Tensor, "3"] = torch.from_numpy(
            init_pred.pred_cam_t.astype(np.float32)
        ).to(self.device).requires_grad_(self.config.optimize_translation)

        # Triangulate 3D keypoints for supervision
        triangulated_3d: Float32[ndarray, "n_kpts 4"] = self._triangulate_keypoints(keypoints_2d_per_view)

        triangulated_kpts: Float32[Tensor, "n_kpts 3"] = torch.from_numpy(
            triangulated_3d[:, :3].astype(np.float32)
        ).to(self.device)

        triangulated_conf: Float32[Tensor, "n_kpts"] = torch.from_numpy(
            triangulated_3d[:, 3].astype(np.float32)
        ).to(self.device)

        # Target 2D keypoints per view
        target_2d: Float32[Tensor, "n_views n_kpts 2"] = torch.from_numpy(
            keypoints_2d_per_view[:, :, :2].astype(np.float32)
        ).to(self.device)

        confidence_2d: Float32[Tensor, "n_views n_kpts"] = torch.from_numpy(
            keypoints_2d_per_view[:, :, 2].astype(np.float32)
        ).to(self.device)

        # Build parameter list for optimizer
        params: list[Tensor] = []
        if self.config.optimize_global_rot:
            params.append(global_rot)
        if self.config.optimize_body_pose:
            params.append(body_pose)
        if self.config.optimize_translation:
            params.append(translation)

        if len(params) == 0:
            # Nothing to optimize - return initial values
            return OptimizationResult(
                global_rot=init_pred.global_rot.astype(np.float32),
                body_pose=init_pred.body_pose_params.astype(np.float32),
                translation=init_pred.pred_cam_t.astype(np.float32),
                triangulated_kpts_3d=triangulated_3d,
                loss_history=[],
                final_loss=MultiviewLossOutput(
                    total=torch.tensor(0.0),
                    reprojection_2d=torch.tensor(0.0),
                    keypoint_3d=torch.tensor(0.0),
                    temporal=torch.tensor(0.0),
                ),
            )

        optimizer = torch.optim.Adam(params, lr=self.config.learning_rate)
        loss_history: list[float] = []

        # Get fixed shape/scale/expression params from initial prediction (not optimized)
        shape_params: Float32[Tensor, "1 45"] = torch.from_numpy(
            init_pred.shape_params[np.newaxis, :].astype(np.float32)
        ).to(self.device)
        scale_params: Float32[Tensor, "1 28"] = torch.from_numpy(
            init_pred.scale_params[np.newaxis, :].astype(np.float32)
        ).to(self.device)
        hand_params: Float32[Tensor, "1 108"] = torch.from_numpy(
            init_pred.hand_pose_params[np.newaxis, :].astype(np.float32)
        ).to(self.device)
        expr_params: Float32[Tensor, "1 72"] = torch.from_numpy(
            init_pred.expr_params[np.newaxis, :].astype(np.float32)
        ).to(self.device)

        # Optimization loop
        final_loss: MultiviewLossOutput | None = None
        for _iter in range(self.config.num_iterations):
            optimizer.zero_grad()

            # Add batch dimension for mhr_forward
            global_rot_batch: Float32[Tensor, "1 3"] = global_rot.unsqueeze(0)
            body_pose_batch: Float32[Tensor, "1 body_dim"] = body_pose.unsqueeze(0)

            # NOTE: mhr_forward uses global_trans=0 internally, and pred_cam_t is
            # applied as a camera-space offset AFTER the FK computation.
            # So we pass zeros here and add translation to keypoints afterward.
            zero_trans: Float32[Tensor, "1 3"] = torch.zeros(1, 3, device=self.device, dtype=torch.float32)

            # Run MHR forward kinematics to get keypoints from pose parameters
            mhr_output = self.mhr_head.mhr_forward(
                global_trans=zero_trans,  # Always zero, translation applied post-FK
                global_rot=global_rot_batch,
                body_pose_params=body_pose_batch,
                hand_pose_params=hand_params,
                scale_params=scale_params,
                shape_params=shape_params,
                expr_params=expr_params,
                return_keypoints=True,
                return_joint_coords=True,
            )
            # mhr_output = (vertices, keypoints, joint_coords)
            vertices: Float32[Tensor, "1 n_verts 3"]
            keypoints_308: Float32[Tensor, "1 308 3"]
            joint_coords: Float32[Tensor, "1 n_joints 3"]
            vertices, keypoints_308, joint_coords = mhr_output[0], mhr_output[1], mhr_output[2]

            # Slice from 308 -> 70 keypoints (MHR70 format, same as head.forward)
            keypoints: Float32[Tensor, "1 70 3"] = keypoints_308[:, :70, :]

            # Apply camera system flip (same as head.forward: Y/Z *= -1)
            keypoints = keypoints.clone()
            keypoints[..., 1] *= -1
            keypoints[..., 2] *= -1

            # Remove batch dimension and add camera-space translation
            # (translation is applied AFTER FK, same as network's forward)
            current_kpts_cam: Float32[Tensor, "70 3"] = keypoints.squeeze(0) + translation

            # Debug: at iter 0, compare FK keypoints with network's original pred_keypoints_3d
            if _iter == 0:
                # Network keypoints need pred_cam_t added (pred_keypoints_3d is before translation)
                net_kpts_raw: Float32[Tensor, "70 3"] = torch.from_numpy(
                    init_pred.pred_keypoints_3d.astype(np.float32)
                ).to(self.device)
                net_cam_t: Float32[Tensor, "3"] = torch.from_numpy(
                    init_pred.pred_cam_t.astype(np.float32)
                ).to(self.device)
                net_kpts: Float32[Tensor, "70 3"] = net_kpts_raw + net_cam_t
                fk_kpts: Float32[Tensor, "70 3"] = current_kpts_cam
                dist: float = float(torch.mean(torch.norm(fk_kpts - net_kpts, dim=1)))
                print(f"  [DEBUG] iter 0: FK vs network keypoints dist = {dist:.4f}m")

            # Project to 2D (this internally transforms cam->world)
            projected_2d: Float32[Tensor, "n_views 70 2"] = self._project_keypoints(current_kpts_cam)

            # Transform to world for 3D loss (same transform as in _project_keypoints)
            n_kpts: int = current_kpts_cam.shape[0]
            ones: Float32[Tensor, "n_kpts 1"] = torch.ones(n_kpts, 1, device=self.device, dtype=current_kpts_cam.dtype)
            kpts_cam_hom: Float32[Tensor, "n_kpts 4"] = torch.cat([current_kpts_cam, ones], dim=-1)
            kpts_world_hom: Float32[Tensor, "n_kpts 4"] = kpts_cam_hom @ self.first_world_T_cam.T
            current_kpts_world: Float32[Tensor, "n_kpts 3"] = kpts_world_hom[:, :3]

            # Compute loss
            loss_output: MultiviewLossOutput = self.loss_fn(
                projected_2d=projected_2d,
                target_2d=target_2d,
                mesh_kpts_3d=current_kpts_world,  # Now in world space!
                triangulated_3d=triangulated_kpts,  # Also in world space
                confidence_2d=confidence_2d,
                confidence_3d=triangulated_conf,
                pose_curr=body_pose if self.config.optimize_body_pose else None,
                pose_prev=self._prev_body_pose,
                trans_curr=translation if self.config.optimize_translation else None,
                trans_prev=self._prev_translation,
            )

            loss: Tensor = loss_output.total
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())
            final_loss = loss_output

            # Debug: print loss components at iter 0 and final
            if _iter == 0 or _iter == self.config.num_iterations - 1:
                print(f"  [OPT] iter={_iter}: total={loss.item():.2f}, 2D={loss_output.reprojection_2d.item():.2f}, 3D={loss_output.keypoint_3d.item():.4f}")

        # Update temporal state for next frame
        self._prev_global_rot = global_rot.detach().clone()
        self._prev_body_pose = body_pose.detach().clone()
        self._prev_translation = translation.detach().clone()

        return OptimizationResult(
            global_rot=global_rot.detach().cpu().numpy(),
            body_pose=body_pose.detach().cpu().numpy(),
            translation=translation.detach().cpu().numpy(),
            triangulated_kpts_3d=triangulated_3d,
            loss_history=loss_history,
            final_loss=final_loss or MultiviewLossOutput(
                total=torch.tensor(0.0),
                reprojection_2d=torch.tensor(0.0),
                keypoint_3d=torch.tensor(0.0),
                temporal=torch.tensor(0.0),
            ),
        )


__all__ = [
    "MultiviewOptimizerConfig",
    "OptimizationResult",
    "MultiviewBodyOptimizer",
]
