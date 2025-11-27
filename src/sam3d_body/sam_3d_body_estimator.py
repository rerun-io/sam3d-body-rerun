"""Core inference utilities for SAM 3D Body model (hands + body fusion)."""

from collections.abc import Callable
from typing import Any, Literal, cast

import numpy as np
import torch
from jaxtyping import Float, Int, UInt8
from numpy import ndarray
from serde import from_dict, serde
from torch import Tensor
from torchvision.transforms import ToTensor

from sam3d_body.data.transforms import (
    Compose,
    GetBBoxCenterScale,
    TopdownAffine,
    VisionTransformWrapper,
)
from sam3d_body.data.utils.prepare_batch import PreparedBatchDict, prepare_batch
from sam3d_body.models.meta_arch import SAM3DBody
from sam3d_body.models.meta_arch.sam3d_body import BodyPredContainer
from sam3d_body.utils import recursive_to


@serde
class PoseOutputsNP:
    """Batch pose outputs in numpy form straight from the model forward pass."""

    pred_pose_raw: Float[ndarray, "n pose_raw=266"]
    """Raw 266D pose vector per item (SMPL-X ordering)."""
    pred_pose_rotmat: Float[ndarray, ""] | None
    """Optional rotation matrices derived from ``pred_pose_raw``."""
    global_rot: Float[ndarray, "n 3"]
    """Root/global rotation in radians for each item (XYZ Euler)."""
    body_pose: Float[ndarray, "n body_pose_params=133"]
    """Body pose parameters (133D continuous) per item."""
    shape: Float[ndarray, "n shape_params=45"]
    """Body shape PCA coefficients (45D) per item."""
    scale: Float[ndarray, "n scale_params=28"]
    """Body scale PCA coefficients (28D) per item."""
    hand: Float[ndarray, "n hand_pose_params=108"]
    """Hand pose parameters in PCA space (108D) per item."""
    face: Float[ndarray, "n expr_params=72"]
    """Facial expression PCA coefficients (72D) per item."""
    pred_keypoints_3d: Float[ndarray, "n joints3d 3"]
    """3D keypoints in camera coordinates for each item."""
    pred_vertices: Float[ndarray, "n verts=18439 3"]
    """Full mesh vertices in camera coordinates per item."""
    pred_joint_coords: Float[ndarray, "n joints3d 3"]
    """Internal skeleton joint centers (camera coordinates) per item."""
    faces: Int[ndarray, "faces 3"]
    """Mesh face indices shared across items."""
    joint_global_rots: Float[ndarray, "n joints_rot 3 3"]
    """Global rotation matrices per joint for each item."""
    mhr_model_params: Float[ndarray, "n mhr_params"]
    """Model hyper-regularization parameters per item."""
    pred_cam: Float[ndarray, "n 3"]
    """Weak-perspective camera parameters (sx, sy, tx) per item."""
    pred_keypoints_2d_verts: Float[ndarray, "n verts 2"]
    """2D projected vertices per item (pixels)."""
    pred_keypoints_2d: Float[ndarray, "n joints2d 2"]
    """2D projected keypoints per item (pixels)."""
    pred_cam_t: Float[ndarray, "n 3"]
    """Camera-space translation vectors applied to each mesh."""
    focal_length: Float[ndarray, "n"]
    """Focal lengths per item (pixels)."""
    pred_keypoints_2d_depth: Float[ndarray, "n joints2d"]
    """Depth values for 2D keypoints per item."""
    pred_keypoints_2d_cropped: Float[ndarray, "n joints2d 2"]
    """2D keypoints in the cropped input frame per item (pixels)."""


@serde
class FinalPosePrediction:
    """Per-person prediction bundle returned by SAM 3D Body."""

    bbox: Float[ndarray, "4"]
    """Axis-aligned XYXY box in the original image (pixels)."""
    focal_length: Float[ndarray, ""]
    """Scalar focal length for the frame (pixels)."""
    pred_keypoints_3d: Float[ndarray, "joints 3"]
    """3D keypoints in camera coordinates (OpenCV: x right, y down, z forward)."""
    pred_keypoints_2d: Float[ndarray, "joints 2"]
    """2D keypoints in image pixel coordinates."""
    pred_vertices: Float[ndarray, "verts 3"]
    """Full body mesh vertices in camera coordinates."""
    pred_cam_t: Float[ndarray, "3"]
    """Camera-space translation (x, y, z) applied to the mesh."""
    pred_pose_raw: Float[ndarray, "pose_params=266"]
    """Raw 266D pose vector (SMPL-X style ordering)."""
    global_rot: Float[ndarray, "3"]
    """Root/global rotation in radians (XYZ Euler)."""
    body_pose_params: Float[ndarray, "body_pose_params=133"]
    """Body pose parameters (133D continuous)."""
    hand_pose_params: Float[ndarray, "hand_pose_params=108"]
    """Hand pose parameters (108D PCA space)."""
    scale_params: Float[ndarray, "scale_params=28"]
    """Body scale PCA coefficients (28D)."""
    shape_params: Float[ndarray, "shape_params=45"]
    """Body shape PCA coefficients (45D)."""
    expr_params: Float[ndarray, "expr_params=72"]
    """Facial expression PCA coefficients (72D)."""
    mask: UInt8[ndarray, "h w 1"] | None = None
    """Optional instance segmentation mask (H×W×1, uint8)."""
    pred_joint_coords: Float[ndarray, "joints 3"] | None = None
    """Full internal skeleton joint centers (camera coordinates)."""
    pred_global_rots: Float[ndarray, "joints 3 3"] | None = None
    """Global rotation matrices per joint aligned with ``pred_joint_coords``."""
    lhand_bbox: Float[ndarray, "4"] | None = None
    """Optional left-hand XYXY box in the original image (pixels)."""
    rhand_bbox: Float[ndarray, "4"] | None = None
    """Optional right-hand XYXY box in the original image (pixels)."""


Transform = Callable[[dict], dict | None]


class SAM3DBodyEstimator:
    """Wraps the SAM 3D Body meta-architecture for single-frame inference."""

    def __init__(
        self,
        sam_3d_body_model: SAM3DBody,
    ) -> None:
        """Initialize preprocessing pipelines and cache reusable assets.

        Args:
            sam_3d_body_model: Loaded ``SAM3DBody`` instance (checkpoints already restored).
        """
        self.model: SAM3DBody = sam_3d_body_model
        self.thresh_wrist_angle: float = 1.4

        # For mesh visualization
        self.faces: Int[ndarray, "n_faces=36874 3"] = self.model.head_pose.faces.cpu().numpy()  # type: ignore

        # Define transforms
        body_transforms: list[Transform] = [
            cast(Transform, GetBBoxCenterScale()),
            cast(Transform, TopdownAffine(input_size=512, use_udp=False)),
            cast(Transform, VisionTransformWrapper(ToTensor())),
        ]
        hand_transforms: list[Transform] = [
            cast(Transform, GetBBoxCenterScale(padding=0.9)),
            cast(Transform, TopdownAffine(input_size=512, use_udp=False)),
            cast(Transform, VisionTransformWrapper(ToTensor())),
        ]

        self.transform: Compose = Compose(body_transforms)
        self.transform_hand: Compose = Compose(hand_transforms)

    @torch.no_grad()
    def process_one_image(
        self,
        rgb_hw3: UInt8[ndarray, "h w 3"],
        xyxy: Float[ndarray, "n 4"] | None = None,
        masks: Float[ndarray, "n h w"] | None = None,
        masks_score: Float[ndarray, "n"] | None = None,
        K_33: Float[ndarray, "3 3"] | None = None,
        inference_type: Literal["full", "body", "hand"] = "full",
    ) -> list[FinalPosePrediction]:
        """Run full SAM 3D Body inference for one RGB frame.

        Args:
            rgb_hw3: Input image in RGB order with dtype ``uint8`` and shape ``[H, W, 3]``.
            xyxy: Optional person boxes (XYXY, pixels) to bypass detector; defaults to the
                full-frame box when ``None``.
            masks: Optional binary instance masks aligned with ``xyxy`` (shape ``[N, H, W]``);
                when provided, segmentation is skipped.
            masks_score: Optional confidence scores for ``masks``.
            K_33: Optional camera intrinsic matrix ``[3, 3]``; if ``None``, the model will rely on
                its default relative-FOV heuristic. Intrinsics follow the project convention
                of mapping world points into the camera frame via ``cam_T_world`` style matrices.
            inference_type: Controls which decoders run: ``"full"`` (body + hands), ``"body"``
                (body-only), or ``"hand"`` (hand-only output paths).

        Returns:
            A list of ``FinalPosePrediction`` structures, one per detected person.
        """

        height: int = rgb_hw3.shape[0]
        width: int = rgb_hw3.shape[1]

        if xyxy is None:
            xyxy = np.array([0, 0, width, height], dtype=np.float32).reshape(1, 4)

        # If there are no detected humans, don't run prediction
        if len(xyxy) == 0:
            return []

        # number of people detected
        n_dets: int = xyxy.shape[0]

        #################### Construct batch data samples ####################
        batch: PreparedBatchDict = prepare_batch(rgb_hw3, self.transform, xyxy, masks, masks_score)

        #################### Run model inference on an image ####################
        batch: PreparedBatchDict = recursive_to(batch, "cuda")
        self.model._initialize_batch(batch)
        batch_img: Float[Tensor, "B=1 N 3 H W"] = batch["img"]

        # Handle camera intrinsics
        # - either provided externally or generated via default FOV estimator
        if K_33 is None:
            print("")
        else:
            K_b33: Float[Tensor, "b=1 3 3"] = torch.as_tensor(
                K_33[np.newaxis, ...], device=batch_img.device, dtype=batch_img.dtype
            )
            batch["cam_int"] = K_b33.clone()

        outputs: BodyPredContainer = self.model.run_inference(
            rgb_hw3,
            batch,
            inference_type=inference_type,
            transform_hand=self.transform_hand,
            thresh_wrist_angle=self.thresh_wrist_angle,
        )
        pose_output: dict[str, Any] = outputs.pose_output
        batch_lhand: dict[str, Any] | None = outputs.batch_lhand
        batch_rhand: dict[str, Any] | None = outputs.batch_rhand

        mhr_dict: dict[str, Any] = pose_output["mhr"]
        out_np_dict: dict[str, ndarray] = cast(dict[str, ndarray], recursive_to(recursive_to(mhr_dict, "cpu"), "numpy"))
        out_np: PoseOutputsNP = from_dict(PoseOutputsNP, out_np_dict)

        all_out: list[FinalPosePrediction] = []
        bbox_tensor: Float[Tensor, "B=1 N 4"] = batch["bbox"]

        for idx in range(n_dets):
            mask_arr: UInt8[ndarray, "h w 1"] | None = None
            if masks is not None:
                mask_arr = masks[idx]
                if mask_arr.ndim == 2:
                    mask_arr = mask_arr[..., np.newaxis]
                mask_arr = (mask_arr > 0.5).astype(np.uint8, copy=False)
            pred = FinalPosePrediction(
                bbox=bbox_tensor[0, idx].cpu().numpy(),
                focal_length=np.asarray(out_np.focal_length[idx]),
                pred_keypoints_3d=out_np.pred_keypoints_3d[idx],
                pred_keypoints_2d=out_np.pred_keypoints_2d[idx],
                pred_vertices=out_np.pred_vertices[idx],
                pred_cam_t=out_np.pred_cam_t[idx],
                pred_pose_raw=out_np.pred_pose_raw[idx],
                global_rot=out_np.global_rot[idx],
                body_pose_params=out_np.body_pose[idx],
                hand_pose_params=out_np.hand[idx],
                scale_params=out_np.scale[idx],
                shape_params=out_np.shape[idx],
                expr_params=out_np.face[idx],
                mask=mask_arr,
                pred_joint_coords=out_np.pred_joint_coords[idx],
                pred_global_rots=out_np.joint_global_rots[idx],
            )

            if inference_type == "full" and batch_lhand is not None and batch_rhand is not None:
                lhand_center = batch_lhand["bbox_center"].flatten(0, 1)[idx]
                lhand_scale = batch_lhand["bbox_scale"].flatten(0, 1)[idx]
                pred.lhand_bbox = np.array(
                    [
                        (lhand_center[0] - lhand_scale[0] / 2).item(),
                        (lhand_center[1] - lhand_scale[1] / 2).item(),
                        (lhand_center[0] + lhand_scale[0] / 2).item(),
                        (lhand_center[1] + lhand_scale[1] / 2).item(),
                    ]
                )

                rhand_center = batch_rhand["bbox_center"].flatten(0, 1)[idx]
                rhand_scale = batch_rhand["bbox_scale"].flatten(0, 1)[idx]
                pred.rhand_bbox = np.array(
                    [
                        (rhand_center[0] - rhand_scale[0] / 2).item(),
                        (rhand_center[1] - rhand_scale[1] / 2).item(),
                        (rhand_center[0] + rhand_scale[0] / 2).item(),
                        (rhand_center[1] + rhand_scale[1] / 2).item(),
                    ]
                )

            all_out.append(pred)

        return all_out
