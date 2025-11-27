# Copyright (c) Meta Platforms, Inc. and affiliates.

from collections.abc import Callable
from typing import Any, TypedDict, cast

import numpy as np
import torch
from jaxtyping import Float, UInt8
from numpy import ndarray
from torch import Tensor
from torch.utils.data import default_collate


class PreparedBatchDict(TypedDict, total=False):
    img: Float[Tensor, "B N 3 H W"]
    img_size: Float[Tensor, "B N 2"]
    ori_img_size: Float[Tensor, "B N 2"]
    bbox_center: Float[Tensor, "B N 2"]
    bbox_scale: Float[Tensor, "B N 2"]
    bbox: Float[Tensor, "B N 4"]
    affine_trans: Float[Tensor, "B N 2 3"]
    mask: Float[Tensor, "B N 1 H W"]
    mask_score: Float[Tensor, "B N"]
    cam_int: Float[Tensor, "B 3 3"]
    person_valid: Float[Tensor, "B N"]
    img_ori: list["NoCollate"]


class NoCollate:
    def __init__(self, data: Any) -> None:
        self.data: Any = data


def prepare_batch(
    img: UInt8[ndarray, "h w 3"],
    transform: Callable[[dict[str, Any]], dict[str, Any]],
    boxes: Float[ndarray, "n 4"],
    masks: Float[ndarray, "n h w"] | None = None,
    masks_score: Float[ndarray, "n"] | None = None,
    cam_int: Float[Tensor, "B 3 3"] | None = None,
) -> PreparedBatchDict:
    """A helper function to prepare data batch for SAM 3D Body model inference."""
    height, width = img.shape[:2]

    # construct batch data samples
    data_list: list[dict[str, Any]] = []
    for idx in range(boxes.shape[0]):
        data_info: dict[str, Any] = dict(img=img)
        data_info["bbox"] = boxes[idx]  # shape (4,)
        data_info["bbox_format"] = "xyxy"

        if masks is not None:
            data_info["mask"] = masks[idx].astype(np.float32, copy=False)
            if masks_score is not None:
                data_info["mask_score"] = masks_score[idx]
            else:
                data_info["mask_score"] = np.array(1.0, dtype=np.float32)
        else:
            data_info["mask"] = np.zeros((height, width, 1), dtype=np.uint8)
            data_info["mask_score"] = np.array(0.0, dtype=np.float32)

        data_list.append(transform(data_info))

    batch = default_collate(data_list)

    max_num_person = batch["img"].shape[0]
    for key in [
        "img",
        "img_size",
        "ori_img_size",
        "bbox_center",
        "bbox_scale",
        "bbox",
        "affine_trans",
        "mask",
        "mask_score",
    ]:
        if key in batch:
            batch[key] = batch[key].unsqueeze(0).float()
    if "mask" in batch:
        batch["mask"] = batch["mask"].unsqueeze(2)
    batch["person_valid"] = torch.ones((1, max_num_person))

    if cam_int is not None:
        batch["cam_int"] = cam_int.to(batch["img"])
    else:
        # Default camera intrinsics according image size
        batch["cam_int"] = torch.tensor(
            [
                [
                    [(height**2 + width**2) ** 0.5, 0, width / 2.0],
                    [0, (height**2 + width**2) ** 0.5, height / 2.0],
                    [0, 0, 1],
                ]
            ],
        ).to(batch["img"])

    batch["img_ori"] = [NoCollate(img)]
    return cast(PreparedBatchDict, batch)
