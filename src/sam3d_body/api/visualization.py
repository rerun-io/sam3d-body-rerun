from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Bool, Float32, Int, UInt8
from monopriors.depth_utils import depth_edges_mask
from monopriors.relative_depth_models import RelativeDepthPrediction
from numpy import ndarray
from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.ops.pc_utils import estimate_voxel_size
from simplecv.rerun_log_utils import log_pinhole

from sam3d_body.metadata.mhr70 import MHR70_ID2NAME, MHR70_IDS, MHR70_LINKS
from sam3d_body.sam_3d_body_estimator import FinalPosePrediction

BOX_PALETTE: UInt8[np.ndarray, "n_colors 4"] = np.array(
    [
        [255, 99, 71, 255],  # tomato
        [65, 105, 225, 255],  # royal blue
        [60, 179, 113, 255],  # medium sea green
        [255, 215, 0, 255],  # gold
        [138, 43, 226, 255],  # blue violet
        [255, 140, 0, 255],  # dark orange
        [220, 20, 60, 255],  # crimson
        [70, 130, 180, 255],  # steel blue
    ],
    dtype=np.uint8,
)

# Use a separate id range for segmentation classes to avoid clobbering the person class (id=0).
SEG_CLASS_OFFSET = 1000  # background = 1000, persons start at 1001
MAX_POINT_CLOUD_POINTS = 50_000
MIN_DEPTH_CONFIDENCE = 0.5


def filter_out_of_bounds(
    uv: Float32[ndarray, "n_points 2"],
    h: int,
    w: int,
    xyz_cam: Float32[ndarray, "n_points 3"] | None = None,
) -> Float32[ndarray, "n_points 2"]:
    """Return a copy of ``uv`` with off-screen (and optional behind-camera) points masked.

    Args:
        uv: Pixel coordinates ``[N, 2]`` in (u, v) order.
        h: Image height in pixels.
        w: Image width in pixels.
        xyz_cam: Optional camera-frame coordinates ``[N, 3]`` to mask points with negative ``z``.

    Returns:
        Copy of ``uv`` where out-of-bounds rows are set to ``NaN`` so Rerun hides them.
    """

    uv_filtered: Float32[ndarray, "n_points 2"] = np.asarray(uv, dtype=np.float32).copy()

    out_of_bounds: Bool[ndarray, "n_points"] = np.logical_or(uv_filtered[:, 0] >= float(w), uv_filtered[:, 0] < 0.0)
    out_of_bounds = np.logical_or(out_of_bounds, uv_filtered[:, 1] >= float(h))
    out_of_bounds = np.logical_or(out_of_bounds, uv_filtered[:, 1] < 0.0)

    if xyz_cam is not None:
        out_of_bounds = np.logical_or(out_of_bounds, xyz_cam[:, 2] < 0.0)

    uv_filtered[out_of_bounds, :] = np.nan
    return uv_filtered


def compute_vertex_normals(
    verts: Float32[ndarray, "n_verts 3"],
    faces: Int[ndarray, "n_faces 3"],
    eps: float = 1e-12,
) -> Float32[ndarray, "n_verts 3"]:
    """Compute per-vertex normals for a single mesh.

    Args:
        verts: Float32 array of vertex positions with shape ``(n_verts, 3)``.
        faces: Int array of triangle indices with shape ``(n_faces, 3)``.
        eps: Small epsilon to avoid division by zero when normalizing.

    Returns:
        Float32 array of unit vertex normals with shape ``(n_verts, 3)``; zeros for degenerate vertices.
    """

    # Expand faces to vertex triplets and fetch their positions.
    faces_i: Int[ndarray, "n_faces 3"] = faces.astype(np.int64)
    v0: Float32[ndarray, "n_faces 3"] = verts[faces_i[:, 0]]
    v1: Float32[ndarray, "n_faces 3"] = verts[faces_i[:, 1]]
    v2: Float32[ndarray, "n_faces 3"] = verts[faces_i[:, 2]]

    # Face normal = cross(edge1, edge2).
    e1: Float32[ndarray, "n_faces 3"] = v1 - v0
    e2: Float32[ndarray, "n_faces 3"] = v2 - v0
    face_normals: Float32[ndarray, "n_faces 3"] = np.cross(e1, e2)

    # Accumulate each face normal into its three vertices with a vectorized scatter-add.
    vertex_normals: Float32[ndarray, "n_verts 3"] = np.zeros_like(verts, dtype=np.float32)
    flat_indices: Int[ndarray, "n_faces3"] = faces_i.reshape(-1)
    face_normals_repeated: Float32[ndarray, "n_faces3 3"] = np.repeat(face_normals, 3, axis=0)
    np.add.at(vertex_normals, flat_indices, face_normals_repeated)

    norms: Float32[ndarray, "n_verts 1"] = np.linalg.norm(vertex_normals, axis=-1, keepdims=True)
    denom: Float32[ndarray, "n_verts 1"] = np.maximum(norms, eps).astype(np.float32)
    vn_unit: Float32[ndarray, "n_verts 3"] = (vertex_normals / denom).astype(np.float32)
    mask: ndarray = norms > eps
    vn_unit = np.where(mask, vn_unit, np.float32(0.0))
    return vn_unit


def set_annotation_context() -> None:
    """Register MHR-70 semantic metadata so subsequent logs show names/edges and mask colors."""
    # Base person class (for keypoints / boxes) uses id=0 (original), segmentation uses 1000+ to avoid clashes.
    person_class = rr.ClassDescription(
        info=rr.AnnotationInfo(id=0, label="Person", color=(0, 0, 255)),
        keypoint_annotations=[rr.AnnotationInfo(id=idx, label=name) for idx, name in MHR70_ID2NAME.items()],
        keypoint_connections=MHR70_LINKS,
    )

    # Segmentation classes: id=SEG_CLASS_OFFSET background, ids SEG_CLASS_OFFSET+1..n for each instance color.
    seg_classes: list[rr.ClassDescription] = [
        rr.ClassDescription(info=rr.AnnotationInfo(id=SEG_CLASS_OFFSET, label="Background", color=(64, 64, 64))),
    ]
    for idx, color in enumerate(BOX_PALETTE[:, :3].tolist(), start=1):
        seg_classes.append(
            rr.ClassDescription(
                info=rr.AnnotationInfo(
                    id=SEG_CLASS_OFFSET + idx, label=f"Person-{idx}", color=tuple(int(c) for c in color)
                ),
            )
        )

    rr.log(
        "/",
        rr.AnnotationContext([person_class, *seg_classes]),
        static=True,
    )


def visualize_sample(
    pred_list: list[FinalPosePrediction],
    rgb_hw3: UInt8[ndarray, "h w 3"],
    parent_log_path: Path,
    faces: Int[ndarray, "n_faces 3"],
    relative_depth_pred: RelativeDepthPrediction | None = None,
) -> None:
    h: int = rgb_hw3.shape[0]
    w: int = rgb_hw3.shape[1]
    cam_log_path: Path = parent_log_path / "cam"
    pinhole_log_path: Path = cam_log_path / "pinhole"
    image_log_path: Path = pinhole_log_path / "image"
    pred_log_path: Path = pinhole_log_path / "pred"
    # log the pinhole camera parameters (assume fx=fy and center at image center)
    focal_length: float = float(pred_list[0].focal_length)
    intri: Intrinsics = Intrinsics(
        camera_conventions="RDF",
        fl_x=focal_length,
        fl_y=focal_length,
        cx=float(w) / 2.0,
        cy=float(h) / 2.0,
        height=h,
        width=w,
    )
    world_T_cam: Float32[ndarray, "4 4"] = np.eye(4, dtype=np.float32)
    extri: Extrinsics = Extrinsics(
        world_R_cam=world_T_cam[:3, :3],
        world_t_cam=world_T_cam[:3, 3],
    )

    pinhole_params: PinholeParameters = PinholeParameters(intrinsics=intri, extrinsics=extri, name="pinhole")
    log_pinhole(camera=pinhole_params, cam_log_path=cam_log_path)
    # clear the previous pred logs
    rr.log(f"{pred_log_path}", rr.Clear(recursive=True))
    rr.log(f"{image_log_path}", rr.Image(rgb_hw3, color_model=rr.ColorModel.RGB).compress(jpeg_quality=90))

    # Build per-pixel maps (SEG_CLASS_OFFSET = background). Also build RGBA overlay with transparent background.
    seg_map: Int[ndarray, "h w"] = np.full((h, w), SEG_CLASS_OFFSET, dtype=np.int32)
    seg_overlay: UInt8[ndarray, "h w 4"] = np.zeros((h, w, 4), dtype=np.uint8)
    human_mask: Bool[ndarray, "h w"] = np.zeros((h, w), dtype=bool)

    mesh_root_path: Path = parent_log_path / "pred"
    rr.log(str(mesh_root_path), rr.Clear(recursive=True))

    for i, output in enumerate(pred_list):
        box_color: UInt8[ndarray, "1 4"] = BOX_PALETTE[i % len(BOX_PALETTE)].reshape(1, 4)
        rr.log(
            f"{pred_log_path}/bbox_{i}",
            rr.Boxes2D(
                array=output.bbox,
                array_format=rr.Box2DFormat.XYXY,
                class_ids=0,
                colors=box_color,
                show_labels=True,
            ),
        )

        kpts_cam: Float32[ndarray, "n_kpts 3"] = np.ascontiguousarray(output.pred_keypoints_3d, dtype=np.float32)
        kpts_uv: Float32[ndarray, "n_kpts 2"] = np.ascontiguousarray(output.pred_keypoints_2d, dtype=np.float32)
        kpts_uv_in_bounds: Float32[ndarray, "n_kpts 2"] = filter_out_of_bounds(
            uv=kpts_uv,
            h=h,
            w=w,
            xyz_cam=None,  # Depth sign from the model can be negative; only cull by image bounds.
        )
        rr.log(
            f"{pred_log_path}/uv_{i}",
            rr.Points2D(
                positions=kpts_uv_in_bounds,
                keypoint_ids=MHR70_IDS,
                class_ids=0,
                colors=(0, 255, 0),
            ),
        )

        # Accumulate segmentation masks (if present) into a single segmentation image.
        mask = output.mask
        if mask is not None:
            mask_arr: ndarray = np.asarray(mask).squeeze()
            if mask_arr.shape != seg_map.shape:
                mask_arr = cv2.resize(
                    mask_arr.astype(np.uint8), (seg_map.shape[1], seg_map.shape[0]), interpolation=cv2.INTER_NEAREST
                )
            mask_bool = mask_arr.astype(bool)
            human_mask = np.logical_or(human_mask, mask_bool)
            seg_id = SEG_CLASS_OFFSET + i + 1  # keep person class (0) separate from seg classes
            seg_map = np.where(mask_bool, np.uint16(seg_id), seg_map)

            # Color overlay for this instance, background stays transparent.
            color = BOX_PALETTE[i % len(BOX_PALETTE), :3]
            seg_overlay[mask_bool] = np.array([color[0], color[1], color[2], 120], dtype=np.uint8)

        # Log 3D keypoints in world coordinates
        cam_t: Float32[ndarray, "3"] = np.ascontiguousarray(output.pred_cam_t, dtype=np.float32)
        kpts_world: Float32[ndarray, "n_kpts 3"] = np.ascontiguousarray(kpts_cam + cam_t, dtype=np.float32)
        rr.log(
            f"{parent_log_path}/pred/kpts3d_{i}",
            rr.Points3D(
                positions=kpts_world,
                keypoint_ids=MHR70_IDS,
                class_ids=0,
                colors=(0, 255, 0),
            ),
        )

        # Log the full-body mesh in world coordinates so it shows in 3D
        verts_cam: Float32[ndarray, "n_verts 3"] = np.ascontiguousarray(output.pred_vertices, dtype=np.float32)
        verts_world: Float32[ndarray, "n_verts 3"] = np.ascontiguousarray(verts_cam + cam_t, dtype=np.float32)
        faces_int: Int[ndarray, "n_faces 3"] = np.ascontiguousarray(faces, dtype=np.int32)
        vertex_normals: Float32[ndarray, "n_verts 3"] = compute_vertex_normals(verts_world, faces_int)
        rr.log(
            f"{parent_log_path}/pred/mesh_{i}",
            rr.Mesh3D(
                vertex_positions=verts_world,
                triangle_indices=faces_int,
                vertex_normals=vertex_normals,
                albedo_factor=(
                    float(box_color[0, 0]) / 255.0,
                    float(box_color[0, 1]) / 255.0,
                    float(box_color[0, 2]) / 255.0,
                    0.35,
                ),
            ),
        )

    # Log segmentation ids (full map) and an RGBA overlay with transparent background.
    if np.any(seg_map != SEG_CLASS_OFFSET):
        rr.log(f"{pred_log_path}/segmentation_ids", rr.SegmentationImage(seg_map))
        rr.log(f"{pred_log_path}/segmentation_overlay", rr.Image(seg_overlay, color_model=rr.ColorModel.RGBA))

    # Optionally log depth and a background-only point cloud (for 3D view only).
    if relative_depth_pred is not None:
        depth_hw: Float32[ndarray, "h w"] = np.asarray(relative_depth_pred.depth, dtype=np.float32)
        conf_hw: Float32[ndarray, "h w"] = np.asarray(relative_depth_pred.confidence, dtype=np.float32)
        if depth_hw.shape != (h, w):
            depth_hw = cv2.resize(depth_hw, (w, h), interpolation=cv2.INTER_NEAREST)
        if conf_hw.shape != (h, w):
            conf_hw = cv2.resize(conf_hw, (w, h), interpolation=cv2.INTER_NEAREST)
        depth_hw = np.nan_to_num(depth_hw, nan=0.0, posinf=0.0, neginf=0.0)

        # Remove flying pixels along depth discontinuities.
        edges_mask: Bool[ndarray, "h w"] = depth_edges_mask(depth_hw, threshold=0.01)
        depth_hw = depth_hw * np.logical_not(edges_mask)

        # Remove low-confidence pixels.
        conf_mask: Bool[ndarray, "h w"] = conf_hw >= MIN_DEPTH_CONFIDENCE
        depth_hw = depth_hw * conf_mask

        background_mask: Bool[ndarray, "h w"] = np.logical_not(human_mask)
        depth_bg: Float32[ndarray, "h w"] = depth_hw * background_mask

        # Log depth image (not referenced by the 2D blueprint).
        # rr.log(f"{pinhole_log_path}/depth", rr.DepthImage(depth_bg, meter=1.0))

        fx: float = float(relative_depth_pred.K_33[0, 0])
        fy: float = float(relative_depth_pred.K_33[1, 1])
        cx: float = float(relative_depth_pred.K_33[0, 2])
        cy: float = float(relative_depth_pred.K_33[1, 2])

        u: Float32[ndarray, "w"] = np.arange(w, dtype=np.float32)
        v: Float32[ndarray, "h"] = np.arange(h, dtype=np.float32)
        uu: Float32[ndarray, "h w"]
        vv: Float32[ndarray, "h w"]
        uu, vv = np.meshgrid(u, v)

        z_cam: Float32[ndarray, "h w"] = depth_bg
        valid: Bool[ndarray, "h w"] = np.logical_and(z_cam > 0.0, np.isfinite(z_cam))
        if np.any(valid):
            x_cam: Float32[ndarray, "h w"] = (uu - cx) * z_cam / fx
            y_cam: Float32[ndarray, "h w"] = (vv - cy) * z_cam / fy
            points_cam: Float32[ndarray, "h w 3"] = np.stack([x_cam, y_cam, z_cam], axis=-1)

            points_flat: Float32[ndarray, "n_valid 3"] = points_cam[valid]
            colors_flat: UInt8[ndarray, "n_valid 3"] = rgb_hw3[valid]

            if points_flat.shape[0] > MAX_POINT_CLOUD_POINTS:
                voxel_size: float = estimate_voxel_size(
                    points_flat, target_points=MAX_POINT_CLOUD_POINTS, tolerance=0.25
                )
                pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_flat)
                pcd.colors = o3d.utility.Vector3dVector(colors_flat.astype(np.float32) / 255.0)
                pcd_ds: o3d.geometry.PointCloud = pcd.voxel_down_sample(voxel_size)
                points_flat = np.asarray(pcd_ds.points, dtype=np.float32)
                colors_flat = (np.asarray(pcd_ds.colors, dtype=np.float32) * 255.0).astype(np.uint8)

            rr.log(
                f"{parent_log_path}/depth_point_cloud",
                rr.Points3D(
                    positions=points_flat,
                    colors=colors_flat,
                ),
            )


def create_view() -> rrb.ContainerLike:
    view_2d = rrb.Vertical(
        contents=[
            # Top: people-only overlay on the RGB image.
            rrb.Spatial2DView(
                name="image",
                origin="/world/cam/pinhole",
                contents=[
                    "/world/cam/pinhole/image",
                    "/world/cam/pinhole/pred/segmentation_overlay",
                ],
            ),
            # Bottom: 2D boxes + keypoints; segmentation hidden.
            rrb.Spatial2DView(
                name="mhr",
                origin="/world/cam/pinhole",
                contents=[
                    "/world/cam/pinhole/image",
                    "/world/cam/pinhole/pred/**",
                    "- /world/cam/pinhole/pred/segmentation_overlay/**",
                    "- /world/cam/pinhole/pred/segmentation_ids/**",
                ],
            ),
        ],
    )
    view_3d = rrb.Spatial3DView(name="mhr_3d", line_grid=rrb.LineGrid3D(visible=False))
    main_view = rrb.Horizontal(contents=[view_2d, view_3d], column_shares=[2, 3])
    view = rrb.Tabs(contents=[main_view], name="sam-3d-body-demo")
    return view
