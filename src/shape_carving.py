"""
Shape carving utilities

"""
__date__ = "November 2024 - August 2025"

import numpy as np


def create_3d_grid(length, n, volume_idx=None):
    # Create an n x n x n 3D grid centered at `center` with radius `length / 2`
    offset = np.linspace(-length / 2, length / 2, n)
    grid_x, grid_y, grid_z = np.meshgrid(offset, offset, offset, indexing='ij')
    grid_points = np.stack([grid_x, grid_y, grid_z], axis=-1)
    if volume_idx is not None:
        (i1, i2), (i3, i4), (i5,i6) = volume_idx
        grid_points = grid_points[i1:i2,i3:i4,i5:i6]
    return grid_points


def project_points(points, intrinsic_matrix, extrinsic_matrix):
    # Convert points to homogeneous coordinates
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    # Apply extrinsic transformation (world to camera space)
    camera_points = (extrinsic_matrix @ points_homogeneous.T).T
    # Apply intrinsic transformation (camera space to image plane)
    pixel_coords_homogeneous = (intrinsic_matrix @ camera_points[:, :3].T).T
    # Normalize to get pixel coordinates (divide by z to get x, y)
    pixel_coords = pixel_coords_homogeneous[:, :2] / pixel_coords_homogeneous[:, 2:]
    return pixel_coords  # (N, 2), where N is the number of points


def sample_nearest_pixels(images, pixel_coords, average=True):
    sampled_values = []
    h, w = images[0].shape[:2]
    
    for image, coords in zip(images, pixel_coords):
        # Round to nearest pixel coordinates
        pixel_x = np.clip(np.round(coords[..., 0]).astype(int), 0, w - 1)
        pixel_y = np.clip(np.round(coords[..., 1]).astype(int), 0, h - 1)
        # Sample the image at the nearest pixel coordinates
        sampled_values.append(image[pixel_y, pixel_x])

    if average:
        return np.array(sampled_values).mean(axis=0)  # Average over all images
    return np.array(sampled_values)


def shift_and_rotate_grid_points(grid_points, shift, angle, angle_offset=0.0):
    assert grid_points.ndim == 4
    assert grid_points.shape[3] == 3
    assert shift.shape == (3,)
    n1, n2, n3 = grid_points.shape[:3]
    grid_points = grid_points.reshape(-1, 3)
    c, s = np.cos(angle + angle_offset), np.sin(angle + angle_offset)
    mat = np.array([[c,-s,0],[s,c,0],[0,0,1]])
    grid_points = grid_points @ mat.T + shift[None]
    return grid_points.reshape(n1, n2, n3, 3)


def get_volume(
        images,
        intrinsic_matrices,
        extrinsic_matrices,
        grid_points,
        adaptive=False,
    ):
    """Get the unprojected images."""
    assert grid_points.ndim == 4
    assert grid_points.shape[3] == 3

    if adaptive:
        assert images.ndim == 3
        intrinsic_matrices, seed_point = adjust_principal_points_to_seed(images, intrinsic_matrices, extrinsic_matrices)

    n1, n2, n3 = grid_points.shape[:3]
    grid_points = grid_points.reshape(-1, 3)
    all_projected_coords = []
    for intrinsic, extrinsic in zip(intrinsic_matrices, extrinsic_matrices):
        projected_coords = project_points(grid_points, intrinsic, extrinsic)
        all_projected_coords.append(projected_coords)
    all_projected_coords = np.array(all_projected_coords)
    averaged_values = sample_nearest_pixels(images, all_projected_coords) # [n1*n2*n3, c]
    averaged_values = averaged_values.T.reshape(-1, n1, n2, n3) # [c, n1, n2, n3]
    if adaptive:
        return averaged_values, intrinsic_matrices, seed_point
    return averaged_values


def shape_carve_volume(mask_volume, image_volume, C=6, eps=1e-2):
    assert mask_volume.ndim == 4 # [c, n1, n2, n3]
    assert image_volume.shape[1:] == mask_volume.shape[1:]
    mult_volume = mask_volume > (C - 1.0) / C - eps
    image_volume[np.tile(mult_volume, (3, 1, 1, 1))] = 1.0
    return image_volume


def shape_carve_mask(volume, C=6, eps=1e-2):
    th1 = (C-1) / C - eps
    th2 = 1.0 - eps
    th3 = (C-2) / C - eps
    volume[0][volume[0] > th1] = 1.0
    volume[0][volume[0] <= th1] = 0.0

    volume[1][volume[1] > th2] = 1.0
    volume[1][volume[1] <= th2] = 0.0

    volume[2][volume[2] > th3] = 1.0
    volume[2][volume[2] <= th3] = 0.0
    return volume


def ray_cast_visibility(grid_points_flat, intrinsic_matrices, extrinsic_matrices):
    """
    Perform ray casting to determine visibility of each voxel from each camera.
    """
    visibility = np.ones((len(intrinsic_matrices), len(grid_points_flat)), dtype=bool)

    for cam_idx, (intrinsic, extrinsic) in enumerate(zip(intrinsic_matrices, extrinsic_matrices)):
        # Sort voxels by distance to the camera
        cam_pos = -extrinsic[:3, :3].T @ extrinsic[:3, 3]
        distances = np.linalg.norm(grid_points_flat - cam_pos, axis=1)
        sorted_indices = np.argsort(distances)
        sorted_voxels = grid_points_flat[sorted_indices]

        # Project sorted voxels and determine visibility
        projected_coords = project_points(sorted_voxels, intrinsic, extrinsic)
        depth_buffer = {}
        for idx, voxel in zip(sorted_indices, sorted_voxels):
            pixel = tuple(projected_coords[idx].astype(int))
            depth = np.linalg.norm(voxel - cam_pos)
            if pixel not in depth_buffer or depth < depth_buffer[pixel]:
                depth_buffer[pixel] = depth
            else:
                visibility[cam_idx, idx] = False

    return visibility


def compute_voxel_colors(grid_points_flat, images, intrinsic_matrices, extrinsic_matrices, nonvisible_weight=0.25):
    """
    Compute realistic colors for each voxel based on unoccluded images (NumPy version).
    """
    # Get visibility for all voxels and cameras
    visibility = ray_cast_visibility(grid_points_flat, intrinsic_matrices, extrinsic_matrices)  # Shape: (num_cameras, num_voxels)

    # Project all voxels into all camera views
    projected_coords = []
    for intrinsic, extrinsic in zip(intrinsic_matrices, extrinsic_matrices):
        projected_coords.append(project_points(grid_points_flat, intrinsic, extrinsic))  # Shape: (num_voxels, 2)
    projected_coords = np.stack(projected_coords)  # Shape: (num_cameras, num_voxels, 2)

    # Sample colors from all images
    sampled_colors = []
    for cam_idx, image in enumerate(images):
        h, w = image.shape[:2]
        coords = projected_coords[cam_idx]  # Shape: (num_voxels, 2)
        pixel_x = np.clip(np.round(coords[:, 0]).astype(int), 0, w - 1)
        pixel_y = np.clip(np.round(coords[:, 1]).astype(int), 0, h - 1)
        sampled_colors.append(image[pixel_y, pixel_x])  # Shape: (num_voxels, 3)
    sampled_colors = np.stack(sampled_colors)  # Shape: (num_cameras, num_voxels, 3)

    # Compute visibility weights
    weights = np.where(visibility, 1.0, nonvisible_weight)  # Shape: (num_cameras, num_voxels)
    weights = weights / weights.sum(axis=0, keepdims=True)  # Normalize weights per voxel

    # Compute weighted color for each voxel
    voxel_colors = (weights[..., None] * sampled_colors).sum(axis=0)  # Shape: (num_voxels, 3)

    return voxel_colors


def adjust_principal_points_to_seed(
    masks: np.ndarray,
    Ks: np.ndarray,
    extrinsics: np.ndarray,
) -> np.ndarray:
    """
    Adjust each camera's principal point so that the common 3D seed (triangulated
    from mask medoids) projects exactly through its original medoid.

    Parameters
    ----------
    masks : np.ndarray, shape (V, H, W)
        Binary silhouettes for V views.
    Ks : np.ndarray, shape (V, 3, 3)
        Original intrinsics matrices for each view.
    extrinsics : np.ndarray, shape (V, 4, 4)
        OpenCV‐style extrinsic matrices [R | t; 0 0 0 1] for each view.

    Returns
    -------
    new_Ks : np.ndarray, shape (V, 3, 3)
        Copies of Ks with updated (cx, cy) so that the triangulated seed projects
        through each medoid pixel.
    """
    V, H, W = masks.shape
    assert Ks.shape == (V, 3, 3)
    assert extrinsics.shape == (V, 4, 4)

    # 1) compute medoid (pixel nearest the centroid) in each view
    medoids = []
    for i in range(V):
        ys, xs = np.nonzero(masks[i])
        if xs.size == 0:
            raise ValueError(f"Mask {i} is empty")
        cy, cx = ys.mean(), xs.mean()
        # choose the mask‐pixel closest to (cx, cy)
        d2 = (ys - cy)**2 + (xs - cx)**2
        j = np.argmin(d2)
        medoids.append((xs[j], ys[j]))  # (u*, v*)
    medoids = np.array(medoids, dtype=np.float64)  # shape (V, 2)

    # 2) build projection matrices P_i = K_i @ [R|t]
    Ps = []
    for i in range(V):
        R = extrinsics[i][:3, :3]   # (3,3)
        t = extrinsics[i][:3, 3:]   # (3,1)
        Rt = np.concatenate([R, t], axis=1)  # (3,4)
        P = Ks[i] @ Rt                      # (3,4)
        Ps.append(P)
    Ps = np.stack(Ps, axis=0)  # (V, 3, 4)

    # 3) triangulate the 3D seed X via DLT
    A_rows = []
    for i in range(V):
        u, v = medoids[i]
        P = Ps[i]
        A_rows.append(u * P[2] - P[0])
        A_rows.append(v * P[2] - P[1])
    A = np.vstack(A_rows)     # (2V, 4)
    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1]
    X_h /= X_h[3]
    X = X_h[:3]               # Euclidean 3D seed

    # 4) adjust each K_i's principal point
    new_Ks = Ks.copy()
    for i in range(V):
        # camera‐space coordinates of X
        R = extrinsics[i][:3, :3]
        t = extrinsics[i][:3, 3]
        X_cam = R @ X + t       # (3,)

        fx = Ks[i, 0, 0]
        fy = Ks[i, 1, 1]
        u_star, v_star = medoids[i]

        cx_new = u_star - fx * (X_cam[0] / X_cam[2])
        cy_new = v_star - fy * (X_cam[1] / X_cam[2])

        new_Ks[i, 0, 2] = cx_new
        new_Ks[i, 1, 2] = cy_new

    return new_Ks, X
