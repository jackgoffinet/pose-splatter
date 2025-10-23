"""
Useful functions

"""
__date__ = "November 2024 - February 2025"

import h5py
import numpy as np
import os


def get_rough_center_3d(masks, Ps):
    """Triangulate the median points for each image to estimate a rough 3d center."""
    assert masks.ndim == 3
    assert len(masks) == len(Ps)
    medians_x1 = batch_weighted_median(masks.sum(axis=-2))
    medians_x2 = batch_weighted_median(masks.sum(axis=-1))
    medians = np.array([medians_x1, medians_x2]).T  # [camera,2]
    _, p_3d = triangulate_and_reproject(medians, Ps)
    return p_3d


def rotation_matrix_between(a, b):
    """Compute the rotation matrix that rotates vector a to vector b.

    Args:
        a: The vector to rotate.
        b: The vector to rotate to.
    Returns:
        The rotation matrix.
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)  # Axis of rotation.

    # Handle cases where `a` and `b` are parallel.
    eps = 1e-6
    if np.sum(np.abs(v)) < eps:
        x = np.array([1.0, 0, 0]) if abs(a[0]) < eps else np.array([0, 1.0, 0])
        v = np.cross(a, x)

    v = v / np.linalg.norm(v)
    skew_sym_mat = np.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )
    theta = np.arccos(np.clip(np.dot(a, b), -1, 1))

    # Rodrigues rotation formula. https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    return np.eye(3) + np.sin(theta) * skew_sym_mat + (1 - np.cos(theta)) * (skew_sym_mat @ skew_sym_mat)


def get_cam_params(
        cam_fn, ds=1,
        auto_orient=True,
        load_up_direction=True,
        up_fn="vertical_lines.npz",
        holdout_views=None,
    ):
    with h5py.File(cam_fn, "r") as f:
        d = dict(dict(f)["camera_parameters"])
        rotation = np.array(d["rotation"])
        translation = np.array(d["translation"])
        intrinsic = np.array(d["intrinsic"])

    extrinsic = np.stack([np.eye(4) for _ in range(len(intrinsic))], 0)
    extrinsic[:,:3,:3] = rotation
    extrinsic[:,:3,-1] = translation

    if auto_orient and load_up_direction:
        assert os.path.exists(up_fn)
        up = -np.load(up_fn)["up"]

    if ds != 1:
        intrinsic[...,0,0] /= ds
        intrinsic[...,1,1] /= ds
        intrinsic[...,0,2] /= ds
        intrinsic[...,1,2] /= ds

    if auto_orient:
        R = rotation
        if not load_up_direction:
            up = np.mean(extrinsic[:, :3, 1], axis=0)
            up = up / np.linalg.norm(up)
        R_2 = rotation_matrix_between(np.array([0, 0, 1.]), up)
        mean_translation = np.mean(np.transpose(R, (0,2,1)) @ translation[...,None], axis=0)
        rotation = R @ R_2.T[None]
        translation = (R @ mean_translation.reshape(1,3,1))[..., 0] + translation
        extrinsic = np.stack([np.eye(4) for _ in range(len(intrinsic))], 0)
        try:
            positions = np.linalg.solve(rotation, translation)
        except:
            positions = np.stack([np.linalg.solve(r, e) for r, e in zip(rotation, translation)], 0)
            # print(rotation.shape, translation.shape)
            # quit()
        scale_factor = 1.0 / np.max(np.linalg.norm(positions, axis=1))
        translation = scale_factor * translation
        extrinsic[:,:3,:3] = rotation
        extrinsic[:,:3,-1] = translation
        
    KR = intrinsic @ rotation
    Kt = intrinsic @ translation[..., None]
    Ps = np.concatenate([KR, Kt], axis=-1)

    if holdout_views is not None:
        obs = np.array([i for i in range(len(Ps)) if i not in holdout_views], dtype=int)
        intrinsic, extrinsic, Ps = intrinsic[obs], extrinsic[obs], Ps[obs]

    return intrinsic, extrinsic, Ps


def w2c_to_c2w(w2c):
    c2w = np.linalg.inv(w2c)
    c2w[:, 0:3, 1:3] *= -1
    c2w = c2w[:, np.array([1,0,2,3]), :]
    c2w[:,2] *= -1
    return c2w


def weighted_median(weights):
    values = np.arange(len(weights))
    c = np.cumsum(weights)
    idx = np.searchsorted(c, 0.5 * c[-1]).clip(0, len(values) - 1)
    return values[idx]


def batch_weighted_median(weights):
    return np.array([weighted_median(w) for w in weights])


def triangulatePoints(P1, P2, x1, x2):
    """
    Triangulate points from two views using camera projection matrices.

    This function uses a method similar to the one in OpenCV's triangulatePoints. It
    triangulates a point visible in two camera views, by creating and solving a linear
    system constructed from the camera projection matrices and the homogenized image points.

    Parameters
    ----------
    P1, P2 : ndarray of shape (3, 4)
        Projection matrices for the two views.
    x1, x2 : ndarray of shape (n_points, 3)
        Homogenized coordinates of the points in each view.

    Returns
    -------
    ndarray of shape (n_points, 4)
        Triangulated points in homogenous coordinates.
    """
    if not len(x2) == len(x1):
        raise ValueError("Number of points don't match.")

    X = []
    for x_1, x_2 in zip(x1, x2):
        M = np.zeros((6, 6))
        M[:3, :4] = P1
        M[3:, :4] = P2
        M[:3, 4] = -x_1
        M[3:, 5] = -x_2

        _, _, V = np.linalg.svd(M)
        X.append(V[-1, :4])

    return np.array(X) / X[-1][3]


def triangulate_and_reproject(points, Ps):
    """
    points: [C,2]
    Ps: [C,3,4]
    """
    idx = np.array([i for i in range(len(points)) if points[i] is not None], dtype=int)
    if len(idx) < 2:
        print("Need more points to project!")
        return points, np.nan * np.zeros(3)

    all_projs = []
    all_positions = []
    arr_points = np.array([[points[i][0], points[i][1]] for i in idx]).reshape(-1,2)
    for i in range(len(idx)):
        cam_input_i = Ps[idx[i]]
        arr_points_i = np.concatenate([arr_points[i:i+1,:], np.ones((1,1))], axis=1)
        for j in range(i+1,len(idx)):
            cam_input_j = Ps[idx[j]]
            arr_points_j = np.concatenate([arr_points[j:j+1,:], np.ones((1,1))], axis=1)
            pos_3d = triangulatePoints(
                cam_input_i,
                cam_input_j,
                arr_points_i,
                arr_points_j,
            ).flatten()
            pos_3d /= pos_3d[-1] # homogeneous coordinates, [4]
            all_positions.append(pos_3d[:3])
            new_points = np.array([P @ pos_3d for P in Ps])
            all_projs.append(new_points)
    all_projs = np.array(all_projs)
    all_positions = np.array(all_positions)
    all_projs = all_projs[..., :2] / all_projs[..., 2:3]
    return np.median(all_projs, axis=0), np.median(all_positions, axis=0)

