"""
Calculate the center and rotation from volumes.

"""
__date__ = "December 2024 - January 2025"


import cv2
from joblib import Parallel, delayed
import numpy as np
import os
import sys

from src.config_utils import Config
from src.shape_carving import (
    create_3d_grid,
    get_volume,
    shift_and_rotate_grid_points,
    adjust_principal_points_to_seed,
)
from src.utils import get_cam_params, get_rough_center_3d
from src.tracking import track_principal_axes

USAGE = "Usage:\n$ python calculate_center_rotation.py <config.json>"



def mean_and_covariance_3d(volume, coordinates):
    """
    Calculate the mean and covariance of a 3D volume in space.
    
    Parameters:
    - volume: np.ndarray of shape (n, n, n), the nonnegative volume values.
    - coordinates: np.ndarray of shape (n, n, n, 3), the spatial coordinates corresponding to the volume.
    
    Returns:
    - mean: np.ndarray of shape (3,), the mean position in 3D space weighted by the volume.
    - covariance: np.ndarray of shape (3, 3), the covariance matrix in 3D space weighted by the volume.
    """
    # Normalize the volume to obtain weights
    total_weight = np.sum(volume)
    if total_weight == 0:
        raise ValueError("Volume is all zeros, cannot compute mean and covariance.")
    
    weights = volume / total_weight
    
    # Compute the mean (weighted centroid)
    mean = np.sum(coordinates * weights[..., None], axis=(0, 1, 2))
    
    # Compute the covariance matrix
    centered_coords = coordinates - mean
    covariance = np.einsum('ijkl,ijk,ijkm->lm', centered_coords, weights, centered_coords)
    
    return mean, covariance


def process_chunk_center_angle(config, chunk_num, frame_list):
    mask_caps= []
    holdout_views = config["holdout_views"]
    for i, mask_video_fn in enumerate(config["mask_video_fns"]):
        if i not in holdout_views:
            mask_caps.append(cv2.VideoCapture(mask_video_fn))

    for cap in mask_caps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_list[0])

    grid = create_3d_grid(config["ell_tracking"], config["grid_size"])
    ds = config["image_downsample"]

    # Get volumes and calculate center.
    intrinsic, extrinsic, Ps = get_cam_params(
        config["camera_fn"],
        ds=ds,
        auto_orient=True,
        load_up_direction=not config["adaptive_camera"],
    )

    obs = np.array([i for i in range(len(Ps)) if i not in holdout_views], dtype=int)
    intrinsic, extrinsic, Ps = intrinsic[obs], extrinsic[obs], Ps[obs]
    C = len(Ps)

    res = []
    error_flag = False

    num_samples = len(frame_list)
    if chunk_num == 0:
        print("num_samples:", num_samples)

    for volume_idx, frame_idx in enumerate(frame_list):
        if chunk_num == 0:
            print(volume_idx)
        # Get the masks from the videos.
        masks = []
        for cap_num, cap in enumerate(mask_caps):
            ret, frame = cap.read()
            if not ret:
                print(f"Error reading frame {frame_idx}, video {cap_num}")
                error_flag = True
                break
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                masks.append(frame)

            for _ in range(config["frame_jump"] - 1):
                cap.read()
        if error_flag:
            print("Error!", frame_idx)
            break

        masks = np.array(masks).astype(np.float32) / 255.0 # [cameras,H,W]

        if ds != 1:
            masks = masks[:,::ds][:,:,::ds]

        masks = np.where(masks > 0.5, 1.0, 0.0)

        # Update camera parameters.
        if config["adaptive_camera"]:
            temp_intrinsic, p_3d = adjust_principal_points_to_seed(masks, intrinsic, extrinsic)
        else:
            # Estimate center roughly.
            p_3d = get_rough_center_3d(masks, Ps)
            temp_intrinsic = intrinsic

        # Shift the grid.
        temp_grid = shift_and_rotate_grid_points(grid[:], p_3d, 0)

        # Make a shape carving volume.
        # [1,n,n,n]
        volume = get_volume(masks[...,None], temp_intrinsic, extrinsic, temp_grid)

        # Shape carving to volume.
        volume = volume[0]
        thresh = (C - 1) / C
        volume = np.where(volume >= thresh, 1.0, 0.0)

        # # Plot to check the carving parameters.
        # from src.plots import plot_gsplat_color
        # plot_gsplat_color(np.stack([volume, 0.5*volume, 0.5*volume, 0.5*volume], 0), temp_grid, intrinsic, extrinsic, 512, 1536//ds)
        # quit()

        # Approximate center and rotation.
        center, cov = mean_and_covariance_3d(volume, temp_grid)
        res.append((center, cov))

    
    for cap in mask_caps:
        cap.release()
    
    return res



if __name__ == '__main__':
    assert len(sys.argv) == 2, USAGE
    config = Config(sys.argv[1])
    N_JOBS = len(os.sched_getaffinity(0)) # available CPUs

    cap = cv2.VideoCapture(config.video_fns[0])
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("total_frames:", total_frames)
    cap.release()

    # Figure out frames for each job to process.
    frame_lists = np.array_split(np.arange(0, total_frames, config.frame_jump), N_JOBS)

    # The config needs to be serializable to parallelize.
    s_config = config.to_serializable()

    # Use joblib to parallelize processing
    res = Parallel(n_jobs=N_JOBS)(
        delayed(process_chunk_center_angle)(s_config, i, frame_list) for i, frame_list in enumerate(frame_lists)
    )
    
    # Flatten the list of results to get means and covariances.
    processed_frames = [frame for chunk in res for frame in chunk]
    centers = [i[0] for i in processed_frames]
    covs = [i[1] for i in processed_frames]
    centers = np.array(centers)
    covs = np.array(covs)

    # Track the principal axis through time to get the x/y angles.
    axes = track_principal_axes(centers, covs)
    angles = np.angle(axes[:,0] + 1j * axes[:,1])

    # Save the results.
    np.savez(config.center_rotation_fn, centers=centers, angles=angles, covs=covs)


###