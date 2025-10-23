"""
Calculate the volume crop indices.

"""
__date__ = "December 2024 - January 2025"


import argparse
import cv2
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
import os

from src.config_utils import Config
from src.shape_carving import create_3d_grid, get_volume, shift_and_rotate_grid_points, adjust_principal_points_to_seed
from src.utils import get_cam_params

MAX_NUM_FRAMES = 5000


def find_div_n_subarray(arr, thresh=1, n=16):
    assert arr.ndim == 1
    assert len(arr) % n == 0
    # Find indices of all entries >= thresh
    ones = np.where(arr >= thresh)[0]
    
    # If no ones, no subarray needed
    if len(ones) == 0:
        return 0, 0
    
    min_idx = ones[0]
    max_idx = ones[-1] + 1
    rem = (max_idx - min_idx) % n
    if rem != 0:
        rem_mod_2 = (n - rem) % 2
        half_rem = (n - rem) // 2
        assert rem_mod_2 + 2 * half_rem == n - rem
        min_idx -= half_rem + rem_mod_2
        max_idx += half_rem
        assert (max_idx - min_idx) % n == 0, f"{(min_idx, max_idx)}"
        if min_idx < 0:
            diff = -min_idx
            assert diff > 0
            min_idx += diff
            max_idx += diff
        elif max_idx > len(arr):
            diff = max_idx - len(arr)
            assert diff > 0
            min_idx -= diff
            max_idx -= diff
    assert (max_idx - min_idx) % n == 0
    assert min_idx >= 0
    assert max_idx <= len(arr)
    return min_idx, max_idx


def process_chunk_volume_sum(config, chunk_num, frame_list, centers, angles):
    mask_caps= []
    holdout_views = config["holdout_views"]
    for i, mask_video_fn in enumerate(config["mask_video_fns"]):
        if i not in holdout_views:
            mask_caps.append(cv2.VideoCapture(mask_video_fn))

    for cap in mask_caps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_list[0])

    grid = create_3d_grid(config["ell"], config["grid_size"])
    ds = config["image_downsample"]

    # Get volumes and calculate center.
    intrinsic, extrinsic, Ps = get_cam_params(
        config["camera_fn"],
        ds=ds,
        up_fn=config["vertical_lines_fn"],
        auto_orient=True,
        load_up_direction=not config["adaptive_camera"],
        holdout_views=config["holdout_views"],
    )
    C = len(Ps)

    error_flag = False
    num_samples = len(frame_list)
    if chunk_num == 0:
        print("num_samples:", num_samples)
    volume_sum = np.zeros((config["grid_size"], config["grid_size"], config["grid_size"]), dtype=int)

    for frame_num, frame_idx in enumerate(frame_list):
        if chunk_num == 0:
            print(frame_num)
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
            break

        masks = np.array(masks).astype(np.float32) / 255.0 # [cameras,H,W]

        if ds != 1:
            masks = masks[:,::ds][:,:,::ds]

        masks = np.where(masks > 0.5, 1.0, 0.0)

        i = frame_idx // config["frame_jump"]
        temp_grid = shift_and_rotate_grid_points(grid[:], centers[i], angles[i], angle_offset=0.0)

        if config["adaptive_camera"]:
            temp_intrinsic, _ = adjust_principal_points_to_seed(masks, intrinsic, extrinsic)
        else:
            temp_intrinsic = intrinsic

        # Make volume.
        # [1,n,n,n]
        volume = get_volume(masks[..., None], temp_intrinsic, extrinsic, temp_grid)
        
        # Shape carving to volume.
        volume = volume[0]
        if config["adaptive_camera"]:
            mask = volume >= 1.0
        else:
            mask = (volume >= (C - 1) / C)
        volume_sum += mask.astype(int)

    
    for cap in mask_caps:
        cap.release()
    
    return volume_sum



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate the volume crop indices")
    parser.add_argument("config", type=str, help="Path to the config JSON file")
    parser.add_argument("--force", action="store_true", help="Recalculate the volume sum")
    
    args = parser.parse_args()

    config = Config(args.config)
    N_JOBS = os.cpu_count() // 2

    assert os.path.exists(config.center_rotation_fn)
    d = np.load(config.center_rotation_fn)
    centers = d["centers"]
    angles = d["angles"]

    cap = cv2.VideoCapture(config.mask_video_fns[0])
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 3
    cap.release()


    total_frames = min(total_frames, MAX_NUM_FRAMES * config.frame_jump)
    print("total_frames:", total_frames)


    if args.force or not os.path.exists(config.volume_sum_fn):
        if not os.path.exists(config.volume_sum_fn):
            print("Volume sum doesn't exist. Calculating...")
        else:
            print("Recalculating the volume sum...")

        # Figure out frames for each job to process.
        frame_lists = np.array_split(np.arange(0, total_frames, config.frame_jump), N_JOBS)
    
        # The config needs to be serializable to parallelize.
        s_config = config.to_serializable()

        # Use joblib for parallelize processing
        res = Parallel(n_jobs=N_JOBS)(
            delayed(process_chunk_volume_sum)(s_config, i, frame_list, centers, angles) for i, frame_list in enumerate(frame_lists)
        )
        
        # Save the volume sum.
        volume_sum = sum(res)
        np.save(config.volume_sum_fn, volume_sum)
    else:
        print("Volume sum exists. Loading...")
        volume_sum = np.load(config.volume_sum_fn)

    # Print out some summaries.
    print(f"Total volumes: {total_frames // config.frame_jump}\n")
    for thresh in [1,3,10,30,100,300, 400,500, 1000]:
        volume_idx = []
        for i in range(3):
            i2, i3 = (i+1) % 3, (i+2) % 3
            idx1, idx2 = find_div_n_subarray(
                volume_sum.sum(axis=(i2, i3)),
                n=16,
                thresh=thresh,
            )
            volume_idx.append((idx1, idx2))

        print("Threshold:", thresh)
        print("volume_idx:", volume_idx)
        print("n1, n2, n3:", [j-i for i, j in volume_idx])
        print()

    # Make a plot.
    for i in range(3):
        i2, i3 = (i+1) % 3, (i+2) % 3
        temp_sum = volume_sum.sum(axis=(i2, i3))
        plt.semilogy(temp_sum, label=f"axis {i}")
    plt.ylim(1, None)
    plt.legend(loc="best")
    plt.savefig("temp.pdf")






###