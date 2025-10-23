"""
Shape carve and plot the voxels to tune shape carving parameters.

"""
__date__ = "January - August 2025"

import cv2
import numpy as np
import sys

from src.config_utils import Config
from src.plots import plot_color_voxel_grid, plot_voxel_grid
from src.shape_carving import create_3d_grid, get_volume, shift_and_rotate_grid_points, compute_voxel_colors
from src.utils import get_cam_params, get_rough_center_3d

USAGE = "Usage:\n$ python plot_voxels.py <config.json> <frame_number>"
WHITE = np.ones(3)
ADAPTIVE = False


if __name__ == '__main__':
    assert len(sys.argv) in [2, 3], USAGE
    config = Config(sys.argv[1])
    
    if len(sys.argv) == 3:
        frame_idx = int(sys.argv[2])
    else:
        # Choose a random frame.
        cap = cv2.VideoCapture(config.mask_video_fns[0])
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        frame_idx = np.random.randint(total_frames)
        print("Random frame number:", frame_idx)

    # Get the videos.
    mask_caps = [cv2.VideoCapture(mask_video_fn) for mask_video_fn in config.mask_video_fns]
    video_caps = [cv2.VideoCapture(video_fn) for video_fn in config.video_fns]
    for cap in mask_caps + video_caps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    grid = create_3d_grid(config.ell, config.grid_size)
    ds = config.image_downsample

    # # Get volumes and calculate center.
    # intrinsic, extrinsic, Ps = get_cam_params(
    #     config.camera_fn,
    #     ds=ds,
    #     up_fn=config.vertical_lines_fn,
    #     auto_orient=True,
    #     load_up_direction=True,
    # )
    
    # Get volumes and calculate center.
    intrinsic, extrinsic, Ps = get_cam_params(
        config.camera_fn,
        ds=ds,
        up_fn=config.vertical_lines_fn,
        auto_orient=True,
        load_up_direction=False,
    )
    
    cam_positions = []
    for E in extrinsic:
        cam_positions.append(np.linalg.solve(E[:3,:3], E[:3,-1]))
    diffs = []
    for i in range(len(cam_positions)):
        for j in range(i+1, len(cam_positions)):
            diffs.append(np.linalg.norm(cam_positions[i] - cam_positions[j]))
    print("mean camera distance:", np.mean(diffs))
    C = len(Ps)

    # Get the masks from the videos.
    error_flag = False
    masks = []
    frames = []
    for cap_num, (mask_cap, video_cap) in enumerate(zip(mask_caps, video_caps)):
        ret, frame = mask_cap.read()
        if not ret:
            print(f"Error reading mask frame {frame_idx}, video {cap_num}")
            error_flag = True
            break
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            masks.append(frame)

        ret, frame = video_cap.read()
        if not ret:
            print(f"Error reading video frame {frame_idx}, video {cap_num}")
            error_flag = True
            break
        else:
            frames.append(frame[...,::-1])

    for cap in mask_caps + video_caps:
        cap.release()

    if error_flag:
        print("Error!")
        quit()

    masks = np.array(masks).astype(np.float32) / 255.0 # [cameras,H,W]
    frames = np.array(frames).astype(np.float32) / 255.0 # [cameras,H,W,3]

    if ds != 1:
        masks = masks[:,::ds][:,:,::ds]
        frames = frames[:,::ds][:,:,::ds]

    masks = np.where(masks > 0.5, 1.0, 0.0)
    print("mean mask:", np.mean(masks))
    frames[masks == 0] = WHITE

    # Estimate center roughly.
    p_3d = get_rough_center_3d(masks, Ps)
    print("rough center:", p_3d)

    # Shift the grid.
    temp_grid_points = shift_and_rotate_grid_points(grid[:], p_3d, 0)

    # Shape carve.
    if config.adaptive_grid:
        import cv2

        mask_volume, intrinsic, p_3d = get_volume(
            masks,
            intrinsic,
            extrinsic,
            temp_grid_points,
            adaptive=config.adaptive_grid,
        ) # [1, n1, n2, n3]

        temp_grid_points = shift_and_rotate_grid_points(grid[:], p_3d, 0)
        mask_volume, intrinsic, _ = get_volume(
            masks,
            intrinsic,
            extrinsic,
            temp_grid_points,
            adaptive=config.adaptive_grid,
        ) # [1, n1, n2, n3]
    else:
        mask_volume = get_volume(
            masks,
            intrinsic,
            extrinsic,
            temp_grid_points,
            adaptive=config.adaptive_grid,
        ) # [1, n1, n2, n3]
    
    print("volume min/max:", np.min(mask_volume), np.max(mask_volume))
    binary_volume = (mask_volume >= 1).flatten()
    print("number of voxels:", binary_volume.sum())
    means = temp_grid_points.reshape(-1,3)[binary_volume] # [n,3]
    colors = compute_voxel_colors(means, frames, intrinsic, extrinsic)
    print("mean color:", np.mean(colors, axis=0))
    volume = config.volume_fill_color * np.ones((4, config.grid_size**3), dtype=np.float16)
    volume[0] = binary_volume
    volume[1:,binary_volume] = colors.T
    volume = volume.reshape(4, config.grid_size, config.grid_size, config.grid_size)

    # Plot.
    # plot_voxel_grid(volume)
    plot_color_voxel_grid(volume)


    
