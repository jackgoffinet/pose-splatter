"""
Render a single image and save as a PNG file.

"""
__date__ = "January - March 2025"

import argparse
import numpy as np
import os
from PIL import Image
import torch

from src.config_utils import Config
from src.data import FrameDataset
from src.model import PoseSplatter
from src.utils import get_cam_params

os.environ['TORCH_CUDA_ARCH_LIST'] = "8.6" # NOTE: move this!


def render_image(config, args, out_fn, model_fn, ablation):
    """Render a full-size image."""
    device = "cuda"

    intrinsic, extrinsic, _ = get_cam_params(
        config.camera_fn,
        ds=config.image_downsample,
        up_fn=config.vertical_lines_fn,
        auto_orient=True,
        load_up_direction=not config.adaptive_camera,
    )
    C = len(intrinsic)
    print("C", C)

    K, E, _ = get_cam_params(
        config.camera_fn,
        ds=1,
        up_fn=config.vertical_lines_fn,
        auto_orient=True,
        load_up_direction=not config.adaptive_camera,
    )

    K = torch.tensor(K).to(device, torch.float32)
    E = torch.tensor(E).to(device, torch.float32)

    img_fn = os.path.join(config.image_directory, "images.h5")
    volume_fn = os.path.join(config.volume_directory, "volumes.h5")
    dset = FrameDataset(img_fn, volume_fn, config.center_rotation_fn, C, holdout_views=config.holdout_views, split="all")

    w = config.image_width // config.image_downsample
    h = config.image_height // config.image_downsample

    model = PoseSplatter(
        intrinsics=intrinsic,
        extrinsics=extrinsic,
        W=w,
        H=h,
        ell=config.ell,
        grid_size=config.grid_size,
        volume_idx=config.volume_idx,
        ablation=ablation,
        volume_fill_color=config.volume_fill_color,
        holdout_views=config.holdout_views,
        adaptive_camera=config.adaptive_camera
    )
    model.to(device)
    if model_fn is None:
        if args.ablation:
            model_fn = config.model_fn[:-3] + "_ablation.pt"
        else:
            model_fn = config.model_fn
    model.load_state_dict(torch.load(model_fn)["model_state_dict"])
    model.eval()

    dset_idx = C * args.frame_num + args.view_num
    center_offset = torch.tensor([args.delta_x, args.delta_y, args.delta_z])
    print("center_offset", center_offset)
    angle_offset = args.angle_offset

    # Get the item from the dataset.
    mask, img, p_3d, angle, view_idx = dset.__getitem__(dset_idx)
    p_3d = p_3d + center_offset
    
    # Forward pass
    with torch.no_grad():
        # rgb, _ = model(
        #     mask[None].to(device),
        #     img[None].to(device),
        #     p_3d[None].to(device),
        #     angle,
        #     view_num=view_idx,
        # ) # [1, 3, H, W], [1, H, W, 1]
        # rgb = rgb[0].detach().cpu().numpy()

        # # Make the volume.
        # volume = model.carver(mask[:,None].to(device), img.to(device), p_3d.to(device), angle)

        # Make the volume.
        if config.adaptive_camera:
            volume, temp_K = model.carver(mask[:,None].to(device), img.to(device), p_3d.to(device), angle, adaptive=config.adaptive_camera)
        else:
            volume = model.carver(mask[:,None].to(device), img.to(device), p_3d.to(device), angle, adaptive=config.adaptive_camera)

        # Run the volume through the U-Nets.
        volume = model.process_volume(volume[None])

        # Get Gaussian parameters.
        means, quats, scales, opacities, colors = model.get_gaussian_params_from_volume(volume)

        # if False:
        #     # Output point cloud.
        #     import numpy as np
        #     import pymeshlab

        #     pos = means.detach().cpu().numpy()
        #     pos -= np.mean(pos, axis=0, keepdims=True)
        #     rgb = colors.detach().cpu().numpy()
        #     opacity = opacities.detach().cpu().numpy()
        #     colors = np.concatenate([rgb, opacity[:,None]], axis=1)

        #     # 2) No faces → empty (0×3) array
        #     faces = np.empty((0, 3), dtype=np.uint32)

        #     # 3) Build a Mesh with vertex colors
        #     #    v_color_matrix expects floats in [0,1] for each RGBA channel :contentReference[oaicite:0]{index=0}
        #     mesh = pymeshlab.Mesh(pos, faces, v_color_matrix=colors)

        #     # 4) Add to a MeshSet and save as PLY
        #     ms = pymeshlab.MeshSet()
        #     ms.add_mesh(mesh, "colored_point_cloud")
        #     ms.save_current_mesh(f"colored_pointcloud_{args.frame_num}.ply")
        #     quit()

        # Rotate and shift the Gaussian means.
        c, s = np.cos(angle), np.sin(angle)
        rot_mat = torch.tensor([[c,-s,0], [s,c,0], [0,0,1]]).to(volume.device, torch.float32)

        # TEMP
        # perm = torch.tensor([1,0,2]).to(device, torch.long)
        # rot_mat = rot_mat[perm][:,perm]
        # rot_mat = torch.tensor([[c,0,-s], [0,1,0], [s,0,c]]).to(volume.device, torch.float32)
        # rot_mat = torch.tensor([[0,c,-s], [0,s,c], [1,0,0]]).to(volume.device, torch.float32)
        means = means @ rot_mat.T + p_3d.to(device, torch.float32) # [n,3]

        center = torch.mean(means, dim=0, keepdim=True)
        c, s = np.cos(angle_offset), np.sin(angle_offset)
        # rot_mat = torch.tensor([[c,0,-s], [0,1,0], [s,0,c]]).to(volume.device, torch.float32)
        rot_mat = torch.tensor([[c,-s,0], [s,c,0], [0,0,1]]).to(volume.device, torch.float32)

        means = (means - center) @ rot_mat.T + center

        if config.adaptive_camera:
            out_K = temp_K[view_idx].view(-1,3,3)
        else:
            out_K = K[view_idx].view(-1,3,3)

        # Splat.
        rgb, _ = model.splat(
            means,
            quats,
            scales,
            opacities,
            colors,
            E[view_idx:view_idx+1],
            out_K,
            config.image_width,
            config.image_height,
        ) # [b,H,W,3]
        rgb = rgb[0].detach().cpu().numpy()
    
    rgb = (255 * rgb.clip(0, 1)).astype(np.uint8)
    image = Image.fromarray(rgb)
    image.save(out_fn)
    print("Saved:", out_fn)



if __name__ == '__main__':
    # Create argument parser
    parser = argparse.ArgumentParser(description="Render an image")
    
    # Add positional arguments
    parser.add_argument("config", type=str, help="Path to the configuration file (e.g., config.json)")
    parser.add_argument("frame_num", type=int, help="Frame number (integer)")
    parser.add_argument("view_num", type=int, help="View number (integer)")
    
    # Add optional arguments with default values
    parser.add_argument("--angle_offset", type=float, default=0.0, help="Angle offset (float, default: 0.0)")
    parser.add_argument("--delta_x", type=float, default=0.0, help="Delta X (float, default: 0.0)")
    parser.add_argument("--delta_y", type=float, default=0.0, help="Delta Y (float, default: 0.0)")
    parser.add_argument("--delta_z", type=float, default=0.0, help="Delta Z (float, default: 0.0)")
    parser.add_argument("--model_fn", type=str, default=None, help="Model filename")
    parser.add_argument("--out_fn", type=str, default=None, help="Image filename")
    parser.add_argument("--ablation", action="store_true", help="Flag to use the ablation model")
    
    # Parse arguments
    args = parser.parse_args()
    config = Config(args.config)

    if args.out_fn is None:
        out_fn = f"render_{args.frame_num}_{args.view_num}_{args.angle_offset:.1f}_"
        out_fn += f"{args.delta_x:.1f}_{args.delta_y:.1f}_{args.delta_z:.1f}"
        if args.ablation:
            out_fn += "_ablation.png"
        else:
            out_fn += ".png"
        out_fn = os.path.join(config.render_directory, out_fn)
    else:
        out_fn = args.out_fn

    # Make the directory, if necessary,
    if not os.path.exists(config.render_directory):
        os.makedirs(config.render_directory)

    # Render.
    render_image(config, args, out_fn, args.model_fn, args.ablation)

###