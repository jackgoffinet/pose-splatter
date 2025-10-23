"""
Evaluate the model by rendering images and calculating metrics.

"""
__date__ = "January - March 2025"

import argparse
import h5py
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure

from src.config_utils import Config
from src.data import FrameDataset
from src.model import PoseSplatter
from src.utils import get_cam_params

os.environ['TORCH_CUDA_ARCH_LIST'] = "8.6" # NOTE: move this!



def render_images(render_fn, config, ablation=False, load_fn=None):
    """Render the images using the trained model."""
    intrinsic, extrinsic, Ps = get_cam_params(
        config.camera_fn,
        ds=config.image_downsample,
        up_fn=config.vertical_lines_fn,
        auto_orient=True,
        load_up_direction=not config.adaptive_camera,
    )
    C = len(Ps)

    device = "cuda"

    img_fn = os.path.join(config.image_directory, "images.h5")
    volume_fn = os.path.join(config.volume_directory, "volumes.h5")

    num_workers = len(os.sched_getaffinity(0))
    dset = FrameDataset(img_fn, volume_fn, config.center_rotation_fn, C, split="all_volumes")
    total_num_frames = len(dset)
    print("total_num_frames:", total_num_frames)

    dset = FrameDataset(img_fn, volume_fn, config.center_rotation_fn, C, split="test")
    num_frames = len(dset)
    print("frames to write:", num_frames)
    offset = total_num_frames - num_frames
    loader = DataLoader(dset, batch_size=1, shuffle=False, num_workers=num_workers)

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
        adaptive_camera=config.adaptive_camera,
    )
    model.to(device)
    if load_fn is None:
        if ablation:
            load_fn = config.model_fn[:-3] + "_ablation.pt"
        else:
            load_fn = config.model_fn
    model.load_state_dict(torch.load(load_fn)["model_state_dict"])
    model.eval()

    # Batch size for writing to HDF5
    write_batch_frames = 50

    view_idx = torch.tensor(np.arange(C)).to(device, torch.long)

    with h5py.File(render_fn, "w") as hdf:
        # Create a dataset with gzip compression
        images_dataset = hdf.create_dataset(
            "images", 
            (total_num_frames, C, h, w, 4),
            dtype='uint8', 
            compression="gzip",
            compression_opts=config.image_compression_level # (0-9 for gzip)
        )

        # Buffer to hold a batch of images
        buffer = []
        local_frame_idx = 0

        pbar = tqdm(loader, total=len(dset))

        for mask, img, p_3d, angle, _ in pbar:
            assert mask.shape[0] == 1, "batch size must be 1"
            
            rgb, alpha = model(
                mask[:,torch.tensor(model.observed_views)].to(device),
                img[:,torch.tensor(model.observed_views)].to(device),
                p_3d.to(device),
                float(angle[0]),
                view_num=view_idx,
            ) # [1,H,W,3], [1,H,W,1]
            rgba = torch.cat([rgb, alpha], -1)  # [C,h,w,4]

            # Normalize and convert to uint8
            rgba = (255 * rgba.detach().cpu().numpy().clip(0, 1)).astype(np.uint8)

            # # TEMP: plot to make sure things are working
            # import matplotlib.pyplot as plt
            # fig, axarr = plt.subplots(ncols=C, nrows=2)
            # img = (255 * img[0].detach().cpu().numpy().clip(0,1)).astype(np.uint8)
            # img = np.transpose(img, (0,2,3,1))
            # print("img", img.shape, rgba.shape)
            
            # for i in range(C):
            #     axarr[0,i].imshow(img[i,:,:,:3])
            # for i in range(C):
            #     axarr[1,i].imshow(rgba[i,:,:,-1])
            # plt.savefig("temp.pdf")
            # quit()
            
            # Add to buffer
            buffer.append(rgba)

            # Write to dataset if buffer is full
            if len(buffer) >= write_batch_frames:
                buffer = np.array(buffer) # .reshape(-1, C, h, w, 4)
                # images_dataset[frame_idx-write_batch_frames+1 : frame_idx+1] = buffer
                i1 = offset + local_frame_idx
                i2 = offset + local_frame_idx + len(buffer)
                images_dataset[i1:i2] = buffer
                local_frame_idx += len(buffer)
                buffer = []  # Clear buffer
                

        # Write any remaining data in the buffer
        if buffer:
            i1 = offset + local_frame_idx
            i2 = offset + local_frame_idx + len(buffer)
            images_dataset[i1:i2] = buffer
            local_frame_idx += len(buffer)
            buffer = []  # Clear buffer
            # buffer = np.array(buffer) # .reshape(-1, C, h, w, 4)
            # images_dataset[-len(buffer):] = buffer


def calculate_image_metrics(pred_fn, gt_fn, metrics_fn, batch_size=32, split="test"):
    """Calculate the image metrics."""
    assert split in ["train", "valid", "test"]
    pred_file = h5py.File(pred_fn, "r")
    pred_images = pred_file["images"]
    
    gt_file = h5py.File(gt_fn, "r")
    gt_images = gt_file["images"]

    ssim_obj = StructuralSimilarityIndexMeasure(data_range=1.0, reduction=None)

    assert pred_images.shape[:-1] == gt_images.shape[:-1], \
        f"{pred_images.shape}[:-1] != {gt_images.shape}[:-1]"
    C = pred_images.shape[1]

    # Figure out split indices.
    a1, a2 = 0, len(gt_images) // 3
    a3, a4 = 2 * a2, len(gt_images)
    if split == "train":
        i1, i2 = a1, a2
    elif split == "valid":
        i1, i2 = a2, a3
    else: # split == "test"
        i1, i2 = a3, a4

    # Iterate over the frames.
    metrics = dict(
        l1=np.zeros(C),
        iou=np.zeros(C),
        soft_iou=np.zeros(C),
        ssim=np.zeros(C),
        psnr=np.zeros(C),
    )
    sorted_keys = sorted(list(metrics.keys()))

    for start_idx in tqdm(range(i1, i2, batch_size)):
        end_idx = min(start_idx + batch_size, i2)
        
        gt_img = torch.tensor(gt_images[start_idx:end_idx], dtype=torch.float32) / 255.0
        mask = torch.where(gt_img[..., 0] == 1.0, 0.0, 1.0) # [b,C,h,w]

        temp = torch.tensor(pred_images[start_idx:end_idx], dtype=torch.float32) / 255.0
        pred_alpha = temp[..., -1]
        pred_img = temp[..., :3] # [b,C,h,w,3]

        gt_img = torch.permute(gt_img, (0,1,4,2,3)) # [b,C,3,h,w]
        pred_img = torch.permute(pred_img, (0,1,4,2,3)) # [b,C,3,h,w]
        h, w = pred_img.shape[-2:]

        # Calculate all the metrics.
        l1 = torch.abs(gt_img - pred_img).sum(dim=(-3,-2,-1)) / mask.sum(dim=(-2,-1))
        l1 = l1.sum(dim=0)
        metrics["l1"] += l1.numpy()

        iou = get_iou(torch.where(pred_alpha > 0.5, 1.0, 0.0), mask).sum(dim=0)
        metrics["iou"] += iou.numpy()

        soft_iou = get_iou(pred_alpha, mask).sum(dim=0)
        metrics["soft_iou"] += soft_iou.numpy()

        psnr = get_psnr(pred_img, gt_img).sum(dim=0)
        metrics["psnr"] += psnr.numpy()

        ssim = ssim_obj(pred_img.view(-1,3,h,w), gt_img.view(-1,3,h,w))
        ssim = ssim.view(end_idx - start_idx, -1).sum(dim=0)
        metrics["ssim"] += ssim.numpy()
        
    # Average.
    for key in sorted_keys:
        metrics[key] = metrics[key] / (i2 - i1)
    
    data = np.column_stack([metrics[key] for key in sorted_keys])
    header = '\t'.join(sorted_keys)  # Create a single string of column headers
    np.savetxt(metrics_fn, data, delimiter=',', header=header, fmt='%.6f')

    print(f"Data saved to {metrics_fn}")



def get_iou(pred_mask, gt_mask, eps=1e-6):
    assert pred_mask.ndim == 4
    assert pred_mask.shape == gt_mask.shape
    intersection = (pred_mask * gt_mask).sum(dim=(-2, -1))
    union = (pred_mask + gt_mask - pred_mask * gt_mask).sum(dim=(-2, -1))
    iou = (intersection + eps) / (union + eps)
    return iou


def get_psnr(pred_img, gt_img, data_range=1.0):
    mse = ((pred_img - gt_img)**2).mean(dim=(-3,-2,-1))
    psnr = 10 * torch.log10(data_range**2 / mse)
    return psnr



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train script for the model")
    parser.add_argument("config", type=str, help="Path to the config JSON file")
    parser.add_argument("--force", action="store_true", help="Flag to recompute the renderings")
    parser.add_argument("--ablation", action="store_true", help="Flag to use ablation model")
    parser.add_argument("--render_fn", type=str, default=None, help="Render filename (absolute)")
    parser.add_argument("--metrics_fn", type=str, default=None, help="Metrics filename (absolute)")
    parser.add_argument("--model_fn", type=str, default=None, help="Model filename")


    args = parser.parse_args()
    
    config = Config(args.config)
    force = args.force  # This will be True if --force is provided, False otherwise

    print(f"Config file: {args.config}")
    print(f"Force flag: {force}")

    gt_fn = os.path.join(config.image_directory, "images.h5")
    render_fn = args.render_fn
    if render_fn is None:
        if args.ablation:
            render_fn = os.path.join(config.image_directory, "rendered_images_ablation.h5")
        else:
            render_fn = os.path.join(config.image_directory, "rendered_images.h5")
    rendered_images_exist = os.path.exists(render_fn)

    # Render the images.
    if force or not rendered_images_exist:
        # print("WHOOPS: TMEP!")
        # quit()
        print("Rendering images...")
        render_images(render_fn, config, ablation=args.ablation, load_fn=args.model_fn)
    else:
        print("Using previously rendered images...")

    metrics_fn = args.metrics_fn
    split = "test" # hard-coded, TODO: generalize
    if metrics_fn is None:
        if args.ablation:
            metrics_fn = os.path.join(config.project_directory, f"metrics_ablation_{split}.csv")
        else:
            metrics_fn = os.path.join(config.project_directory, f"metrics_{split}.csv")

    # Calculate the image metrics.
    calculate_image_metrics(render_fn, gt_fn, metrics_fn, split="test")
    


###