"""
Train a model

"""
__date__ = "December 2024 - February 2025"

import argparse
import matplotlib
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
matplotlib.use("agg")
from torchmetrics.image import StructuralSimilarityIndexMeasure

from src.config_utils import Config
from src.data import FrameDataset
from src.model import PoseSplatter
from src.unet_3d import init_unet_primary_skip
from src.utils import get_cam_params

os.environ['TORCH_CUDA_ARCH_LIST'] = "8.6" # NOTE: move this!

LOSS_NAMES = ("iou", "ssim", "img")
LOSS_COLORS = ["goldenrod", "deepskyblue", "lightcoral", "darkorchid", "mediumseagreen"]



def get_iou_loss(predicted_mask, target_mask, eps=1e-6):
    if predicted_mask.shape != target_mask.shape:
        raise ValueError("Predicted and target masks must have the same shape.")
    intersection = (predicted_mask * target_mask).sum(dim=(-2, -1))
    union = (predicted_mask + target_mask - predicted_mask * target_mask).sum(dim=(-2, -1))
    iou = (intersection + eps) / (union + eps)
    return 1 - iou.mean()


def calculate_validation_loss(model, valid_loader, device, ssim_lambda, img_lambda, max_n_batches=None):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch_num, (mask, img, p_3d, angle, view_idx) in enumerate(valid_loader):
            assert mask.shape[0] == 1, "batch size must be 1"
            view_idx = int(view_idx[0])
            
            rgb, alpha = model(
                mask.to(device),
                img.to(device),
                p_3d.to(device),
                float(angle[0]),
                view_num=view_idx,
            )
            rgb = torch.permute(rgb[0], (2, 0, 1))  # [3, H, W]
            alpha = alpha[0, ..., 0] # [H, W]

            # Compute loss.
            img_idx = model.observed_views.index(view_idx)
            target_mask = mask[0,img_idx].to(device) # [H, W]
            target_img = img[0,img_idx].to(device) # [3, H, W]
            iou_loss = get_iou_loss(alpha, target_mask)
            ssim_loss = ssim_lambda * (1.0 - ssim(target_img[None], rgb[None]))
            img_loss = img_lambda * torch.abs(target_img - rgb).sum() / target_mask.sum()
            total_loss += iou_loss.item() + ssim_loss.item() + img_loss.item()

            batch_size = mask.shape[0]
            total_samples += batch_size

            if batch_num + 1 == max_n_batches:
                break
    model.train()
    return total_loss / total_samples


def train_one_epoch(
        model,
        optimizer,
        loader,
        device,
        ssim_lambda,
        img_lambda,
        pbar,
        last_epoch_loss,
        max_n_batches=None,
    ):
    """
    Train the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        loader (torch.utils.data.DataLoader): DataLoader for the training set.
        device (str): Device to use for computation ('cuda' or 'cpu').
        loss_names (tuple): Tuple of loss names for logging.
        ssim_lambda (float): Weight for the SSIM loss.
        img_lambda (float): Weight for the image loss.

    Returns:
        list: List of average losses for the epoch corresponding to loss_names.
    """
    model.train()  # Set the model to training mode
    epoch_loss = [0.0 for _ in LOSS_NAMES]
    total_samples = 0

    for batch_num, (mask, img, p_3d, angle, view_idx) in enumerate(loader):
        assert mask.shape[0] == 1, "batch size must be 1"
        view_idx = int(view_idx[0])
        
        # Zero gradients.
        optimizer.zero_grad()

        # Forward pass
        rgb, alpha = model(
            mask.to(device),
            img.to(device),
            p_3d.to(device),
            float(angle[0]),
            view_num=view_idx,
        ) # [1, 3, H, W], [1, H, W, 1]
        rgb = torch.permute(rgb[0], (2, 0, 1))  # [3, H, W]
        alpha = alpha[0, ..., 0] # [H, W]

        # Compute loss.
        img_idx = model.observed_views.index(view_idx)
        target_mask = mask[0,img_idx].to(device) # [H, W]
        target_img = img[0,img_idx].to(device) # [3, H, W]
        iou_loss = get_iou_loss(alpha, target_mask)
        ssim_loss = ssim_lambda * (1.0 - ssim(target_img[None], rgb[None]))
        img_loss = img_lambda * torch.abs(target_img - rgb).sum() / target_mask.sum()

        # Backward pass
        total_loss = iou_loss + img_loss + ssim_loss
        total_loss.backward()
        optimizer.step()

        # Accumulate losses.
        batch_size = len(rgb)
        epoch_loss[0] += iou_loss.item() * batch_size
        epoch_loss[1] += ssim_loss.item() * batch_size
        epoch_loss[2] += img_loss.item() * batch_size
        total_samples += batch_size
        loss = sum(epoch_loss) / total_samples

        pbar.set_description(f"epoch loss: {last_epoch_loss:.5f} b {batch_num:04d}: {loss:.5f}")

        if batch_num + 1 == max_n_batches:
            break

    # Average losses over the entire dataset
    avg_losses = [loss / total_samples for loss in epoch_loss]
    return avg_losses


@torch.no_grad()
def plot_predictions(model, loader, device, save_path="temp.pdf", num_examples=5):
    """
    Plot model predictions alongside ground truth images and save the plot to a file.

    Args:
        model (torch.nn.Module): The model to use for generating predictions.
        loader (torch.utils.data.DataLoader): DataLoader for the dataset.
        device (str): Device to use for computation ('cuda' or 'cpu').
        save_path (str): Path to save the output plot. Default is "temp.pdf".
        num_examples (int): Number of examples to plot. Default is 5.
    """
    model.eval()  # Set the model to evaluation mode
    _, axarr = plt.subplots(ncols=2, nrows=num_examples, figsize=(4, 2 * num_examples))
    
    j = 0
    for mask, img, p_3d, angle, view_idx in loader:
        assert mask.shape[0] == 1, "batch size must be 1"
        view_idx = int(view_idx[0])
        img_idx = model.observed_views.index(view_idx)

        rgb, _ = model(
            mask.to(device),
            img.to(device),
            p_3d.to(device),
            float(angle[0]),
            view_num=view_idx,
        )
        rgb = torch.permute(rgb, (0, 3, 1, 2))  # [B, C, H, W]

        axarr[j, 0].imshow(torch.permute(img[0,img_idx], (1, 2, 0)).cpu().numpy())
        axarr[j, 0].axis("off")
        

        axarr[j, 1].imshow(torch.permute(rgb[0], (1, 2, 0)).detach().cpu().numpy())
        axarr[j, 1].axis("off")
        
        j += 1
        if j >= num_examples:
            break

    axarr[0, 0].set_title("Ground Truth")
    axarr[0, 1].set_title("Prediction")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close("all")


def plot_losses(losses, validation_losses=None, valid_every=None, save_path="loss.pdf"):
    """
    Plot training and validation losses over epochs and save the plot to a file.

    Args:
        losses (list of lists): Training losses for each epoch. Each sublist contains losses for different metrics.
        validation_losses (list of tuples): Validation losses with epochs as (epoch, loss). Default is None.
        loss_names (tuple): Names of the loss components for plotting. Default is ("iou", "ssim", "img").
        save_path (str): Path to save the output plot. Default is "loss.pdf".
    """
    num_epochs = len(losses)

    # Plot training losses
    epochs = range(1, num_epochs + 1)
    for i, loss_name in enumerate(LOSS_NAMES):
        plt.semilogy(epochs, [loss[i] for loss in losses], c=LOSS_COLORS[i], label=loss_name)
    plt.semilogy(epochs, [sum(loss) for loss in losses], c=LOSS_COLORS[-2], label="all")

    # Plot validation losses as scatter points
    if validation_losses and valid_every:
        val_epochs = range(valid_every, num_epochs + 1, valid_every)
        plt.plot(val_epochs, validation_losses, marker='o', color=LOSS_COLORS[-1], label="val")

    ax = plt.gca()
    ax.minorticks_on()
    ax.grid(which='both')
    plt.legend(loc="best")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.title("Training and Validation Losses")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close("all")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train script for the model")
    parser.add_argument("config", type=str, help="Path to the config JSON file")
    parser.add_argument("--load", action="store_true", help="Flag to load a pre-trained model")
    parser.add_argument("--ablation", action="store_true", help="Flag to train the ablation model")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs (int, default: 50)")
    parser.add_argument("--max_batches", type=int, default=None, help="Max # batches (None or int, default: None")

    args = parser.parse_args()
    
    config = Config(args.config)
    load = args.load  # This will be True if --load is provided, False otherwise
    n_epochs = args.epochs

    print(f"Config file: {args.config}")
    print(f"Load flag: {args.load}")
    print(f"Ablation flag: {args.ablation}")
    print(f"Epochs: {n_epochs}")
    print(f"Max Batches: {args.max_batches}")

    intrinsic, extrinsic, Ps = get_cam_params(
        config.camera_fn,
        ds=config.image_downsample,
        up_fn=config.vertical_lines_fn,
        auto_orient=True,
        load_up_direction=not config.adaptive_camera,
    )
    C = len(Ps)

    device = "cuda"
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    img_fn = os.path.join(config.image_directory, "images.h5")
    volume_fn = os.path.join(config.volume_directory, "volumes.h5")
    dset_args = (img_fn, volume_fn, config.center_rotation_fn, C, config.holdout_views)

    num_workers = len(os.sched_getaffinity(0)) # available CPUs
    print("num workers:", num_workers)
    loader_kwargs = dict(batch_size=1, shuffle=True, num_workers=num_workers, prefetch_factor=2)
    dset = FrameDataset(*dset_args, split="train")
    loader = DataLoader(dset, **loader_kwargs)

    valid_dset = FrameDataset(*dset_args, split="valid")
    valid_loader = DataLoader(valid_dset, **loader_kwargs)

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
        ablation=args.ablation,
        volume_fill_color=config.volume_fill_color,
        holdout_views=config.holdout_views,
        adaptive_camera=config.adaptive_camera,
    )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    if load:
        if args.ablation:
            d = torch.load(config.model_fn[:-3] + "_ablation.pt")
        else:
            d = torch.load(config.model_fn)
        epoch = d["epoch"]
        losses = d["loss"]
        validation_losses = d["validation_losses"]
        model.load_state_dict(d["model_state_dict"])
        optimizer.load_state_dict(d["optimizer_state_dict"])
        print(f"Loaded checkpoint from epoch {epoch}.")
    else:
        if not args.ablation:
            for unet in model.unets:
                init_unet_primary_skip(unet)
            init_unet_primary_skip(model.final_unet)
        epoch = 0
        losses = []
        validation_losses = []

    pbar = tqdm(range(n_epochs))
    last_epoch_loss = 0.0
    for _ in pbar:
        epoch += 1

        avg_losses = train_one_epoch(
            model=model,
            optimizer=optimizer,
            loader=loader,
            device=device,
            ssim_lambda=config.ssim_lambda,
            img_lambda=config.img_lambda,
            pbar=pbar,
            last_epoch_loss=last_epoch_loss,
            max_n_batches=args.max_batches,
        )
        losses.append(avg_losses)
        last_epoch_loss = sum(avg_losses)

        if epoch % config.valid_every == 0:
            validation_loss = calculate_validation_loss(
                model,
                valid_loader,
                device,
                ssim_lambda=config.ssim_lambda,
                img_lambda=config.img_lambda,
                max_n_batches=args.max_batches,
            )
            validation_losses.append(validation_loss)

        if epoch % config.plot_every == 0:
            if model.ablation:
                prediction_fn = "reconstruction_ablation.pdf"
                loss_fn = "loss_ablation.pdf"
            else:
                prediction_fn = "reconstruction.pdf"
                loss_fn = "loss.pdf"
            
            plot_predictions(
                model,
                loader,
                device,
                save_path=os.path.join(config.project_directory, prediction_fn),
                num_examples=5,
            )

            plot_losses(
                losses,
                validation_losses=validation_losses,
                valid_every=config.valid_every,
                save_path=os.path.join(config.project_directory, loss_fn),
            )

        if epoch % config.save_every == 0:
            if model.ablation:
                checkpoint_fn = config.model_fn[:-3] + "_ablation.pt"
            else:
                checkpoint_fn = config.model_fn
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': losses,
                    'validation_losses': validation_losses,
                    'loss_names': LOSS_NAMES,
                },
                checkpoint_fn,
            )
