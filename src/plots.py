"""
Plotting functions.

"""
__date__ = "December 2024"


import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.patches import Ellipse
from matplotlib.colors import Normalize
from matplotlib.cm import viridis
from gsplat.rendering import rasterization

from .tracking import track_principal_axes


def plot_point_clouds_gsplat(pcs, intrinsic, extrinsic, W, H, device="cuda"):
    all_means = [pc[:,:3] for pc in pcs]
    all_colors = [pc[:,3:] for pc in pcs]

    view_idx = np.random.randint(len(intrinsic))
    _, axarr = plt.subplots(ncols=1, nrows=len(pcs))
    for i in range(len(axarr)):
        plt.sca(axarr[i])
        means = all_means[i].to(device, torch.float32)
        colors = all_colors[i].to(device, torch.float32).clamp(0,1)

        Ks = torch.tensor(intrinsic).to(device, torch.float32) # [6,3,3]
        n = len(means)
        quats = torch.ones((n,4)).to(device, torch.float32)
        scales = -7.0 * torch.ones((n,3)).to(device, torch.float32)
        opacities = torch.ones((n,1)).to(device, torch.float32)
        background_color = torch.ones(3).to(device, torch.float32)

        viewmat = torch.tensor(extrinsic[view_idx]).to(device, torch.float32).unsqueeze(0)
        K = Ks[i][None]
        
        render, alpha, _ = rasterization(
            means=means,
            quats=quats,  # rasterization does normalization internally
            scales=torch.exp(scales),
            opacities=torch.sigmoid(opacities).squeeze(-1),
            colors=colors,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=W,
            height=H,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode="RGB",
            sh_degree=None,
            sparse_grad=False,
            absgrad=True,
            rasterize_mode="classic",
            # radius_clip=3.0,
        ) # [1, H, W, 3] [1, H, W, 1]
    
        rgb = render[:, ..., :3] + (1 - alpha) * background_color
        rgb = torch.clamp(rgb, 0.0, 1.0)

        plt.imshow(rgb[0].detach().cpu().numpy())
    plt.savefig("temp.pdf")
    plt.close("all")


def plot_gsplat_color(volume, grid, intrinsic, extrinsic, W, H, device="cuda"):
    mask = volume[0].reshape(-1) > 0.5
    colors = volume[1:4].reshape(3,-1)[:,mask].T
    grid = torch.tensor(grid).to(device, torch.float32)
    mask = torch.tensor(mask).to(device, torch.bool)
    means = grid.view(-1,3)[mask] # [n,3]
    colors = torch.tensor(colors).to(device, torch.float32).clamp(0, 1)

    Ks = torch.tensor(intrinsic).to(device, torch.float32) # [6,3,3]
    n = len(means)
    quats = torch.ones((n,4)).to(device, torch.float32)
    scales = -7.0 * torch.ones((n,3)).to(device, torch.float32)
    opacities = torch.ones((n,1)).to(device, torch.float32)
    background_color = torch.ones(3).to(device, torch.float32)

    _, axarr = plt.subplots(ncols=3, nrows=2)
    axarr = axarr.flatten()
    for i in range(len(axarr)):
        plt.sca(axarr[i])

        viewmat = torch.tensor(extrinsic[i]).to(device, torch.float32).unsqueeze(0)
        K = Ks[i][None]
        
        render, alpha, _ = rasterization(
            means=means,
            quats=quats,  # rasterization does normalization internally
            scales=torch.exp(scales),
            opacities=torch.sigmoid(opacities).squeeze(-1),
            colors=colors,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=W,
            height=H,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode="RGB",
            sh_degree=None,
            sparse_grad=False,
            absgrad=True,
            rasterize_mode="classic",
            # radius_clip=3.0,
        ) # [1, H, W, 3] [1, H, W, 1]
    
        rgb = render[:, ..., :3] + (1 - alpha) * background_color
        rgb = torch.clamp(rgb, 0.0, 1.0)

        plt.imshow(rgb[0].detach().cpu().numpy())
    plt.savefig("temp.pdf")
    plt.close("all")


def plot_voxel_grid(volume, thresh=5/6, fn="temp.jpg"):
    voxel_grid = volume[0] >= thresh

    # Set up the figure and subplots
    fig = plt.figure(figsize=(12, 9))

    # Angles for viewing (elevation, azimuth)
    el = 20
    angles = [(el,i*30) for i in range(9)]

    for i, angle in tqdm(enumerate(angles)):
        ax = fig.add_subplot(3, len(angles) // 3, i + 1, projection='3d')
        ax.voxels(voxel_grid, facecolors='tab:blue', alpha=0.5)
        ax.view_init(elev=angle[0], azim=angle[1])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    plt.tight_layout()
    plt.savefig(fn)
    plt.close("all")


def get_gsplat_color(volume, grid, intrinsic, extrinsic, W, H, device="cuda", thresh=0.5, return_alpha=False):
    C = len(intrinsic)
    if not isinstance(volume, torch.Tensor):
        volume = torch.tensor(volume)
    mask = volume[0].reshape(-1) > thresh
    colors = volume[1:4].reshape(3,-1)[:,mask].T
    grid = torch.tensor(grid).to(device, torch.float32)
    mask = mask.to(device, torch.bool)
    means = grid.view(-1,3)[mask] # [n,3]
    colors = colors.to(device, torch.float32).clamp(0, 1)

    Ks = torch.tensor(intrinsic).to(device, torch.float32) # [6,3,3]
    n = len(means)
    quats = torch.ones((n,4)).to(device, torch.float32)
    scales = -7.0 * torch.ones((n,3)).to(device, torch.float32)
    opacities = torch.ones((n,1)).to(device, torch.float32)
    background_color = torch.ones(3).to(device, torch.float32)

    rgbs = []
    for i in range(C):
        viewmat = torch.tensor(extrinsic[i]).to(device, torch.float32).unsqueeze(0)
        K = Ks[i][None]
        
        render, alpha, _ = rasterization(
            means=means,
            quats=quats,  # rasterization does normalization internally
            scales=torch.exp(scales),
            opacities=torch.sigmoid(opacities).squeeze(-1),
            colors=colors,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=W,
            height=H,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode="RGB",
            sh_degree=None,
            sparse_grad=False,
            absgrad=True,
            rasterize_mode="classic",
            # radius_clip=3.0,
        ) # [1, H, W, 3] [1, H, W, 1]
    
        if return_alpha:
            rgb = alpha
        else:
            rgb = render[:, ..., :3] + (1 - alpha) * background_color
        rgb = torch.clamp(rgb, 0.0, 1.0)
        rgbs.append(rgb[0].clone().detach().cpu().numpy())

    return np.array(rgbs)


def plot_imgs_and_volume(imgs, volume, grid, intrinsic, extrinsic, W, H, fn="temp.pdf"):
    C = len(intrinsic)
    _, axarr = plt.subplots(nrows=C, ncols=3)
    rgbs = get_gsplat_color(volume, grid, intrinsic, extrinsic, W, H)
    rgbs_a = get_gsplat_color(volume, grid, intrinsic, extrinsic, W, H, return_alpha=True)
    rgbs_a = np.concatenate([rgbs_a, 0*rgbs_a, 0*rgbs_a], -1)

    print("W", W, "H", H, "imgs", imgs.shape)
    for i in range(C):
        plt.sca(axarr[i,0])
        plt.imshow(imgs[i])
        plt.axis("off")
        if i == 0:
            plt.title("True Images")
        plt.sca(axarr[i,1])
        plt.imshow(rgbs[i])
        plt.axis("off")
        if i == 0:
            plt.title("Rendered Images")

        plt.sca(axarr[i,2])
        img = imgs[i].mean(axis=-1)
        img = np.where(img < 0.95, 1.0, 0.0)
        img = img[...,None]
        img = np.concatenate([0*img, 0*img, img], -1)
        plt.imshow(img + rgbs_a[i])
        plt.axis("off")
        if i == 0:
            plt.title("Overlay Images")

    plt.savefig(fn)
    plt.close("all")


def plot_color_voxel_grid(volume, thresh=5/6, alpha=0.8, shade=False, fn="temp.jpg"):
    voxel_grid = volume[0] >= thresh

    rgba = np.concatenate([volume[1:], alpha * np.ones_like(volume[:1])], 0)
    rgba = np.transpose(rgba, (1,2,3,0))

    # Set up the figure and subplots
    fig = plt.figure(figsize=(12, 9))

    # Angles for viewing (elevation, azimuth)
    el = 20
    angles = [(el,i*30) for i in range(9)]

    for i, angle in tqdm(enumerate(angles)):
        ax = fig.add_subplot(3, len(angles) // 3, i + 1, projection='3d')
        ax.voxels(voxel_grid, facecolors=rgba, edgecolors=rgba, shade=shade)
        ax.view_init(elev=angle[0], azim=angle[1])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    plt.tight_layout()
    plt.savefig(fn)
    plt.close("all")


def plot_ellipses(means, covariances, fn="temp.pdf"):
    assert means.ndim == 2
    assert covariances.ndim == 3
    n = len(means)

    principal_axes = track_principal_axes(means, covariances)
    principal_axes = principal_axes[:, :2]
    means = means[:, :2]

    # Colormap
    cmap = viridis
    norm = Normalize(vmin=0, vmax=n)

    # Function to plot an ellipse for a Gaussian
    def plot_gaussian(ax, mean, cov, color, pa):
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        a, b = eigenvectors[:, -1]
        angle = np.angle(a + 1j * b)
        height, width = 2 * np.sqrt(eigenvalues)

        ellipse = Ellipse(xy=mean, width=width, height=height, angle=np.degrees(angle), edgecolor='black', facecolor=color, alpha=0.7)
        ax.add_patch(ellipse)
        pa *= 0.8 * np.sqrt(np.max(eigenvalues))
        plt.arrow(mean[0], mean[1], pa[0], pa[1], color="k")

    # Plotting
    _, ax = plt.subplots(figsize=(8, 6))

    for i in range(n):
        color = cmap(norm(i))
        plot_gaussian(ax, means[i], covariances[i], color, principal_axes[i])

    # Configure the plot
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_aspect("equal")

    stds = np.sqrt(covariances[:,np.arange(2),np.arange(2)])
    min_x = min(mu[0] - 2 * std[0] for mu, std in zip(means, stds))
    max_x = max(mu[0] + 2 * std[0] for mu, std in zip(means, stds))
    min_y = min(mu[1] - 2 * std[1] for mu, std in zip(means, stds))
    max_y = max(mu[1] + 2 * std[1] for mu, std in zip(means, stds))
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    # ax.grid(True)

    # Show the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Gaussian Index')

    plt.savefig(fn)
    plt.close("all")
