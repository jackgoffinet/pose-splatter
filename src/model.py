"""
Splatting model with 3d volumetric input

"""
__date__ = "November 2024 - January 2025"

import numpy as np
import torch
import torch.nn as nn
from gsplat.rendering import rasterization

from .shape_carving import create_3d_grid
from .shape_carver import ShapeCarver
from .unet_3d import Unet3D



class PoseSplatter(nn.Module):
    def __init__(
            self,
            intrinsics=None,
            extrinsics=None,
            W=None,
            H=None,
            device="cuda",
            in_channels=4,
            out_channels=8,
            base_filters=8,
            ell=0.18,
            grid_size=64,
            min_n=1024,
            max_n=16000,
            num_unets=3,
            color_clip=(0,0.99),
            prob_threshold=0.25,
            mask_threshold=0.25,
            mask_threshold_delta=0.05,
            volume_idx=None,
            ablation=False,
            volume_fill_color=0.45,
            holdout_views=[],
            adaptive_camera=False,
        ):
        super(PoseSplatter, self).__init__()
        assert volume_idx is not None

        self.H, self.W = H, W
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.min_n = min_n
        self.max_n = max_n
        self.num_unets = num_unets
        self.color_clip = color_clip
        self.prob_threshold = prob_threshold
        self.mask_threshold = mask_threshold
        self.mask_threshold_delta = mask_threshold_delta
        self.input_size = [(i2-i1) for i1,i2 in volume_idx]
        self.ablation = ablation
        self.holdout_views = holdout_views
        C = len(intrinsics)
        self.observed_views = [i for i in range(C) if i not in holdout_views]
        self.adaptive_camera = adaptive_camera
        
        self.background_color = torch.ones(3).to(device)

        # Define the camera stuff.
        self.Ks = torch.tensor(intrinsics).to(device, torch.float32) # [6,3,3]
        self.viewmats = torch.tensor(extrinsics).to(device, torch.float32)

        # Define the trainable scale offset.
        self.scale = nn.Parameter(-5.5 * torch.ones(1))

        # Define the grid.
        self.grid = create_3d_grid(ell, grid_size, volume_idx=volume_idx)
        self.grid = torch.tensor(self.grid).to(device, torch.float32)
        
        self.voxel_size = ell / grid_size

        # Define the ShapeCarver.
        self.carver = ShapeCarver(
            ell,
            grid_size,
            intrinsics[np.array(self.observed_views)],
            extrinsics[np.array(self.observed_views)],
            volume_idx=volume_idx,
            volume_fill_color=volume_fill_color,
        )

        # Define the Gaussian parameter network.
        self.gaussian_param_net = nn.Sequential(
            nn.Linear(out_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 14),
        )

        # Define all the U-Nets.
        if not ablation:
            self.unets = nn.ModuleList(
                [
                    Unet3D(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        base_filters=base_filters,
                        input_size=self.input_size,
                    ) for _ in range(num_unets - 1)
                ],
            )

            self.final_unet = Unet3D(
                in_channels=in_channels,
                out_channels=out_channels,
                base_filters=base_filters,
                input_size=self.input_size,
            )
    

    def forward(self, mask, img, p_3d, angle, view_num=None):
        if view_num is None:
            view_num = np.random.choice(self.observed_views)
        assert not self.training or view_num in self.observed_views, \
            f"view_num ({view_num[0]}) not in observed views ({self.observed_views})"
        
        # Make the volume.
        if self.adaptive_camera:
            volume, temp_K = self.carver(mask[0,:,None], img[0], p_3d, angle, adaptive=self.adaptive_camera)
        else:
            volume = self.carver(mask[0,:,None], img[0], p_3d, angle, adaptive=self.adaptive_camera)

        # Run the volume through the U-Nets.
        volume = self.process_volume(volume[None])

        # Get Gaussian parameters.
        means, quats, scales, opacities, colors = self.get_gaussian_params_from_volume(volume)

        # Rotate and shift the Gaussian means.
        c, s = np.cos(angle), np.sin(angle)
        rot_mat = torch.tensor([[c,-s,0], [s,c,0], [0,0,1]]).to(volume.device, torch.float32)
        means = means @ rot_mat.T + p_3d # [n,3]

        # Rotate the quaternions.
        rot_mat_2 = torch.tensor([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).to(quats.device)
        r = quaternion_matrix_torch_batch(quats)
        r = torch.einsum("ij,bjk->bik", rot_mat_2.to(torch.float32), r)
        quats = quaternion_from_matrix_torch_batch(r)

        if self.adaptive_camera:
            out_K = temp_K[view_num].view(-1,3,3)
        else:
            out_K = self.Ks[view_num].view(-1,3,3)

        # Splat.
        rgb, alpha = self.splat(
            means,
            quats,
            scales,
            opacities,
            colors,
            self.viewmats[view_num].view(-1,4,4),
            out_K,
            self.W,
            self.H,
        )
        
        return rgb, alpha
    

    def get_gaussian_params_from_volume(self, volume):
        # Figure out which Gaussians to render.
        mt = self.mask_threshold
        probs = torch.sigmoid(volume[0] - mt) # [n^3]
        pt = self.prob_threshold
        mask = probs > pt
        while mask.sum() > self.max_n:
            mt += self.mask_threshold_delta
            probs = torch.sigmoid(volume[0] - mt) # [n^3]
            mask = probs > pt
        while mask.sum() < self.min_n:
            mt -= self.mask_threshold_delta
            probs = torch.sigmoid(volume[0] - mt) # [n^3]
            mask = probs > pt

        if mask.sum() > self.max_n:
            indices = torch.nonzero(mask, as_tuple=True)[0]
            rand_idx = torch.randperm(len(indices))[:self.max_n].to(mask.device)
            keep_indices = indices[rand_idx]
            mask[:] = False  # Reset all values to False
            mask[keep_indices] = True  # Set only the selected indices to True
        
        # Send each Gaussian through an MLP to get parameters.
        net_out = self.gaussian_param_net(volume[:,mask].T)
        quats, scales, opacities, colors, delta_means = torch.split(net_out, (4,3,1,3,3), dim=1)

        # Process the parameters so that they're in good ranges.
        colors = torch.sigmoid(colors).clamp(self.color_clip[0], self.color_clip[1])
        scales = torch.exp(scales + self.scale[0])  
        opacities = ((1 / (1 - pt)) * (probs[mask] - pt)).clamp(0.0, 1.0)
        opacities = torch.sigmoid(opacities)
        means = self.grid.view(-1,3)[mask] + 2 * self.voxel_size * torch.tanh(delta_means)

        return means, quats, scales, opacities, colors
    

    def process_volume(self, volume):
        if self.ablation:
            # Just concatenate zeros to the volume.
            pad_len = self.out_channels - self.in_channels
            pad = torch.zeros((pad_len, *volume.shape[2:])).to(volume.device, volume.dtype)
            volume = volume[0]
            volume = torch.cat([volume, pad], 0)
            return volume.view(volume.shape[0], -1)
        # Run the volume through the U-Nets.
        for unet in self.unets:
            out, _ = unet(volume) # [b,c,n,n,n]
            volume = volume + out
        volume, _ = self.final_unet(volume) # [b,c,n,n,n]
        volume = volume[0].view(volume.shape[1], -1) # [c,n^3]
        return volume

        
    def splat(self, means, quats, scales, opacities, colors, viewmats, Ks, width, height, radius_clip=2.0):
        """Splat the Gaussians onto an image plane."""
        # Render the image.
        render, alpha, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode="RGB",
            sh_degree=None,
            sparse_grad=False,
            absgrad=True,
            rasterize_mode="classic",
            radius_clip=radius_clip,
        ) # [1, H, W, 3] [1, H, W, 1]

        rgb = render[:, ..., :3] + (1 - alpha) * self.background_color
        rgb = torch.clamp(rgb, 0.0, 1.0) # [1, H, W, 3]
        return rgb, alpha


def quaternion_matrix_torch_batch(quats):
    b = quats.shape[0]
    _EPS = 4 * np.finfo(float).eps
    quats = quats.clone().double()
    n = torch.sum(quats**2, dim=1)
    mask = n < _EPS
    n[mask] = 1.0
    quats = quats * torch.sqrt(2.0 / n).unsqueeze(1)
    outer = torch.einsum("bi,bj->bij", quats, quats)
    eye = torch.eye(4, dtype=torch.float64, device=quats.device)
    res = eye.unsqueeze(0).repeat(b, 1, 1)

    res[:, 0, 0] = res[:, 0, 0] - outer[:, 2, 2] - outer[:, 3, 3]
    res[:, 0, 1] = outer[:, 1, 2] - outer[:, 3, 0]
    res[:, 0, 2] = outer[:, 1, 3] + outer[:, 2, 0]
    res[:, 1, 0] = outer[:, 1, 2] - outer[:, 3, 0]
    res[:, 1, 1] = res[:, 1, 1] + outer[:, 0, 0] - outer[:, 0, 0]
    res[:, 1, 2] = outer[:, 2, 3] - outer[:, 1, 0]
    res[:, 2, 0] = outer[:, 1, 3] - outer[:, 2, 0]
    res[:, 2, 1] = outer[:, 2, 3] + outer[:, 1, 0]
    res[:, 2, 2] = res[:, 2, 2] - outer[:, 1, 1] - outer[:, 2, 2]

    res[mask] = eye
    return res.to(torch.float32)


def quaternion_from_matrix_torch_batch(mats):
    mats = mats.double()
    
    m00 = mats[:, 0, 0]
    m01 = mats[:, 0, 1]
    m02 = mats[:, 0, 2]
    m10 = mats[:, 1, 0]
    m11 = mats[:, 1, 1]
    m12 = mats[:, 1, 2]
    m20 = mats[:, 2, 0]
    m21 = mats[:, 2, 1]
    m22 = mats[:, 2, 2]

    K = torch.stack([
        torch.stack([m00 - m11 - m22, m01 + m10, m02 + m20, m21 - m12], 1),
        torch.stack([m01 + m10, m11 - m00 - m22, m12 + m21, m02 - m20], 1),
        torch.stack([m02 + m20, m12 + m21, m22 - m00 - m11, m10 - m01], 1),
        torch.stack([m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22], 1),
    ], 1)
    K = K / 3.0

    _, V = torch.linalg.eigh(K)

    quats = V[:, :, -1]
    quats = quats[:, [3 ,0, 1, 2]]
    mask = quats[:, 0] < 0
    quats[mask] = -1.0 * quats[mask]
    return quats.to(torch.float32)


if __name__ == '__main__':
    pass


###