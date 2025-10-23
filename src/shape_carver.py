"""
Shape carving module.

"""
__date__ = "February 2025"


import numpy as np
import torch
import torch.nn as nn
import torch_scatter

from .shape_carving import create_3d_grid, adjust_principal_points_to_seed


def get_volume_torch(images, intrinsic_matrices, extrinsic_matrices, grid_points):
    """
    images:            [n_cameras, c, h, w] (float or uint8, etc.)
    intrinsic_matrices:[n_cameras, 3, 3]
    extrinsic_matrices:[n_cameras, 4, 4]
    grid_points:       [n1, n2, n3, 3]
    
    Returns:
        averaged_values: [c, n1, n2, n3]
    """
    # Reshape grid_points to [N, 3]
    assert grid_points.ndim == 4 and grid_points.shape[-1] == 3
    n1, n2, n3 = grid_points.shape[:3]
    N = n1 * n2 * n3
    
    grid_points_flat = grid_points.reshape(-1, 3)  # [N, 3]

    # Project all points for all cameras in a batched way
    # --> all_projected_coords: [n_cameras, N, 2]
    all_projected_coords = project_points_torch(
        grid_points_flat, intrinsic_matrices, extrinsic_matrices
    )  # [n_cameras, N, 2]

    # Sample nearest pixels from each camera's image
    # --> sampled_values: [n_cameras, N, c]
    sampled_values = sample_nearest_pixels_torch(images, all_projected_coords)

    # Average across cameras: [N, c]
    averaged_values = sampled_values.mean(dim=0)  # now [N, c]

    # Reshape and transpose to get shape [c, n1, n2, n3]
    averaged_values = averaged_values.permute(1, 0)  # [c, N]
    averaged_values = averaged_values.reshape(-1, n1, n2, n3)  # [c, n1, n2, n3]

    return averaged_values


def project_points_torch(points, intrinsic_matrices, extrinsic_matrices):
    """
    points:             [N, 3]
    intrinsic_matrices: [n_cameras, 3, 3]
    extrinsic_matrices: [n_cameras, 4, 4]
    
    Return:
       pixel_coords: [n_cameras, N, 2]
    """
    device = points.device
    N = points.shape[0]
    
    # 1) Convert points to homogeneous: [N, 4]
    ones = torch.ones(N, 1, device=device, dtype=points.dtype)
    points_h = torch.cat([points, ones], dim=-1)  # [N, 4]

    # 2) Apply extrinsic: world -> camera
    # We want a result of shape [n_cameras, N, 4]
    # extrinsic_matrices: [n_cameras, 4, 4]
    # points_h:           [   1,           N, 4] (we'll unsqueeze for broadcast)
    points_h = points_h.unsqueeze(0).transpose(1, 2)  # -> [1, 4, N]
    # Now do batch matmul: [n_cameras, 4, 4] x [1, 4, N] => [n_cameras, 4, N]
    camera_points_h = extrinsic_matrices @ points_h  # [n_cameras, 4, N]
    
    # Now transpose to get [n_cameras, N, 4]
    camera_points_h = camera_points_h.transpose(1, 2)  # [n_cameras, N, 4]

    # 3) Apply intrinsic: camera -> image plane (still homogeneous)
    #    camera_points: [n_cameras, N, 3] (discard last column if you like, but typically use first 3)
    camera_points = camera_points_h[..., :3]  # [n_cameras, N, 3]
    
    # We'll do batched matmul with intrinsic: [n_cameras, 3, 3] x [n_cameras, N, 3]
    # But we need to transpose the second argument
    camera_points = camera_points.transpose(1, 2)  # [n_cameras, 3, N]
    pixel_coords_h = intrinsic_matrices @ camera_points  # [n_cameras, 3, N]
    pixel_coords_h = pixel_coords_h.transpose(1, 2)      # [n_cameras, N, 3]
    
    # 4) Normalize by z
    # pixel_coords: [n_cameras, N, 2]
    pixel_coords = pixel_coords_h[..., :2] / (pixel_coords_h[..., 2:3] + 1e-8)

    return pixel_coords


def sample_nearest_pixels_torch(images, pixel_coords):
    """
    images:       [n_cameras, c, h, w]
    pixel_coords: [n_cameras, N, 2] in (x=column, y=row) format
    
    Returns:
        sampled_values: [n_cameras, N, c]
    """
    device = images.device
    n_cameras, c, h, w = images.shape
    _, N, _ = pixel_coords.shape  # same n_cameras

    # Round coordinates and clamp
    x = pixel_coords[..., 0].round().long().clamp(min=0, max=w-1)  # [n_cameras, N]
    y = pixel_coords[..., 1].round().long().clamp(min=0, max=h-1)  # [n_cameras, N]

    # We want to index images[k, :, y, x] for each camera k and each point index
    # One way to do that in a batched manner is:
    #   images[ torch.arange(n_cameras)[:,None], :, y, x]
    
    # Build camera index array: shape [n_cameras, 1] -> broadcast with [n_cameras, N]
    camera_idx = torch.arange(n_cameras, device=device)[:, None]  # [n_cameras, 1]

    # Now gather the pixel values: [n_cameras, N, c]
    # we reorder dimensions to do advanced indexing on the last two dims for y and x.
    sampled = images[camera_idx, :, y, x]  # [n_cameras, c, N]

    # Transpose to get [n_cameras, N, c]
    sampled = sampled.permute(0, 2, 1)     # [n_cameras, N, c]

    return sampled


def ray_cast_visibility_torch(
        grid_points,
        intrinsic_matrices,
        extrinsic_matrices,
        image_height,
        image_width,
    ):
    """
    grid_points:       [N, 3] float tensor of voxel coordinates (on CPU or GPU).
    intrinsic_matrices:[C, 3, 3] float tensor (C cameras).
    extrinsic_matrices:[C, 4, 4] float tensor (C cameras).
    image_height, image_width: ints
    Returns:
        visibility:     [C, N] bool tensor; visibility[c, i] == True if voxel i is visible to camera c.
    """

    device = grid_points.device
    C = intrinsic_matrices.shape[0]
    N = grid_points.shape[0]

    # We'll store final visibility in a [C, N] boolean tensor (all False initially)
    visibility = torch.zeros(C, N, dtype=torch.bool, device=device)

    # For each camera, do:
    # 1) Compute distances from camera center to each voxel
    # 2) Project each voxel => pixel coords
    # 3) Flatten pixel coords to 1D index
    # 4) Use scatter_min to find the frontmost distance among the voxels that land in each pixel
    # 5) Mark those voxels as visible

    # Precompute each camera center (camera_position) from extrinsics
    # camera center in world coords:  -R^T t   (where extrinsic = [R | t; 0 0 0 1])
    # R = extrinsic[:3,:3], t = extrinsic[:3, 3]
    R = extrinsic_matrices[:, :3, :3]   # [C, 3, 3]
    t = extrinsic_matrices[:, :3, 3]    # [C, 3]
    # camera_positions = - R^T @ t
    # shape [C, 3]
    camera_positions = - torch.einsum('cij,cj->ci', R.permute(0,2,1), t)

    for c_idx in range(C):
        # Camera center
        cam_pos = camera_positions[c_idx]  # [3]
        # Distances: [N]
        distances = (grid_points - cam_pos).norm(dim=-1)

        # Project points into pixel coords
        # project_points_torch -> (N, 2) for a single camera
        pixel_coords = project_points_torch_single_cam(
            grid_points, intrinsic_matrices[c_idx], extrinsic_matrices[c_idx]
        )  # [N, 2], float

        # Round to nearest pixel and clamp to [0, W-1], [0, H-1]
        px = pixel_coords[:, 0].round().long().clamp(0, image_width  - 1)
        py = pixel_coords[:, 1].round().long().clamp(0, image_height - 1)

        # Flatten (row=y, col=x) => single index
        # For pixel (y,x), flattened_index = y * W + x
        pixel_idx = py * image_width + px  # [N]

        # Use scatter_min to find the minimal distance among all points that land in each pixel index
        # shape: (N,) -> we scatter into a buffer of size (H*W,)
        max_pixels = image_height * image_width  # total number of possible pixel positions
        init_val = distances.new_full((max_pixels,), float('inf'))  # [H*W] for min scatter

        # out: hold the per-pixel minimal distance
        # arg: hold the per-pixel argmin (which voxel index has that minimal distance)
        # out and arg will each be shape [H*W].
        out, arg = torch_scatter.scatter_min(distances, pixel_idx, out=init_val)
        idx_range = torch.arange(N, device=device)
        visible_mask = (idx_range == arg[pixel_idx]) & (out[pixel_idx] < float('inf'))

        # Mark those voxels as visible
        visibility[c_idx, visible_mask] = True

    return visibility


def project_points_torch_single_cam(points, intrinsic_matrix, extrinsic_matrix):
    """
    points:          [N, 3]
    intrinsic_matrix:[3, 3]
    extrinsic_matrix:[4, 4]
    Returns:
        pixel_coords: [N, 2]
    """
    # Convert to homogeneous
    N = points.shape[0]
    device = points.device
    ones = torch.ones(N, 1, device=device, dtype=points.dtype)
    points_h = torch.cat([points, ones], dim=-1)  # [N, 4]

    # World->camera
    # extrinsic: [4,4], points_h: [N,4] => [N,4]
    camera_points_h = (extrinsic_matrix @ points_h.T).T  # [N,4]

    # Keep camera coords [N,3]
    camera_points = camera_points_h[:, :3]

    # Apply intrinsics [3,3]
    # => pixel_coords_h: [N,3]
    pixel_coords_h = (intrinsic_matrix @ camera_points.transpose(0,1)).transpose(0,1)

    # Normalize by z (avoid /0 by small epsilon)
    pixel_coords = pixel_coords_h[:, :2] / (pixel_coords_h[:, 2:3].clamp(min=1e-8))

    return pixel_coords


def compute_voxel_colors_torch(
        grid_points,
        images,
        intrinsic_matrices,
        extrinsic_matrices,
        nonvisible_weight=0.25,
    ):
    """
    grid_points:        [N, 3] float
    images:             [C, 3, H, W] 
    intrinsic_matrices: [C, 3, 3]
    extrinsic_matrices: [C, 4, 4]
    nonvisible_weight:  float
    
    Returns:
      voxel_colors: [N, 3] float
    """
    device = grid_points.device
    C = images.shape[0]
    N = grid_points.shape[0]

    # 1) Compute visibility with our new GPU-friendly approach
    #    We need image H, W
    #    Suppose images is [C, H, W, 3]. Then:
    _, H, W, _ = images.shape
    visibility = ray_cast_visibility_torch(grid_points,
                                           intrinsic_matrices,
                                           extrinsic_matrices,
                                           H, W)  # [C, N] bool
    
    # 2) Project all voxels into all cameras (batched or loop).
    #    For clarity, we’ll do a loop. 
    #    projected_coords[c] => [N, 2]
    projected_coords = []
    for c_idx in range(C):
        pc = project_points_torch_single_cam(
            grid_points, 
            intrinsic_matrices[c_idx],
            extrinsic_matrices[c_idx]
        )
        projected_coords.append(pc)
    # stacked: [C, N, 2]
    projected_coords = torch.stack(projected_coords, dim=0)

    # 3) Sample colors for each camera from images
    #    images: [C, H, W, 3]. 
    #    So for camera c, pixel = (y,x).
    #    We'll produce an array: [C, N, 3].
    sampled_colors = sample_nearest_pixels_torch(images, projected_coords)  

    # 4) Build weights. If visible => weight=1, else = nonvisible_weight.
    weights = torch.where(visibility, 
                          torch.tensor(1.0, device=device),
                          torch.tensor(nonvisible_weight, device=device))  # [C, N]
    
    # Normalize weights per voxel:
    # sum over camera dimension => shape [1, N]
    denom = weights.sum(dim=0, keepdim=True).clamp(min=1e-8)
    weights_norm = weights / denom  # [C, N]

    # 5) Weighted color
    # multiply by sampled_colors => [C, 3, N]
    weighted_colors = weights_norm[:, None] * sampled_colors
    # sum over cameras => [N, 3]
    voxel_colors = weighted_colors.sum(dim=0)

    return voxel_colors # [3,N]



class ShapeCarver(nn.Module):

    def __init__(self, ell, grid_size, K, E, volume_idx=None, device="cuda", volume_fill_color=0.45):
        super(ShapeCarver, self).__init__()
        self.device = device
        self.volume_fill_color = volume_fill_color
        grid = create_3d_grid(ell, grid_size, volume_idx=volume_idx)
        self.grid = torch.tensor(grid).to(device, torch.float32)
        self.K = torch.tensor(K).to(device, torch.float32)
        self.E = torch.tensor(E).to(device, torch.float32)
        self.C = len(K)


    def forward(self, mask, rgb, center, angle, adaptive=False):
        assert mask.ndim == 4 # [C,1,H,W]
        assert rgb.ndim == 4 # [C,3,H,W]
        assert len(mask) == self.C, f"{mask.shape}, {self.C}"
        assert len(rgb) == self.C, f"{rgb.shape}, {self.C}"
        
        if adaptive:
            temp_K, center = adjust_principal_points_to_seed(
                mask[:,0].cpu().numpy(),
                self.K.cpu().numpy(),
                self.E.cpu().numpy(),
            )
            temp_K = torch.tensor(temp_K).to(rgb.device, torch.float32)
            center = torch.tensor(center).to(rgb.device, torch.float32)
        else:
            temp_K = self.K

        grid = self.get_grid_points(center, angle) # [n1,n2,n3,3]
        n1, n2, n3 = grid.shape[:3]
            
        mask_volume = get_volume_torch(
            mask,
            temp_K,
            self.E,
            grid,
        ) # [1, n1, n2, n3]

        out = 0.0
        for thresh in [1, (self.C-1) / self.C]:
            binary_volume = (mask_volume >= thresh).flatten() # [N]
            
            means = grid.view(-1,3)[binary_volume] # [n,3]
            colors = compute_voxel_colors_torch(means, rgb, self.K, self.E) # [3,n]

            volume = self.volume_fill_color * torch.ones((4, n1 * n2 * n3), dtype=torch.float32, device=self.device)
            volume[0] = binary_volume.to(torch.float32)
            volume[1:,binary_volume] = colors

            volume = volume.view(4, n1, n2, n3)

            # # Check the carved volume.
            # from .plots import plot_gsplat_color
            # plot_gsplat_color(volume.detach().cpu().numpy(),
            #                   grid.detach().cpu().numpy(),
            #                   self.K.cpu().numpy(), self.E.cpu().numpy(), 2048//4, 1536//4)
            # quit()
            
            out = out + volume / 2
        if adaptive:
            return out, temp_K
        return out



    def get_grid_points(self, center, angle):
        c, s = np.cos(angle), np.sin(angle)
        rot_mat = torch.tensor([[c,-s,0],[s,c,0],[0,0,1]]).to(self.device, torch.float32)
        temp_grid = torch.einsum("abci,ji->abcj", self.grid, rot_mat)
        temp_grid = temp_grid + center.view(1,1,1,3).to(self.device)
        return temp_grid

