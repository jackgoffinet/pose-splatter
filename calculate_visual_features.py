"""
Calculate visual features of the frames.

"""
__date__ = "January - March 2025"


import argparse
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm
from numpy.polynomial.legendre import leggauss
from scipy.special import sph_harm

from src.config_utils import Config
from src.data import FrameDataset
from src.model import PoseSplatter
from src.utils import get_cam_params

os.environ['TORCH_CUDA_ARCH_LIST'] = "8.6" # NOTE: move this!



def camera_extrinsic_spherical(radius, theta, phi):
    """
    Returns a 4x4 extrinsic matrix (in OpenCV convention) for a camera located
    on the sphere of radius `radius` centered at the origin, at angles `theta` and `phi`.
    
    The camera looks at the origin, and its 'up' direction is aligned with the global +Z axis.

    Parameters
    ----------
    radius : float
        The radius of the sphere on which the camera is placed.
    theta : float
        Polar angle in radians from the positive Z-axis down.
    phi : float
        Azimuthal angle in radians around the Z-axis.
        
    Returns
    -------
    E : (4, 4) ndarray
        The 4x4 extrinsic matrix that transforms points from world coordinates
        to this camera's coordinates.
    """
    # 1. Convert from spherical to Cartesian coordinates for the camera center
    #    Here we assume the usual physics convention:
    #       theta is the angle from the +Z axis down  (0 <= theta <= pi)
    #       phi   is the azimuth around the Z axis    (0 <= phi <= 2*pi)
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    
    # Camera center in world coordinates
    C = np.array([x, y, z], dtype=float)

    # 2. Define the camera's forward direction (the +Z axis in camera space).
    #    We want the camera to look at the origin, so the forward vector is
    #    (origin - C).  Here, that is -C (normalized).
    forward = -C
    forward /= np.linalg.norm(forward)

    # 3. Define the "global up" vector in world coordinates
    global_up = -np.array([0.0, 0.0, 1.0], dtype=float)

    # 4. Compute the camera's right vector as the cross product of (global_up, forward)
    #    and normalize it.  This will be the +X axis of the camera.
    right = np.cross(global_up, forward)
    right /= np.linalg.norm(right)

    # 5. Compute the camera's true up vector as cross(forward, right), and normalize.
    #    This will be the +Y axis of the camera.
    up = np.cross(forward, right)
    up /= np.linalg.norm(up)

    # 6. Assemble the rotation matrix (world-to-camera).
    #    Rows of R are [right; up; forward].
    #    That is, row 0 is the camera's X axis in world coords,
    #    row 1 is the camera's Y axis in world coords,
    #    row 2 is the camera's Z axis in world coords.
    R = np.stack([right, up, forward], 1).T

    # R = np.stack([right, up, forward], axis=0).T

    # 7. Compute the translation such that X_cam = R * (X_world - C).
    #    In the usual [R | t] form, we have t = -R * C.
    t = -R @ C

    # 8. Build the full 4x4 extrinsic matrix in homogeneous coordinates.
    E = np.eye(4)
    E[:3, :3] = R
    E[:3,  3] = t

    return E


def build_A(L, w, thetas, phis):
    """
    Construct the matrix A of shape ((L+1)^2, N_theta*N_phi).

    Parameters
    ----------
    L      : int
        Maximum spherical-harmonic degree (0 <= ell <= L).
    w      : array_like, shape=(N_theta,)
        Gauss-Legendre weights for the theta integration (accounting for sin(theta)).
    thetas : array_like, shape=(N_theta,)
        The Gauss-Legendre nodes mapped to theta in (0, pi).
    phis   : array_like, shape=(N_phi,)
        Uniformly spaced phi values in [0, 2*pi).

    Returns
    -------
    A : np.ndarray, shape=((L+1)^2, N_theta*N_phi), dtype=complex
        The matrix whose row corresponds to (ell,m) and column to (k,j).
        Multiplying A by flattened fvals yields the vector of spherical-harmonic coefficients.
    """
    N_theta = len(thetas)
    N_phi   = len(phis)
    dphi    = 2.0 * np.pi / N_phi

    # total_sph_harmonics = sum_{ell=0..L} (2 ell + 1) = (L+1)^2
    total_sph_harmonics = (L+1)*(L+1)

    # Create empty matrix
    A = np.zeros((total_sph_harmonics, N_theta*N_phi), dtype=complex)

    row_index = 0
    for ell in range(L+1):
        for m in range(-ell, ell+1):
            # Build the row for this (ell, m)
            # We'll fill A[row_index, k*N_phi + j].
            for k in range(N_theta):
                theta_k = thetas[k]
                # The factor w[k] * dphi is the quadrature weight
                weight_k = w[k] * dphi
                for j in range(N_phi):
                    phi_j = phis[j]
                    Y_lm_conj = np.conjugate(sph_harm(m, ell, phi_j, theta_k))
                    col_index = k*N_phi + j
                    A[row_index, col_index] = weight_k * Y_lm_conj

            row_index += 1

    return A



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate the visual embedding")
    parser.add_argument("config", type=str, help="Path to the config JSON file")
    parser.add_argument("--dry_run", action="store_true", help="Flag to plot the rendered images")
    parser.add_argument("--model_fn", type=str, default=None, help="Model filename")
    args = parser.parse_args()
    
    config = Config(args.config)
    device = "cuda"

    W, H = 224, 224
    L = 3
    N_THETA = L + 1
    N_PHI = 2 * N_THETA
    print(f"L={L}, N_THETA={N_THETA}, N_PHI={N_PHI}")

    x, weights = leggauss(N_THETA)  # x_k, w_k for k=0..N_theta-1
    thetas = np.arccos(x)  # in (0, pi)
    phis = np.linspace(0, 2*np.pi, N_PHI, endpoint=False)

    A_mat = torch.tensor(build_A(L, weights, thetas, phis)).to(device, torch.complex64)
    print("A", A_mat.shape)
    
    radius = 1.0
    fov = 7.5 # degrees
    f = 0.5 * W / np.tan(fov / 360 * np.pi)
    
    K = np.array([[f, 0.0, W/2],[0, f, H/2],[0, 0, 1]])
    Ks = np.array([K for _ in range(N_THETA * N_PHI)])
    Ks = torch.tensor(Ks).to(device, torch.float32)

    viewmats = np.zeros((N_THETA, N_PHI, 4, 4))
    for i in range(len(thetas)):
        for j in range(len(phis)):
            viewmats[i,j] = camera_extrinsic_spherical(radius, thetas[i], phis[j])
    viewmats = torch.tensor(viewmats).to(device, torch.float32).view(-1, 4, 4)

    intrinsic, extrinsic, Ps = get_cam_params(
        config.camera_fn,
        ds=config.image_downsample,
        up_fn=config.vertical_lines_fn,
        auto_orient=True,
        load_up_direction=not config.adaptive_camera,
    )
    C = len(Ps)

    img_fn = os.path.join(config.image_directory, "images.h5")
    volume_fn = os.path.join(config.volume_directory, "volumes.h5")
    dset_args = (img_fn, volume_fn, config.center_rotation_fn, C)
    num_workers = len(os.sched_getaffinity(0)) # available CPUs
    dset = FrameDataset(*dset_args, split="all_volumes")

    model = PoseSplatter(
        intrinsics=intrinsic,
        extrinsics=extrinsic,
        W=None,
        H=None,
        ell=config.ell,
        grid_size=config.grid_size,
        volume_idx=config.volume_idx,
        adaptive_camera=config.adaptive_camera,
    )
    model.to(device)
    if args.model_fn is None:
        model_fn = config.model_fn
    else:
        model_fn = args.model_fn
    model.load_state_dict(torch.load(model_fn)["model_state_dict"])
    model.eval()

    feature_extractor = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    feature_extractor = nn.Sequential(*list(feature_extractor.children())[:-1]).to(device)
    feature_extractor.eval()

    preprocess = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    all_features = []
    if args.dry_run:
        loader = DataLoader(dset, batch_size=1, shuffle=False, num_workers=num_workers)
        pbar = enumerate(loader)
    else:
        loader = DataLoader(dset, batch_size=1, shuffle=False, num_workers=num_workers)
        pbar = tqdm(enumerate(loader), total=len(dset))
    for batch_num, (mask, img, p_3d, angle, _) in pbar:

        with torch.no_grad():

            # Make the volume.
            if config.adaptive_camera:
                volume, _ = model.carver(mask[0,:,None].to(device), img[0].to(device), p_3d.to(device), angle, adaptive=config.adaptive_camera)
            else:
                volume = model.carver(mask[0,:,None].to(device), img[0].to(device), p_3d.to(device), angle, adaptive=config.adaptive_camera)
            # # Make the volume.
            # volume = model.carver(mask[0,:,None].to(device), img[0].to(device), p_3d.to(device), angle)

            # Run the volume through the U-Nets.
            volume = model.process_volume(volume[None])

            # Get Gaussian parameters.
            means, quats, scales, opacities, colors = model.get_gaussian_params_from_volume(volume)

            # Center the animal.
            means = means - torch.mean(means, dim=0, keepdim=True)

            # Rotate by a random angle.
            if args.dry_run:
                theta = 0.0
            else:
                theta = 2 * np.pi * np.random.rand()
            c, s = np.cos(theta), np.sin(theta)
            mat = torch.tensor([[c,-s,0], [s,c,0], [0,0,1]]).to(device, torch.float32)
            means = means @ mat.T

            # Splat.
            rgb, _ = model.splat(
                means,
                quats,
                scales,
                opacities,
                colors,
                viewmats,
                Ks,
                W,
                H,
            ) # [b,H,W,3]
            rgb = rgb.clamp(0,1)

            if args.dry_run:
                # Plot.
                rgb = rgb.detach().cpu().numpy().reshape(N_THETA, N_PHI, H, W, 3)
                import matplotlib.pyplot as plt
                _, axarr = plt.subplots(nrows=N_THETA, ncols=N_PHI)
                for i in range(N_THETA):
                    for j in range(N_PHI):
                        plt.sca(axarr[i,j])
                        plt.xticks([])
                        plt.yticks([])
                        plt.imshow(rgb[i,j])
                plt.tight_layout()
                out_fn = os.path.join(config.project_directory, "visual_feature_example.pdf")
                plt.savefig(out_fn)
                print("Saved:", out_fn)
                quit()

            # Run rendered images through the feature extractor.
            rgb = torch.permute(rgb, (0,3,1,2)) # [b,3,H,W]
            features = feature_extractor(preprocess(rgb)) # [b,512,1,1]
            features = features.squeeze() # [b,512]
            features = torch.einsum("ij,jx->ix", A_mat, features.to(torch.complex64))
            features = torch.abs(features) # [harmonics,512]
            all_features.append(features.detach().cpu().numpy().astype(np.float16))

    # Save the features.
    out_fn = config.feature_fn
    np.save(out_fn, np.array(all_features))
    print("Saved features to:", out_fn)

    del dset # closes HDF5 files

