"""
Collect vertical lines for each camera view to estimate a good up direction.

"""
__date__ = "December 2024 - February 2025"

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys

from src.config_utils import Config
from src.utils import get_cam_params


USAGE = "Usage:\n$ python estimate_up_direction.py <config_file.json>"



def collect_lines(images):
    """
    Allows a user to interactively collect line segments for a list of images,
    one image at a time.

    Parameters
    ----------
    images : list of np.ndarray
        Each element is an image (grayscale or color), to be shown one at a time.

    Returns
    -------
    lines_all : list of length N
        lines_all[i] is a list of line segments for the i-th image.
        Each segment is [[x1, y1], [x2, y2]] in figure coordinate space.
    """
    n_images = len(images)
    img_nums = []
    lines_all = []

    for i, img in enumerate(images):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Image {i+1}/{n_images}")
        ax.axis("off")


        current_line_start = [None]  # store in a list so we can modify inside handler

        def on_click(event):
            # We only care about left-click (1)
            if event.button not in [1]:
                return
            if event.inaxes != ax:
                return

            if current_line_start[0] is None:
                # Start a new line
                current_line_start[0] = (event.xdata, event.ydata)
            else:
                # Complete the line
                x1, y1 = current_line_start[0]
                x2, y2 = event.xdata, event.ydata
                current_line_start[0] = None

                img_nums.append(i)
                lines_all.append([[x1, y1], [x2, y2]])

                ax.plot([x1, x2], [y1, y2], 'r-')
                fig.canvas.draw()

        # Connect the callback
        cid = fig.canvas.mpl_connect('button_press_event', on_click)

        print(f"Collect lines (vertical=left-click, horizontal=right-click) for Image {i+1}.")
        print("Close this figure window when you are done with this image.")

        # Show this image alone until the user closes the window
        plt.show()

        # Disconnect the event callback
        fig.canvas.mpl_disconnect(cid)
        plt.close(fig)

    return img_nums, lines_all



def reproject_corrected_lines(
        K,
        E,
        lines_all,
        best_dir_world,
        image_list=None,
        depth_guess=1.0,
        line_length=0.1,
    ):
    """
    For each image:
      - Plot the original lines in red.
      - Construct 'corrected' lines that have the same average direction in world coords,
        anchor them by un-projecting the original midpoint at a fixed depth,
        and re-project them back to the image in green.

    Parameters
    ----------
    K : ndarray of shape (N, 3, 3)
        Intrinsic matrices for N cameras.
    E : ndarray of shape (N, 4, 4)
        Extrinsic matrices for N cameras, mapping world->camera.
    lines_all : list of length N
        lines_all[i] = array of shape (M_i, 2, 2), each line seg is [[u1,v1],[u2,v2]].
    best_dir_world : ndarray of shape (3,)
        The computed unit direction in world coordinates to enforce.
    image_list : list of length N or None
        If provided, each entry is an image (H x W x 3 or H x W),
        to display as a background in each subplot. Otherwise we use a placeholder.
    depth_guess : float
        The depth (in front of each camera) at which we place the midpoint of the corrected line.
    line_length : float
        The length (in world units) of the corrected 3D line to project back.
    """
    N = len(K)
    fig, axes = plt.subplots(1, N, figsize=(5*N, 5))
    if N == 1:
        axes = [axes]

    for i in range(N):
        # --- Prepare the subplot ---
        ax = axes[i]
        if image_list is not None and image_list[i] is not None:
            ax.imshow(image_list[i], cmap='gray')
        else:
            # placeholder background if no image is provided
            ax.set_facecolor('lightgray')

        ax.set_title(f"Camera {i+1}")
        ax.axis("off")

        K_i = K[i]
        E_i = E[i]
        R_i = E_i[:3, :3]  # world->cam
        t_i = E_i[:3, 3]   # world->cam

        K_inv = np.linalg.inv(K_i)

        # Plot each line
        for seg in lines_all[i]:
            (u1, v1), (u2, v2) = seg

            # 1) Plot the original line in red
            ax.plot([u1, u2], [v1, v2], 'r-')

            # 2) Construct the "corrected" line in 3D
            #    a) Find midpoint in image coords
            um = 0.5*(u1 + u2)
            vm = 0.5*(v1 + v2)

            #    b) Un-project midpoint at guessed depth (in camera coords)
            #       We'll do this by:
            #           (x_cam, y_cam, z_cam) = depth_guess * K_inv @ [u, v, 1]
            #       Because K_inv @ [u, v, 1] is direction in camera coords for that pixel.
            uv1 = np.array([um, vm, 1.0])
            m_cam_dir = K_inv @ uv1
            # scale up to the chosen depth:
            # for a pinhole camera, the z_cam will become 'depth_guess'
            # so factor = depth_guess / m_cam_dir[2]
            factor = depth_guess / m_cam_dir[2]
            m_cam = factor * m_cam_dir  # 3D midpoint in camera coords

            #    c) Convert that midpoint to world coords
            #       X_cam = R_i * X_world + t_i => X_world = R_i^T (X_cam - t_i)
            m_world = R_i.T @ (m_cam - t_i)

            #    d) Build a 3D segment about that midpoint, with length 'line_length'
            #       and direction 'best_dir_world'.
            half_len = 0.5*line_length
            p1_world = m_world - half_len * best_dir_world
            p2_world = m_world + half_len * best_dir_world

            # 3) Re-project these corrected endpoints into the image
            #    pX_cam = R_i * pX_world + t_i
            p1_cam = R_i @ p1_world + t_i
            p2_cam = R_i @ p2_world + t_i

            #    Then apply intrinsics
            #    [u, v, 1]^T = K_i * [x_cam/z_cam, y_cam/z_cam, 1]^T
            def project_point(cam_point, K_):
                x, y, z = cam_point
                if z <= 1e-6:
                    # Just skip or clamp if the point is behind the camera or extremely close to zero
                    return None
                u = K_[0, 0]*(x/z) + K_[0, 2]
                v = K_[1, 1]*(y/z) + K_[1, 2]
                return (u, v)

            p1_img = project_point(p1_cam, K_i)
            p2_img = project_point(p2_cam, K_i)

            if p1_img is not None and p2_img is not None:
                ax.plot([p1_img[0], p2_img[0]], [p1_img[1], p2_img[1]], 'g-')

    plt.tight_layout()
    plt.show()



def estimate_world_up_subspaces(K, E, lines_all):
    """
    Estimate the 'up' direction in world coordinates by:
      (1) For each camera, find a single 2D subspace (plane) in world space
          that best fits the 'vertical' lines in that camera's image.
      (2) Combine these camera planes to find the single 1D 'up' vector
          that is consistent with all planes.

    Parameters
    ----------
    K : ndarray of shape (C, 3, 3)
        Intrinsic matrices for the C cameras.
    E : ndarray of shape (C, 4, 4)
        Extrinsic matrices (world->camera). E[i] = [[R_i, t_i],
                                                   [   0,     1]]
        R_i is 3x3, t_i is 3x1.
    lines_all : list of length C
        lines_all[i] is an array of shape (N_i, 2, 2), storing the pixel
        endpoints of line segments in camera i. Each segment = [[u1,v1],[u2,v2]].

    Returns
    -------
    up_world : ndarray of shape (3,)
        A unit vector in world coordinates representing the 'up' direction.

    Reference geometry:
      - Each 2D line in camera i => plane in 3D passing through camera center
        => that plane's normal in camera coords is n_cam = K_i^T * line_abc,
        => in world coords: n_world = R_i^T * n_cam.
      - We average these normals per camera to define the best-fit plane
        for that camera. Finally, we combine all camera planes to find
        the single direction that is inside all planes => up.
    """

    C = len(K)
    # We'll store one plane normal n_i per camera
    plane_normals_per_camera = []

    for i in range(C):
        K_i = K[i]
        E_i = E[i]
        R_i = E_i[:3, :3]  # rotation world->cam

        # lines for this camera
        lines_i = lines_all[i]

        # We'll accumulate plane normals from each vertical line
        normals_list = []

        for seg in lines_i:

            # Convert the 2D segment => line (a,b,c)
            (u1,v1), (u2,v2) = seg
            # Homogeneous line: a*u + b*v + c=0
            # Using standard formula from 2 points:
            a = (v1 - v2)
            b = (u2 - u1)
            c = (u1*v2 - u2*v1)
            line_abc = np.array([a, b, c], dtype=float)

            # Plane normal in camera coords: n_cam = K^T * line_abc
            # However, note we typically do n_cam = (K^-1)^T * line_abc
            # Actually, standard pinhole geometry for a line l=(a,b,c) => plane normal is n_cam = K^T l if
            # the line equation is normalized. Let's do the straightforward approach:
            n_cam = K_i.T @ line_abc  # shape (3,)

            # Transform to world coords: n_world = R^T * n_cam
            n_world = R_i.T @ n_cam

            # Normalize
            norm_val = np.linalg.norm(n_world)
            if norm_val > 1e-12:
                n_world /= norm_val
                normals_list.append(n_world)

        if len(normals_list) == 0:
            # If no vertical lines for this camera, skip or produce a dummy normal
            # We'll skip storing anything for now.
            continue

        # Sum or average the normals to get a single plane normal for camera i
        normals_arr = np.array(normals_list)
        n_i = normals_arr.sum(axis=0)
        n_i /= np.linalg.norm(n_i)
        plane_normals_per_camera.append(n_i)

    if len(plane_normals_per_camera) == 0:
        raise ValueError("No vertical lines found in any camera; cannot estimate up.")

    # 2) Now we have one plane normal per camera => n_i
    #    The 'up' direction up_world should be orthonormal to each n_i.
    #    Let M = sum_i n_i n_i^T. We find the eigenvector of M with smallest eigenvalue.

    M = np.zeros((3,3), dtype=float)
    for n_i in plane_normals_per_camera:
        M += np.outer(n_i, n_i)

    # Eigendecomposition of M
    vals, vecs = np.linalg.eig(M)
    # We want the eigenvector with the smallest eigenvalue
    idx_min = np.argmin(vals)
    up_world = vecs[:, idx_min]
    up_world /= np.linalg.norm(up_world)

    return up_world



if __name__ == "__main__":
    assert len(sys.argv) == 2, USAGE
    config = Config(sys.argv[1])
    video_fns = config.video_fns
    num_images = len(video_fns)
    out_fn = config.vertical_lines_fn
    if out_fn.endswith(".npy"):
        out_fn = out_fn[:-4] + ".npz"

    imgs = []
    for video_fn in video_fns:
        cap = cv2.VideoCapture(video_fn)
        ret, frame = cap.read()
        assert ret
        imgs.append(frame[...,::-1])
        cap.release()

    print("Draw the some number of vertical lines in each image.")
    print("Click the bottom point, then the top, for each line.")
    img_nums, lines = collect_lines(imgs)

    collected_lines = [[] for _ in range(len(imgs))]
    for img_num, line in zip(img_nums, lines):
        collected_lines[img_num].append(line)

    K, E, _ = get_cam_params(
        config.camera_fn,
        ds=1,
        auto_orient=True,
        load_up_direction=False,
    )

    # 1) Estimate the best-fit direction in world coords
    best_dir_world = estimate_world_up_subspaces(K, E, collected_lines)
    print("Best direction in world coords:", best_dir_world)

    # Save lines and the up direction:
    np.savez(out_fn, img_nums=img_nums, lines=lines, up=best_dir_world)
    print("Saved:", out_fn)

    # 2) Re-project "corrected" lines that are parallel in world coords
    reproject_corrected_lines(
        K,
        E,
        collected_lines,
        best_dir_world,
        image_list=imgs,
        depth_guess=2.0,
        line_length=1.0,
    )  
