"""
Data utilities

"""
__date__ = "November 2024 - March 2025"

import numpy as np
import torch
from torch.utils.data import Dataset
import zarr
import os



class FrameDataset(Dataset):
    def __init__(
            self,
            img_fn,
            volume_fn,
            angle_fn,
            C,
            holdout_views=[],
            split="train",
        ):
        assert split in ["train", "valid", "test", "all", "all_volumes"]

        # TODO: switch fully to Zarr!
        zarr_fn = img_fn[:-3] + ".zarr"
        if not os.path.exists(zarr_fn):
            print("Zarr file does not exist:", zarr_fn)
            quit()
        self.images = zarr.open(zarr_fn, 'r')['images']
        self.observed_views = np.array([i for i in range(C) if i not in holdout_views], dtype=int)

        # Figure out split indices.
        a1, a2 = 0, len(self.images) // 3
        a3, a4 = 2 * a2, len(self.images)
        if split == "train":
            self.i1, self.i2 = a1, a2
        elif split == "valid":
            self.i1, self.i2 = a2, a3
        elif split == "test":
            self.i1, self.i2 = a3, a4
        else: # split == "all" or split == "all_volumes"
            self.i1, self.i2 = a1, a4

        self.C = C
        self.split = split
        d = np.load(angle_fn)
        self.angles = d["angles"]
        self.centers = d["centers"]

    def __len__(self):
        if self.split == "all":
            return (self.i2 - self.i1) * self.C
        return self.i2 - self.i1

    def __getitem__(self, idx, view_idx=None, angle_offset=0.0, center_offset=0.0):
        if self.split == "all":
            view_idx = idx % self.C
            idx = idx // self.C
        idx += self.i1
        if view_idx is None:
            view_idx = np.random.choice(self.observed_views)
        
        img = torch.tensor(self.images[idx], dtype=torch.float32) / 255.0 # [C,H,W,3]
        mask = torch.where(img[..., 0] == 1.0, 0.0, 1.0) # [C,H,W]

        img = img[torch.tensor(self.observed_views)]
        mask = mask[torch.tensor(self.observed_views)]

        p_3d = self.centers[idx] + center_offset
        angle = self.angles[idx] + angle_offset
        p_3d = torch.tensor(p_3d).to(torch.float32)

        return mask, torch.permute(img, (0,3,1,2)), p_3d, angle, view_idx
        