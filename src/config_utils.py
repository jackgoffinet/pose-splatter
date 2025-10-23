"""
Simple class for handling config files

"""
__date__ = "January 2025"


import json
import os

ATTRIBUTES = [
    "data_directory",
    "project_directory",
    "mask_video_fns",
    "video_fns",
    "holdout_views",
    "volume_directory",
    "image_directory",
    "render_directory",
    "image_compression_level",
    "volume_compression_level",
    "camera_fn",
    "vertical_lines_fn",
    "center_rotation_fn",
    "volume_sum_fn",
    "model_fn",
    "feature_fn",
    "embedding_fn",
    "image_width",
    "image_height",
    "image_downsample",
    "adaptive_camera",
    "fps",
    "train_time",
    "valid_time",
    "ell",
    "ell_tracking",
    "grid_size",
    "frame_jump",
    "volume_idx",
    "volume_fill_color",
    "img_lambda",
    "ssim_lambda",
    "lr",
    "valid_every",
    "plot_every",
    "save_every",
]

DATA_LIST_ATTRIBUTES = ["mask_video_fns", "video_fns"]
DATA_ATTRIBUTES = []
PROJECT_ATTRIBUTES = [
    "volume_directory",
    "image_directory",
    "render_directory",
    "camera_fn",
    "vertical_lines_fn",
    "center_rotation_fn",
    "volume_sum_fn",
    "model_fn",
    "feature_fn",
    "embedding_fn",
]


class Config:
    def __init__(self, json_file):
        """Initialize the class by reading a JSON configuration file."""
        with open(json_file, 'r') as file:
            self._data = json.load(file)
    
    def __getattr__(self, name):
        """Override for specific attribute behavior and general access."""
        if name in DATA_LIST_ATTRIBUTES:
            data_dir = self._data.get("data_directory", "")
            if name in self._data:
                return [os.path.join(data_dir, i) for i in self._data[name]]
        
        elif name in DATA_ATTRIBUTES:
            data_dir = self._data.get("data_directory", "")
            if "volume_directory" in self._data:
                return os.path.join(data_dir, self._data[name])
        
        elif name in PROJECT_ATTRIBUTES:
            proj_dir = self._data.get("project_directory", "")
            if name in self._data:
                return os.path.join(proj_dir, self._data[name])
        
        # General handling for other attributes
        if name in self._data:
            return self._data[name]
        
        raise AttributeError(f"'Config' object has no attribute '{name}'")
    
    def to_serializable(self):
        """Return a serializable dictionary with custom attribute logic."""
        result = {}
        for attr in ATTRIBUTES:
            try:
                result[attr] = getattr(self, attr)
            except AttributeError:
                result[attr] = None  # Handle missing attributes gracefully
        return result
