"""
Write masked images in HDF5 files.

"""
__date__ = "December 2024 - January 2025"

import cv2
import h5py
from joblib import Parallel, delayed
import numpy as np
import os
import sys

from src.config_utils import Config


USAGE = "Usage:\n$ python write_images.py <config.json>"
WHITE = 255 * np.ones(3).astype(np.uint8)



def process_chunk_write_images(config, chunk_num, frame_list):
    mask_caps = [cv2.VideoCapture(mask_video_fn) for mask_video_fn in config["mask_video_fns"]]
    video_caps = [cv2.VideoCapture(video_fn) for video_fn in config["video_fns"]]
    for cap in mask_caps + video_caps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_list[0])

    ds = config["image_downsample"]
    C = len(config["video_fns"])

    error_flag = False
    jump = config["frame_jump"]
    h = config["image_height"] // config["image_downsample"]
    w = config["image_width"] // config["image_downsample"]

    num_samples = len(frame_list)
    if chunk_num == 0:
        print("num_samples", num_samples)

    h5_filename = os.path.join(config["image_directory"], f"images_{chunk_num:04d}.h5")
    with h5py.File(h5_filename, "w") as hdf:
            
        # Create a dataset with gzip compression (level 4)
        images_dataset = hdf.create_dataset(
            "images", 
            (num_samples, C, h, w, 3),
            dtype='uint8', 
            compression="gzip", # gzip
            compression_opts=config["image_compression_level"] # (0-9 for gzip)
        )
        
        for frame_num, frame_idx in enumerate(frame_list):
            if chunk_num == 0:
                print(frame_num)
            # Get the masks from the videos.
            masks = []
            frames = []
            for cap_num, (mask_cap, video_cap) in enumerate(zip(mask_caps, video_caps)):
                ret, frame = mask_cap.read()
                if not ret:
                    print(f"Error reading mask frame {frame_idx}, video {cap_num}")
                    error_flag = True
                    break
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    masks.append(frame)

                for _ in range(jump - 1):
                    mask_cap.read()

                ret, frame = video_cap.read()
                if not ret:
                    print(f"Error reading video frame {frame_idx}, video {cap_num}")
                    error_flag = True
                    break
                else:
                    frames.append(frame[...,::-1])

                for _ in range(jump - 1):
                    video_cap.read()
            if error_flag:
                break

            masks = np.array(masks)
            frames = np.array(frames)
            if ds != 1:
                masks = masks[:,::ds][:,:,::ds]
                frames = frames[:,::ds][:,:,::ds]

            frames[masks < 128] = WHITE
            images_dataset[frame_num] = frames

    for cap in mask_caps + video_caps:
        cap.release()


def concatenate_h5_files(input_files, output_file, dataset_name="images", compression_level=2):
    """
    Concatenates multiple HDF5 files containing a single dataset into a new HDF5 file.
    
    Parameters:
        input_files (list of str): List of input HDF5 file paths.
        output_file (str): Path of the output HDF5 file.
        dataset_name (str): Name of the dataset to concatenate (default: "images").
    """
    # Step 1: Determine total number of samples and dataset shape
    total_samples = 0
    dataset_shape = None
    dtype = None

    for file_path in input_files:
        with h5py.File(file_path, "r") as hdf:
            dataset = hdf[dataset_name]
            total_samples += dataset.shape[0]
            if dataset_shape is None:
                dataset_shape = dataset.shape[1:]  # Shape of a single sample
                dtype = dataset.dtype
            elif dataset.shape[1:] != dataset_shape:
                raise ValueError(f"Dataset shape mismatch in file {file_path}")

    # Step 2: Create the output HDF5 file and dataset
    with h5py.File(output_file, "w") as hdf_out:
        # Create a new dataset with total size
        output_dataset = hdf_out.create_dataset(
            dataset_name, 
            shape=(total_samples,) + dataset_shape, 
            dtype=dtype,
            compression="gzip",
            compression_opts=compression_level,
        )

        # Step 3: Copy data from each input file to the output dataset
        current_index = 0
        for file_path in input_files:
            with h5py.File(file_path, "r") as hdf:
                dataset = hdf[dataset_name]
                num_samples = dataset.shape[0]
                output_dataset[current_index:current_index + num_samples] = dataset[:]
                current_index += num_samples

    print(f"Concatenation complete. Output file saved to: {output_file}")



if __name__ == '__main__':
    assert len(sys.argv) == 2, USAGE
    config = Config(sys.argv[1])
    N_JOBS = len(os.sched_getaffinity(0)) # available CPUs

    if not os.path.exists(config.image_directory):
        os.makedirs(config.image_directory)

    cap = cv2.VideoCapture(config.mask_video_fns[0])
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("total_frames:", total_frames)
    cap.release()

    frame_lists = np.array_split(np.arange(0, total_frames, config.frame_jump), N_JOBS)

    # The config needs to be serializable to parallelize.
    s_config = config.to_serializable()

    # Use joblib for parallelize processing
    print("Creating files...")
    Parallel(n_jobs=N_JOBS)(
        delayed(process_chunk_write_images)(s_config, i, frame_list) for i, frame_list in enumerate(frame_lists)
    )

    # Concatenate the files.
    print("Concatenating files...")
    input_files = [os.path.join(config.image_directory, f"images_{i:04d}.h5") for i in range(N_JOBS)]
    output_file = os.path.join(config.image_directory, "images.h5")
    
    concatenate_h5_files(
        input_files,
        output_file,
        compression_level=config.image_compression_level,
    )

    # Clean up the temporary files.
    for input_file in input_files:
        os.remove(input_file)


  ###