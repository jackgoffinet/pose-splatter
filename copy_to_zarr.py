"""
Copy an HDF5 file (or group) into a Zarr store.

Examples
--------
# Basic usage (defaults to LZ4 compression)
python copy_to_zarr.py input.h5 output.zarr

# Overwrite an existing Zarr store
python copy_to_zarr.py input.h5 output.zarr --overwrite

# Choose compressor
python copy_to_zarr.py input.h5 output.zarr --compressor zstd --clevel 7 --shuffle bit
"""

import argparse
import sys

import h5py
import zarr

try:
    from numcodecs import Blosc
except Exception:
    Blosc = None


def make_compressor(name: str, clevel: int, shuffle: str):
    """
    Build a numcodecs compressor from args.
    Returns None if name == 'none'.
    """
    name = name.lower()
    shuffle = shuffle.lower()
    if name == "none":
        return None
    if Blosc is None:
        print(
            "[warn] numcodecs.Blosc not available; proceeding with no compression.",
            file=sys.stderr,
        )
        return None

    cname = {"lz4": "lz4", "zstd": "zstd", "zlib": "zlib"}.get(name)
    if cname is None:
        raise ValueError(f"Unsupported compressor: {name} (choose: none, lz4, zstd, zlib)")

    shuffle_map = {"none": Blosc.NOSHUFFLE, "byte": Blosc.SHUFFLE, "bit": Blosc.BITSHUFFLE}
    if shuffle not in shuffle_map:
        raise ValueError("shuffle must be one of: none, byte, bit")

    return Blosc(cname=cname, clevel=int(clevel), shuffle=shuffle_map[shuffle])


def copy_attrs(src, dst):
    """Copy HDF5 attributes to Zarr object."""
    for k, v in src.attrs.items():
        try:
            dst.attrs[k] = v
        except Exception as e:
            # Attribute types can occasionally be problematic; skip with a note.
            print(f"[warn] Skipping attribute {k!r}: {e}", file=sys.stderr)


def copy_h5_to_zarr(h5_group: h5py.Group, zarr_group: zarr.Group, compressor=None):
    """
    Recursively copy data from an HDF5 group to a Zarr group.
    """
    # Copy group-level attributes
    copy_attrs(h5_group, zarr_group)

    for key in h5_group:
        item = h5_group[key]
        if isinstance(item, h5py.Group):
            # Create corresponding group in Zarr and recurse
            zarr_subgroup = zarr_group.create_group(key)
            copy_h5_to_zarr(item, zarr_subgroup, compressor=compressor)

        elif isinstance(item, h5py.Dataset):
            # Create dataset in Zarr. Reuse HDF5 chunking if present.
            chunks = item.chunks  # may be None
            # Read data; for very large datasets this loads into memory.
            data = item[()]
            zds = zarr_group.create_dataset(
                name=key,
                data=data,
                chunks=chunks,
                compressor=compressor,
            )
            # Copy dataset attributes
            copy_attrs(item, zds)

        else:
            print(f"[warn] Unknown item type: {key} ({type(item)})", file=sys.stderr)


def parse_args():
    p = argparse.ArgumentParser(
        description="Recursively copy an HDF5 file into a Zarr store."
    )
    p.add_argument("input_h5", help="Path to input HDF5 file")
    p.add_argument("output_zarr", help="Path to output Zarr store (directory or .zarr)")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output Zarr store if it exists",
    )
    p.add_argument(
        "--compressor",
        default="lz4",
        choices=["none", "lz4", "zstd", "zlib"],
        help="Compressor to use for Zarr datasets (default: lz4)",
    )
    p.add_argument(
        "--clevel",
        type=int,
        default=5,
        help="Compression level for Blosc compressors (default: 5)",
    )
    p.add_argument(
        "--shuffle",
        default="bit",
        choices=["none", "byte", "bit"],
        help="Shuffle mode for Blosc (default: bit)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    compressor = make_compressor(args.compressor, args.clevel, args.shuffle)

    mode = "w" if args.overwrite else "x"
    try:
        with h5py.File(args.input_h5, "r") as h5_file:
            zarr_store = zarr.open(args.output_zarr, mode=mode)
            copy_h5_to_zarr(h5_file, zarr_store, compressor=compressor)
    except FileExistsError:
        print(
            f"[error] Output Zarr store already exists: {args.output_zarr}. "
            f"Use --overwrite to replace.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
