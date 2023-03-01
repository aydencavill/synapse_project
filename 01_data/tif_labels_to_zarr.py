from tifffile import imread
import numpy as np
import zarr
import sys
import os


""" Makes zarr dataset from napari tif labels. """


if __name__ == "__main__":

    labels_tiff = sys.argv[1] # path to tif file
    out_zarr = sys.argv[2] # path to zarr container

    ds_name = os.path.basename(labels_tiff).split(".")[0] # get name of dataset from tif file

    print(f"writing {ds_name} to {out_zarr}")

    f = zarr.open(out_zarr,"a") 

    # get dataset resolution and offset from raw dataset
    res = f["raw"].attrs["resolution"]
    offset = f["raw"].attrs["offset"]

    # read tif file into numpy array
    labels = imread(labels_tiff)

    print(f"{labels_tiff} has shape {labels.shape}")

    # write to zarr
    f[ds_name] = labels.astype(np.uint8)
    f[ds_name].attrs["offset"] = offset
    f[ds_name].attrs["resolution"] = res
