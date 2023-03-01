import neuroglancer
import numpy as np
import os
import sys
import webbrowser
import zarr


neuroglancer.set_server_bind_address('127.0.0.1')

f = zarr.open(sys.argv[1]) #input zarr

datasets = [i for i in os.listdir(sys.argv[1]) if '.' not in i]

res = f[datasets[0]].attrs['resolution']

viewer = neuroglancer.Viewer()

dims = neuroglancer.CoordinateSpace(
        names=['z','y','x'],
        units=['nm','nm','nm'],
        scales=res)

with viewer.txn() as s:

    for ds in datasets:

        data = f[ds]
       
        offset = f[ds].attrs['offset']
        offset = [int(i/j) for i,j in zip(offset, res)]

        volume = neuroglancer.LocalVolume(
                data=data,
                voxel_offset=[*offset],
                dimensions=dims)

        try:    
            if 'label' not in ds and 'seg' not in ds and 'painted' not in ds:
                s.layers[ds] = neuroglancer.ImageLayer(
                    source=volume)

            else:
                s.layers[ds] = neuroglancer.SegmentationLayer(
                    source=volume)

        except Exception as e:
            print(ds, e)

print(viewer)
