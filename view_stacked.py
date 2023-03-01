import neuroglancer
import numpy as np
import os
import sys
import zarr
import re


def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


neuroglancer.set_server_bind_address('localhost')

f = zarr.open(sys.argv[1]) #input_zarr

datasets = [i for i in os.listdir(sys.argv[1]) if '.' not in i]
sections = [i for i in os.listdir(os.path.join(sys.argv[1],datasets[0])) if '.' not in i]

print(datasets)

resolution = f[datasets[0]][sections[0]].attrs['resolution']

viewer = neuroglancer.Viewer()

dims = neuroglancer.CoordinateSpace(
        names=['z','y','x'],
        units='nm',
        scales=[50,*resolution])

with viewer.txn() as s:

    for ds in datasets:

        sections = natural_sort(
                [i for i in os.listdir(os.path.join(sys.argv[1],ds)) if '.' not in i]
                )

        data = np.stack([f[ds][section][:] for section in sections])
        
        offset = f[ds][sections[0]].attrs['offset']
        offset = [sections[0]]+[i/vs for i,vs in zip(offset,resolution)]

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

    s.layout = 'yz'

print(viewer)

