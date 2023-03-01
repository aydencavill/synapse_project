import neuroglancer
import numpy as np
import os
import sys
import webbrowser
import zarr

neuroglancer.set_server_bind_address('localhost')

f = zarr.open(sys.argv[1])

try:
    section = sys.argv[2]
except: 
    section = None

datasets = [i for i in os.listdir(sys.argv[1]) if '.' not in i]

if section:
    datasets = [x+f'/{section}' for x in datasets]

res = f[datasets[0]].attrs['resolution']

viewer = neuroglancer.Viewer()

dims = neuroglancer.CoordinateSpace(
        names=['b','c^','y','x'],
        units='nm',
        scales=res+res)

with viewer.txn() as s:

    for ds in datasets:

        offset = f[ds].attrs['offset']

        offset = [0,]*2 + [int(i/j) for i,j in zip(offset, res)]

        data = f[ds][:]

        shader="""
void main() {
    emitRGB(
        vec3(
            toNormalized(getDataValue(0)),
            toNormalized(getDataValue(1)),
            toNormalized(getDataValue()))
        );
}"""

        try:
            
            if 'label' and 'seg' not in ds:
                s.layers[ds] = neuroglancer.ImageLayer(
                    source=neuroglancer.LocalVolume(
                        data=data,
                        voxel_offset=offset,
                        dimensions=dims),
                    shader=shader)
            else:
                s.layers[ds] = neuroglancer.SegmentationLayer(
                    source=neuroglancer.LocalVolume(
                        data=np.expand_dims(data,axis=1),
                        voxel_offset=offset,
                        dimensions=dims))

        except Exception as e:
            print(ds, e)

    s.layout = 'yz' 

print(viewer)
