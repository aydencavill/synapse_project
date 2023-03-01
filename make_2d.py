import numpy as np
import daisy
import sys
import os


def write_section(args):

    data = args['data']
    index = args['index']

    section_number = int(roi.offset[0]/vs[0] + index)
    
    if np.any(data):

        print(f"at section {section_number}..")

        new_ds = daisy.prepare_ds(
                output_zarr,
                f"{dataset}/{section_number}",
                write_roi,
                write_vs,
                dtype)

        new_ds[write_roi] = data

    else:
        print(f"section {section_number} is empty, skipping")
        pass


if __name__ == "__main__":

    input_zarr = sys.argv[1]
    output_zarr = "2d_" + input_zarr
    print(output_zarr)

    datasets = [i for i in os.listdir(sys.argv[1]) if '.' not in i]
    print(datasets)

    for dataset in datasets:

        print(dataset)
        ds = daisy.open_ds(input_zarr,dataset)

        data = ds.to_ndarray()

        roi = ds.roi
        vs = ds.voxel_size
        dtype = ds.dtype
        print('\n\nRoi: ', roi, '\n\n\n')
 
        write_roi = daisy.Roi(roi.offset[1:],roi.shape[1:])
        write_vs = vs[1:]

        args = ({
                'index' : index,
                'data' : section} for index,section in enumerate(data))

        for arg in args:
            write_section(arg)
