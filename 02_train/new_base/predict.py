import json
import gunpowder as gp
import math
import numpy as np
import os
import sys
import torch
import logging
import zarr
import daisy

from model import UNet


setup_dir = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(setup_dir, 'config.json'), 'r') as f:
    config = json.load(f)

# voxels
pad = gp.Coordinate([104,104])
input_shape = gp.Coordinate(tuple(config['input_shape'])) + pad
output_shape = gp.Coordinate(tuple(config['output_shape'])) + pad

# nm
voxel_size = gp.Coordinate(tuple(config['voxel_size']))
input_size = input_shape * voxel_size
output_size = output_shape * voxel_size
context = (input_size - output_size) / 2


def predict(
        iteration,
        raw_file,
        raw_dataset,
        out_file,
        out_dataset):

    raw = gp.ArrayKey('RAW')
    pred_labels = gp.ArrayKey('PRED_LABELS')

    scan_request = gp.BatchRequest()

    scan_request.add(raw, input_size)
    scan_request.add(pred_labels, output_size)

    source = gp.ZarrSource(
                raw_file,
            {
                raw: raw_dataset
            },
            {
                raw: gp.ArraySpec(interpolatable=True)
            })

    with gp.build(source):
        total_output_roi = source.spec[raw].roi
        total_input_roi = source.spec[raw].roi.grow(context,context)
        print(total_output_roi,total_input_roi)

    daisy.prepare_ds(
            out_file,
            out_dataset,
            daisy.Roi(
                total_output_roi.get_offset(),
                total_output_roi.get_shape()
            ),
            voxel_size,
            np.float32,
            write_size=output_size,
            num_channels=1)

    model = UNet(
            config['in_channels'],
            config['num_fmaps'],
            config['fmap_inc_factor'],
            config['downsample_factors'],
            config['kernel_size_down'],
            config['kernel_size_up'])
    
    model.eval()

    predict = gp.torch.Predict(
            model,
            checkpoint=f'model_checkpoint_{iteration}',
            inputs = {
                'input': raw
            },
            outputs = {
                0: pred_labels
            })

    scan = gp.Scan(scan_request)

    write = gp.ZarrWrite(
            dataset_names={
                pred_labels: out_dataset
            },
            output_filename=out_file)

    pipeline = (
            source +
            gp.Normalize(raw) +
            gp.Pad(raw, None) +
            gp.IntensityScaleShift(raw, 2,-1) +
            gp.Unsqueeze([raw]) +
            gp.Unsqueeze([raw]) +
            predict +
            gp.Squeeze([pred_labels]) +
            write+
            scan)

    predict_request = gp.BatchRequest()

    predict_request[raw] = total_input_roi
    predict_request[pred_labels] = total_output_roi

    with gp.build(pipeline):
        pipeline.request_batch(predict_request)

    return total_output_roi

if __name__ == "__main__":

    iteration = 2000
    raw_file = sys.argv[1]
    raw_ds = sys.argv[2]
    section = sys.argv[3]
    raw_dataset = f'{raw_ds}/{section}'
    out_file = os.path.basename(raw_file)
    out_dataset = f'pred_labels/{section}'

    total_output_roi = predict(
        iteration,
        raw_file,
        raw_dataset,
        out_file,
        out_dataset)
    
    raw = zarr.open(raw_file)[raw_dataset][:]
    pred = zarr.open(out_file)[out_dataset][:]

    results_zarr = zarr.open(out_file, 'a')

    results_zarr[raw_dataset] = np.expand_dims(np.expand_dims(raw,axis=0),axis=0)
    results_zarr[raw_dataset].attrs["offset"] = (0,)*2
    results_zarr[raw_dataset].attrs["resolution"] = (2,)*2
    
    results_zarr[out_dataset] = np.expand_dims(pred,axis=0)
    results_zarr[out_dataset].attrs["offset"] = (0,)*2
    results_zarr[out_dataset].attrs["resolution"] = (2,)*2
