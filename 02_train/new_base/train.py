import os
import json
import math
import logging
import numpy as np
import gunpowder as gp
import torch

from model import UNet, WeightedMSELoss

logging.basicConfig(level=logging.INFO)
torch.backends.cudnn.benchmark = True

data_dir = "../../01_data/2d_apical_crop.zarr"

def get_sections(data_dir, dataset):

    ds_path = os.path.join(data_dir, dataset)

    return [int(x) for x in os.listdir(ds_path) if '.' not in x]

available_sections = get_sections(data_dir, "labels")
available_sections = available_sections[0]
print(available_sections)

def train(
        max_iteration,
        in_channels,
        num_fmaps,
        fmap_inc_factor,
        downsample_factors,
        kernel_size_down,
        kernel_size_up,
        input_shape,
        voxel_size,
        batch_size,
        **kwargs):

    model = UNet(
            in_channels,
            num_fmaps,
            fmap_inc_factor,
            downsample_factors,
            kernel_size_down,
            kernel_size_up)

    loss = WeightedMSELoss()
    #loss = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-4)

    if 'output_shape' not in kwargs:
        output_shape = model.forward(torch.empty(size=[batch_size,1]+input_shape))[0].shape[1:]
        with open("config.json","r") as f:
            config = json.load(f)
            
        config['output_shape'] = list(output_shape)
            
        with open("config.json","w") as f:
            json.dump(config,f)

    else: output_shape = kwargs.get("output_shape")

    output_shape = gp.Coordinate(tuple(output_shape))
    input_shape = gp.Coordinate(tuple(input_shape))

    voxel_size = gp.Coordinate(voxel_size)
    output_size = output_shape * voxel_size
    input_size = input_shape * voxel_size

    raw = gp.ArrayKey('RAW')
    labels = gp.ArrayKey('LABELS')
    labels_mask = gp.ArrayKey('LABELS_MASK')
    pred_labels = gp.ArrayKey('PRED_LABELS')

    request = gp.BatchRequest()

    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(labels_mask, output_size)
    request.add(pred_labels, output_size)

    source = tuple(
        gp.ZarrSource(
            data_dir,
            {
                raw: f'raw/{i}',
                labels: f'labels/{i}',
                labels_mask: f'labels_mask/{i}',
            },
            {
                raw: gp.ArraySpec(interpolatable=True),
                labels: gp.ArraySpec(interpolatable=False),
                labels_mask: gp.ArraySpec(interpolatable=False),
            }) +
        gp.Normalize(raw) +
        gp.Normalize(labels, factor=1) +
        gp.Pad(raw, None) +
        gp.RandomLocation(mask=labels_mask) +
        gp.Reject(mask=labels, min_masked=0.05)#, reject_probability=0.95)
        for i in [87])

    pipeline = source
    pipeline += gp.RandomProvider()

    pipeline += gp.SimpleAugment()

    pipeline += gp.ElasticAugment(
        control_point_spacing=(50,50),
        jitter_sigma=(2.0,2.0),
        rotation_interval=(0, math.pi/2))

    pipeline += gp.IntensityAugment(
        raw,
        scale_min=0.9,
        scale_max=1.1,
        shift_min=-0.1,
        shift_max=0.1)

    #pipeline += gp.GrowBoundary(labels, steps=2, background=2)

    pipeline += gp.IntensityScaleShift(raw, 2,-1)

    pipeline += gp.Unsqueeze([raw,labels])
    pipeline += gp.Stack(batch_size)

    # pipeline += gp.PreCache(num_workers=64, cache_size=64)

    pipeline += gp.torch.Train(
        model,
        loss,
        optimizer,
        inputs={
            'input': raw
        },
        outputs={
            0: pred_labels
        },
        loss_inputs={
            0: pred_labels,
            1: labels,
        },
        save_every=1000,
        log_dir='log')

    pipeline += gp.IntensityScaleShift(raw, 0.5, 0.5)

    pipeline += gp.Snapshot(
            dataset_names={
                raw: 'raw',
                labels: 'labels',
                pred_labels: 'pred_labels'
            },
            output_filename='batch_{iteration}.zarr',
            every=1000
    )

    with gp.build(pipeline):
        for i in range(max_iteration):
            batch = pipeline.request_batch(request)


if __name__ == "__main__":

    config_path = "config.json"

    with open(config_path,"r") as f:
        config = json.load(f)

    train(**config)
