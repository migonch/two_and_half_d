from functools import partial
import sys
from pathlib import Path

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from dpipe import layers
from dpipe.batch_iter import Infinite, load_by_random_id, apply_at, unpack_args, multiply, combine_pad, random_apply
from dpipe.dataset.wrappers import apply, cache_methods
from dpipe.im import min_max_scale
from dpipe.im.metrics import dice_score, convert_to_aggregated
from dpipe.im.patch import get_random_patch
from dpipe.io import load_json, save_json
from dpipe.predict import add_extract_dims
from dpipe.torch import train_step, save_model_state, inference_step
from dpipe.train import train, TBLogger
from dpipe import commands
from dpipe.train.validator import compute_metrics
from dpipe.predict import patches_grid

from two_and_half_d.dataset import BraTS2013, ChangeSliceSpacing, CropToBrain
from two_and_half_d.metric import to_binary


BRATS_PATH = Path(sys.argv[1])
SPLIT_PATH = Path(sys.argv[2])
EXPERIMENT_PATH = Path(sys.argv[3])
FOLD = sys.argv[4]

CONFIG = {
    'source_slice_spacing': 5.,
    'batch_size': 5,
    'batches_per_epoch': 100,
    'n_epochs': 300,
    'lr': 1e-4,
    'device': 'cuda',
}

try:
    CONFIG.update(load_json(sys.argv[5]))
except (IndexError, FileNotFoundError):
    pass

PATCH_SIZE = np.array([64, 64, 32])

# dataset
raw_dataset = BraTS2013(BRATS_PATH)
dataset = apply(CropToBrain(raw_dataset), load_image=partial(min_max_scale, axes=0))
train_dataset = cache_methods(ChangeSliceSpacing(dataset, new_slice_spacing=CONFIG['source_slice_spacing']))

# cross validation
split = load_json(SPLIT_PATH)
train_ids, val_ids, test_ids = split[int(FOLD)]

# batch iterator
batch_iter = Infinite(
    load_by_random_id(train_dataset.load_image, train_dataset.load_gt, ids=train_ids),
    unpack_args(get_random_patch, patch_size=PATCH_SIZE),
    random_apply(.5, unpack_args(lambda image, gt: (np.flip(image, 1), np.flip(gt, 0)))),
    batch_size=CONFIG['batch_size'], batches_per_epoch=CONFIG['batches_per_epoch']
)

# model
model = nn.Sequential(
    nn.Conv3d(dataset.n_modalities, 8, kernel_size=3, padding=1),
    layers.FPN(
        layers.ResBlock3d, downsample=nn.MaxPool3d(2, ceil_mode=True), upsample=nn.Identity,
        merge=lambda left, down: torch.cat(layers.interpolate_to_left(left, down, 'trilinear'), dim=1),
        structure=[
            [[8, 8, 8], [16, 8, 8]],
            [[8, 16, 16], [32, 16, 8]],
            [[16, 32, 32], [64, 32, 16]],
            [[32, 64, 64], [128, 64, 32]],
            [[64, 128, 128], [256, 128, 64]],
            [[128, 256, 256], [512, 256, 128]],
                    [256, 512, 256]
        ],
        kernel_size=3, padding=1
    ),
    layers.PreActivation3d(8, dataset.n_classes, kernel_size=1)
).to(CONFIG['device'])

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])


# predict
@patches_grid(patch_size=PATCH_SIZE, stride=7 * PATCH_SIZE // 8)
@add_extract_dims(1)
def predict(image):
    return inference_step(image, architecture=model, activation=nn.Softmax(1))


# metrics
individual_metrics = {f'class {c} dice': to_binary(dice_score, c) for c in range(dataset.n_classes)}
val_metrics = convert_to_aggregated(individual_metrics)


# run experiment
logger = TBLogger(EXPERIMENT_PATH / FOLD / 'logs')
save_json(CONFIG, EXPERIMENT_PATH / 'config.json')
train(train_step, batch_iter, n_epochs=CONFIG['n_epochs'], logger=logger,
      validate=lambda : compute_metrics(predict, train_dataset.load_image, train_dataset.load_gt, val_ids, val_metrics),
      architecture=model, optimizer=optimizer, criterion=criterion, lr=CONFIG['lr'])
save_model_state(model, EXPERIMENT_PATH / FOLD / 'model.pth')

for target_slice_spacing in np.linspace(1, 5, 9):
    test_dataset = ChangeSliceSpacing(dataset, new_slice_spacing=target_slice_spacing)
    commands.predict(
        ids=test_ids,
        output_path=EXPERIMENT_PATH / FOLD / f"predictions_{CONFIG['source_slice_spacing']}_to_{target_slice_spacing}",
        load_x=test_dataset.load_image,
        predict_fn=predict
    )
    commands.evaluate_individual_metrics(
        load_y_true=test_dataset.load_gt,
        metrics=individual_metrics,
        predictions_path=EXPERIMENT_PATH / FOLD / f"predictions_{CONFIG['source_slice_spacing']}_to_{target_slice_spacing}",
        results_path=EXPERIMENT_PATH / FOLD / f"metrics_{CONFIG['source_slice_spacing']}_to_{target_slice_spacing}"
    )
