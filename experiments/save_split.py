import sys
from pathlib import Path

from dpipe.split import train_val_test_split
from dpipe.io import save_json

from two_and_half_d.dataset import BraTS2013

BRATS_PATH = Path(sys.argv[1])
SPLIT_PATH = Path(sys.argv[2])
VAL_SIZE = 1
N_FOLDS = 5
RANDOM_STATE = 42

save_json(
    train_val_test_split(BraTS2013(BRATS_PATH).ids, val_size=VAL_SIZE, n_splits=N_FOLDS, random_state=RANDOM_STATE),
    SPLIT_PATH
)
