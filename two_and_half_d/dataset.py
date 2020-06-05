from pathlib import Path
import numpy as np
import hashlib
import nibabel
from medpy.io import load as load_mha

from dpipe.dataset import Dataset
from dpipe.dataset.wrappers import Proxy
from dpipe.im import zoom, crop_to_box
from dpipe.im.box import mask2bounding_box
from dpipe.io import load

from two_and_half_d.processing import multiclass_to_binary


class WMH(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.ids = [path.name for path in root.glob('*/*')]

    def _find(self, identifier):
        paths = list(self.root.glob(f'*/{identifier}'))
        assert len(paths) == 1
        return paths[0]

    def load_3dt1(self, identifier, folder='pre'):
        return load(self._find(identifier) / f'{folder}/3DT1.nii.gz')

    def load_flair(self, identifier, folder='pre'):
        return load(self._find(identifier) / f'{folder}/FLAIR.nii.gz')

    def load_t1(self, identifier, folder='pre'):
        return load(self._find(identifier) / f'{folder}/T1.nii.gz')

    def load_wmh(self, identifier):
        return load(self._find(identifier) / 'wmh.nii.gz')

    def load_spacing(self, identifier):
        """ returns FLAIR spacing"""
        return nibabel.load(self._find(identifier) / 'orig/FLAIR.nii.gz').header.get_zooms()


class BraTS2013(Dataset):
    n_modalities = 4
    n_classes = 5

    def __init__(self, root):
        self.root = Path(root)
        self.ids = tuple(f"{path.parent.name}_{path.name}" for path in self.root.glob('images/Image_Data/*/*'))

    def _find(self, identifier):
        hg_or_lg, number = identifier.split('_')
        return self.root.glob(f'images/Image_Data/{hg_or_lg}/{number}/*/*.mha')

    def _load_all(self, identifier):
        return {path.parent.name.split('.')[-1]: load_mha(str(path)) for path in self._find(identifier)}

    def _find_individual_gts(self, identifier):
        hg_or_lg, number = identifier.split('_')
        return self.root.glob(f'segmentations/*/{hg_or_lg}{number[1:]}/*/*.mha')

    def load_image(self, identifier):
        data = self._load_all(identifier)
        return np.stack([
            data['MR_Flair'][0],
            data['MR_T1'][0],
            data['MR_T1c'][0],
            data['MR_T2'][0],
        ]).astype(np.float32)

    def load_gt(self, identifier):
        return self._load_all(identifier)['OT'][0].astype(int)

    def load_individual_gts(self, identifier):
        return [load_mha(str(path))[0].astype(int) for path in self._find_individual_gts(identifier)]

    def load_spacing(self, identifier):
        data = self._load_all(identifier)
        assert np.all([header.spacing == (1., 1., 1.) for _, header in data.values()])
        return 1., 1., 1.


class BinaryGT(Proxy):
    def __init__(self, shadowed, positive_classes=(1, 2, 3, 4)):
        super().__init__(shadowed)
        self.n_classes = 2
        self.positive_classes = positive_classes

    def load_gt(self, identifier):
        """ load bool gt """
        return np.isin(self._shadowed.load_gt(identifier), self.positive_classes)


class CropToBrain(Proxy):
    def _bbox(self, identifier):
        boxes = [mask2bounding_box(mask) for mask in self._shadowed.load_image(identifier) > 0]
        return np.array([np.min([box[0] for box in boxes], 0),
                         np.max([box[1] for box in boxes], 0)])

    def load_image(self, identifier):
        return crop_to_box(self._shadowed.load_image(identifier), self._bbox(identifier))

    def load_gt(self, identifier):
        return crop_to_box(self._shadowed.load_gt(identifier), self._bbox(identifier))


class ChangeSliceSpacing(Proxy):
    def __init__(self, shadowed, new_slice_spacing):
        super().__init__(shadowed)
        self.slice_spacing = new_slice_spacing

    def _scale_factor(self, identifier):
        return self._shadowed.load_spacing(identifier)[-1] / self.slice_spacing

    def load_image(self, identifier):
        return zoom(self._shadowed.load_image(identifier), self._scale_factor(identifier), axes=-1)

    def load_gt(self, identifier):
        binary = multiclass_to_binary(self._shadowed.load_gt(identifier), labels=range(self.n_classes))
        zoomed = zoom(np.float32(binary), self._scale_factor(identifier), axes=-1)
        return np.argmax(zoomed, 0)

    def load_spacing(self, identifier):
        return (*self._shadowed.load_spacing(identifier)[:2], self.slice_spacing)


class ZooOfSpacings(Proxy):
    def __init__(self, shadowed, slice_spacings):
        super().__init__(shadowed)
        self.slice_spacings = slice_spacings

    def load_spacing(self, identifier):
        idx = int(hashlib.sha1(identifier.encode('utf-8')).hexdigest(), 16) % len(self.slice_spacings)
        return (*self._shadowed.load_spacing(identifier)[:2], self.slice_spacings[idx])

    def _scale_factor(self, identifier):
        return self._shadowed.load_spacing(identifier)[-1] / self.load_spacing(identifier)[-1]

    def load_image(self, identifier):
        return zoom(self._shadowed.load_image(identifier), self._scale_factor(identifier), axes=-1)

    def load_gt(self, identifier):
        binary = multiclass_to_binary(self._shadowed.load_gt(identifier), labels=range(self.n_classes))
        zoomed = zoom(np.float32(binary), self._scale_factor(identifier), axes=-1)
        return np.argmax(zoomed, 0)
