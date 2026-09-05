"""Stratified train/held-out split from the real So2Sat-LCZ42 validation.h5 (24,119 real
geo-matched Sentinel-1/Sentinel-2 patches, 17-class local climate zone labels). The official
16GB training.h5 is impractically large to download for this post, so this validation split is
further divided into our own train/held-out sets -- noted explicitly, not hidden.
"""
import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from common import LCZ_CLASSES, render_all_views

H5_PATH = "data/v4/validation.h5"


def stratified_split(labels: np.ndarray, n_train_per_class=100, n_val_per_class=15, seed=0):
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    class_ids = labels.argmax(axis=1)
    for c in range(labels.shape[1]):
        idxs = np.where(class_ids == c)[0]
        rng.shuffle(idxs)
        train_idx.extend(idxs[:n_train_per_class])
        val_idx.extend(idxs[n_train_per_class:n_train_per_class + n_val_per_class])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return np.array(train_idx), np.array(val_idx)


class So2SatViewsDataset(Dataset):
    def __init__(self, indices: np.ndarray):
        self.indices = indices
        self._file = None

    @property
    def file(self):
        if self._file is None:
            self._file = h5py.File(H5_PATH, "r")
        return self._file

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        sen1 = np.array(self.file["sen1"][idx])
        sen2 = np.array(self.file["sen2"][idx])
        label = np.array(self.file["label"][idx])
        cls_name = LCZ_CLASSES[int(label.argmax())]
        views = render_all_views(sen1, sen2)  # [(name, PIL.Image), ...] x6
        return views, cls_name
