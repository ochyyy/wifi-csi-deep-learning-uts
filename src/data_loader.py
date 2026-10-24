"""Loader dataset terproses dan normalisasi input sesuai notebook (cell 10, 16)."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from .config import N_CLASS, N_CHANNELS, TIME_STEPS, SPLITS


class CSIData:
    def __init__(self, npz_path):
        self.path = Path(npz_path)
        if not self.path.is_file():
            raise FileNotFoundError(f'Cache .npz tidak ditemukan: {self.path}')
        with np.load(self.path) as z:
            self.X_all, self.y_all, self.split_all, self.sample_ids = (
                z[k].copy() for k in ('X', 'y', 'split', 'sample_id')
            )
        assert self.X_all.shape[1:] == (N_CHANNELS, TIME_STEPS)
        assert np.isfinite(self.X_all).all()
        assert len(self.X_all) == len(self.y_all) == len(self.split_all) == len(self.sample_ids)
        assert set(self.y_all) == set(range(N_CLASS))
        assert set(self.split_all) == {'train', 'val', 'test'}
        meta = pd.DataFrame({'sample_id': self.sample_ids, 'split': self.split_all})
        meta['subject'] = meta.sample_id.str.split('_').str[1]
        for split, subjects in SPLITS.items():
            assert set(meta.loc[meta.split == split, 'subject']) == set(subjects), split

    def prepare_split(self, split, mode='full'):
        if mode not in ('full', 'amplitude', 'amplitude_norm'):
            raise ValueError(f'Mode input tidak dikenal: {mode}')
        if split not in SPLITS:
            raise ValueError(f'Split tidak dikenal: {split}')
        mask = self.split_all == split
        x = torch.from_numpy(self.X_all[mask].copy()).float()
        if mode in ('amplitude', 'amplitude_norm'):
            x[:, 342:] = 0
        if mode == 'amplitude_norm':
            amp = x[:, :342]
            x[:, :342] = (amp - amp.mean(-1, keepdim=True)) / amp.std(
                -1, keepdim=True, unbiased=False).clamp_min(1e-3)
        y = torch.from_numpy(self.y_all[mask].copy()).long()
        assert torch.isfinite(x).all()
        return TensorDataset(x, y), self.sample_ids[mask].copy()
