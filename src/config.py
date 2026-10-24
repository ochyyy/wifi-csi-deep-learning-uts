"""Konfigurasi bersama yang disalin dari notebook Colab asli."""
from pathlib import Path
import torch

N_CLASS = 27
TIME_STEPS = 32
N_CHANNELS = 684
SEED = 42
BATCH_SIZE = 64
EPOCHS = 40
PATIENCE = 8
SPLITS = {
    'train': [f'S{i:02d}' for i in range(1, 7)],
    'val': ['S07', 'S08'],
    'test': ['S09', 'S10'],
}

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEFAULT_DATA = Path('data/processed/processed_E01_v1.npz')
DEFAULT_CHECKPOINTS = Path('models/checkpoints')
DEFAULT_RESULTS = Path('results/generated')
