"""Arsitektur asli notebook: baseline 1D-CNN dan CNN-LSTM (cell 14)."""
import torch
from torch import nn
from .config import N_CLASS, N_CHANNELS


class CSIModel(nn.Module):
    def __init__(self, use_lstm=False, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(N_CHANNELS, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2), nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(128, 64, batch_first=True) if use_lstm else None
        self.classifier = nn.Linear(64 if use_lstm else 128, N_CLASS)

    def forward(self, x):
        x = self.encoder(x)
        x = self.lstm(x.transpose(1, 2))[1][0][-1] if self.lstm else x.mean(-1)
        return self.classifier(x)
