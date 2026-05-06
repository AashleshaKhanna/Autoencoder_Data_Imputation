"""Autoencoder model for tabular data imputation."""
from __future__ import annotations
import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int = 57, h1: int = 50, h2: int = 35) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(h2, h1), nn.ReLU(), nn.Linear(h1, input_dim), nn.Sigmoid())
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))
