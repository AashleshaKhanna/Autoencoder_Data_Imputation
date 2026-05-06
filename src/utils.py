"""Masking, decoding, metrics, and checkpoint utilities."""
from __future__ import annotations
import random
from pathlib import Path
import numpy as np
import torch

def ensure_dirs() -> None:
    Path("results").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)

def get_onehot(record: np.ndarray, feature: str, cat_index: dict[str, int], cat_values: dict[str, list[str]]) -> np.ndarray:
    start = cat_index[feature]
    return record[start:start + len(cat_values[feature])]

def get_categorical_value(onehot: np.ndarray, feature: str, cat_values: dict[str, list[str]]) -> str:
    return cat_values[feature][int(np.argmax(onehot))]

def get_feature(record: np.ndarray, feature: str, cat_index: dict[str, int], cat_values: dict[str, list[str]]) -> str:
    return get_categorical_value(get_onehot(record, feature, cat_index, cat_values), feature, cat_values)

def zero_out_feature(records: torch.Tensor, feature: str, cat_index: dict[str, int], cat_values: dict[str, list[str]]) -> torch.Tensor:
    start = cat_index[feature]
    records[:, start:start + len(cat_values[feature])] = 0
    return records

def zero_out_random_feature(records: torch.Tensor, cat_index: dict[str, int], cat_values: dict[str, list[str]]) -> torch.Tensor:
    return zero_out_feature(records, random.choice(list(cat_index.keys())), cat_index, cat_values)

@torch.no_grad()
def imputation_accuracy(model: torch.nn.Module, loader: torch.utils.data.DataLoader, cat_index: dict[str, int], cat_values: dict[str, list[str]], device="cpu") -> float:
    model.eval(); model.to(device)
    total = correct = 0
    for feature in cat_index:
        for (batch,) in loader:
            original = batch.to(device)
            masked = zero_out_feature(original.clone(), feature, cat_index, cat_values)
            recon = model(masked)
            start = cat_index[feature]
            end = start + len(cat_values[feature])
            correct += int((recon[:, start:end].argmax(dim=1) == original[:, start:end].argmax(dim=1)).sum().item())
            total += int(original.size(0))
    return correct / total

def save_checkpoint(path: str, model: torch.nn.Module, metadata: dict) -> None:
    ensure_dirs()
    torch.save({"model_state": model.state_dict(), "metadata": metadata}, path)
