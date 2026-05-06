"""Data loading, cleaning, encoding, and splitting for Adult dataset imputation."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

ADULT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
HEADER = ["age", "work", "fnlwgt", "edu", "yredu", "marriage", "occupation", "relationship", "race", "sex", "capgain", "caploss", "workhr", "income"]
CONTINUOUS_COLS = ["age", "yredu", "capgain", "caploss", "workhr"]
CATEGORICAL_COLS = ["work", "marriage", "occupation", "edu", "relationship", "sex"]
FEATURE_COLS = CONTINUOUS_COLS + CATEGORICAL_COLS

@dataclass
class ProcessedAdultData:
    data: pd.DataFrame
    raw_no_missing: pd.DataFrame
    train_array: np.ndarray
    val_array: np.ndarray
    test_array: np.ndarray
    feature_names: list[str]
    cat_index: dict[str, int]
    cat_values: dict[str, list[str]]
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray

def download_adult_data(data_dir: str = "data") -> Path:
    path = Path(data_dir) / "adult.data"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"Downloading Adult dataset to {path}...")
        urlretrieve(ADULT_URL, path)
    return path

def load_raw_adult(path: str | Path | None = None, data_dir: str = "data") -> pd.DataFrame:
    if path is None:
        path = download_adult_data(data_dir)
    return pd.read_csv(path, names=HEADER, index_col=False, skipinitialspace=True)

def clean_adult(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = df[FEATURE_COLS].copy()
    missing_mask = pd.concat([(features[col] == "?") for col in CATEGORICAL_COLS], axis=1).any(axis=1)
    return features[~missing_mask].copy(), features[missing_mask].copy()

def normalize_continuous(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in CONTINUOUS_COLS:
        out[col] = (out[col] - out[col].min()) / (out[col].max() - out[col].min())
    return out

def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df, columns=CATEGORICAL_COLS).astype(np.float32)

def build_category_maps(columns: list[str]) -> tuple[dict[str, int], dict[str, list[str]]]:
    cat_index, cat_values = {}, {}
    for idx, col in enumerate(columns):
        if "_" not in col:
            continue
        feature, value = col.split("_", 1)
        if feature in CATEGORICAL_COLS:
            if feature not in cat_index:
                cat_index[feature] = idx
                cat_values[feature] = []
            cat_values[feature].append(value)
    return cat_index, cat_values

def split_array(data: np.ndarray, seed: int = 50, train_frac: float = 0.70, val_frac: float = 0.15):
    np.random.seed(seed)
    n = data.shape[0]
    indices = np.random.permutation(n)
    train_end = int(train_frac * n)
    val_end = int((train_frac + val_frac) * n)
    train_idx, val_idx, test_idx = indices[:train_end], indices[train_end:val_end], indices[val_end:]
    return data[train_idx], data[val_idx], data[test_idx], train_idx, val_idx, test_idx

def preprocess_adult(data_dir: str = "data") -> ProcessedAdultData:
    raw = load_raw_adult(data_dir=data_dir)
    no_missing, _ = clean_adult(raw)
    encoded = one_hot_encode(normalize_continuous(no_missing))
    arr = encoded.values.astype(np.float32)
    train_arr, val_arr, test_arr, train_idx, val_idx, test_idx = split_array(arr)
    cat_index, cat_values = build_category_maps(list(encoded.columns))
    return ProcessedAdultData(encoded, no_missing, train_arr, val_arr, test_arr, list(encoded.columns), cat_index, cat_values, train_idx, val_idx, test_idx)

def make_loaders(processed: ProcessedAdultData, batch_size: int = 64):
    train_tensor = torch.tensor(processed.train_array, dtype=torch.float32)
    val_tensor = torch.tensor(processed.val_array, dtype=torch.float32)
    test_tensor = torch.tensor(processed.test_array, dtype=torch.float32)
    return (
        DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True),
        DataLoader(TensorDataset(val_tensor), batch_size=batch_size, shuffle=False),
        DataLoader(TensorDataset(test_tensor), batch_size=batch_size, shuffle=False),
    )
