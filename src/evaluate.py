"""Evaluate the trained autoencoder on the held-out test split."""
from __future__ import annotations
import argparse
import torch
from src.data import make_loaders, preprocess_adult
from src.model import AutoEncoder
from src.utils import imputation_accuracy

def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint", default="checkpoints/best_autoencoder.pt"); p.add_argument("--batch-size", type=int, default=64); p.add_argument("--data-dir", default="data"); p.add_argument("--device", default=None)
    a=p.parse_args(); device=torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    processed=preprocess_adult(data_dir=a.data_dir); _,_,test_loader=make_loaders(processed, a.batch_size)
    ckpt=torch.load(a.checkpoint, map_location=device); meta=ckpt["metadata"]
    model=AutoEncoder(meta.get("input_dim", processed.test_array.shape[1]), meta.get("h1",50), meta.get("h2",35)).to(device)
    model.load_state_dict(ckpt["model_state"])
    acc=imputation_accuracy(model, test_loader, processed.cat_index, processed.cat_values, device)
    print(f"Test imputation accuracy: {acc:.4f}"); print(f"Test imputation accuracy (%): {acc*100:.2f}")
if __name__ == "__main__": main()
