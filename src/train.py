"""Train the autoencoder imputation model."""
from __future__ import annotations
import argparse
import numpy as np
import torch
import torch.nn as nn
from src.data import make_loaders, preprocess_adult
from src.model import AutoEncoder
from src.utils import ensure_dirs, imputation_accuracy, save_checkpoint, zero_out_random_feature

def train(epochs=30, learning_rate=1e-3, batch_size=64, h1=50, h2=35, data_dir="data", device=None):
    ensure_dirs(); torch.manual_seed(42)
    processed = preprocess_adult(data_dir=data_dir)
    train_loader, val_loader, _ = make_loaders(processed, batch_size=batch_size)
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = AutoEncoder(input_dim=processed.train_array.shape[1], h1=h1, h2=h2).to(device_obj)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train(); train_loss_sum = 0.0; train_examples = 0
        for (data,) in train_loader:
            data = data.to(device_obj)
            masked = zero_out_random_feature(data.clone(), processed.cat_index, processed.cat_values)
            recon = model(masked)
            loss = criterion(recon, data)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            train_loss_sum += loss.item() * data.size(0); train_examples += data.size(0)
        train_loss = train_loss_sum / train_examples
        model.eval(); val_loss_sum = 0.0; val_examples = 0
        with torch.no_grad():
            for (data,) in val_loader:
                data = data.to(device_obj)
                masked = zero_out_random_feature(data.clone(), processed.cat_index, processed.cat_values)
                loss = criterion(model(masked), data)
                val_loss_sum += loss.item() * data.size(0); val_examples += data.size(0)
        val_loss = val_loss_sum / val_examples
        train_acc = imputation_accuracy(model, train_loader, processed.cat_index, processed.cat_values, device_obj)
        val_acc = imputation_accuracy(model, val_loader, processed.cat_index, processed.cat_values, device_obj)
        history["train_loss"].append(train_loss); history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc); history["val_acc"].append(val_acc)
        print(f"Epoch {epoch:02d}: Train Loss={train_loss:.6f}, Validation Loss={val_loss:.6f}, Train Acc={train_acc:.4f}, Validation Acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint("checkpoints/best_autoencoder.pt", model, {"h1": h1, "h2": h2, "input_dim": processed.train_array.shape[1], "feature_names": processed.feature_names, "cat_index": processed.cat_index, "cat_values": processed.cat_values, "best_val_acc": best_val_acc})
    np.savetxt("results/training_history.csv", np.column_stack([history["train_loss"], history["val_loss"], history["train_acc"], history["val_acc"]]), delimiter=",", header="train_loss,val_loss,train_acc,val_acc", comments="")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    return history

def main():
    p=argparse.ArgumentParser(); p.add_argument("--epochs", type=int, default=30); p.add_argument("--learning-rate", type=float, default=1e-3); p.add_argument("--batch-size", type=int, default=64); p.add_argument("--h1", type=int, default=50); p.add_argument("--h2", type=int, default=35); p.add_argument("--data-dir", default="data"); p.add_argument("--device", default=None)
    a=p.parse_args(); train(a.epochs, a.learning_rate, a.batch_size, a.h1, a.h2, a.data_dir, a.device)
if __name__ == "__main__": main()
