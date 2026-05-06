"""Plot training loss and imputation accuracy curves."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def plot_curves(csv_path="results/training_history.csv", output_prefix="results/autoencoder"):
    data=np.genfromtxt(csv_path, delimiter=",", names=True); epochs=np.arange(1, len(data["train_loss"])+1); Path(output_prefix).parent.mkdir(exist_ok=True)
    plt.figure(figsize=(8,5)); plt.plot(epochs, data["train_loss"], label="Train Loss"); plt.plot(epochs, data["val_loss"], label="Validation Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training and Validation Loss"); plt.legend(); plt.savefig(f"{output_prefix}_loss.png", bbox_inches="tight"); plt.close()
    plt.figure(figsize=(8,5)); plt.plot(epochs, data["train_acc"], label="Train Accuracy"); plt.plot(epochs, data["val_acc"], label="Validation Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Training and Validation Imputation Accuracy"); plt.legend(); plt.savefig(f"{output_prefix}_accuracy.png", bbox_inches="tight"); plt.close()
    print(f"Saved plots to {output_prefix}_loss.png and {output_prefix}_accuracy.png")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--csv", default="results/training_history.csv"); p.add_argument("--output-prefix", default="results/autoencoder"); a=p.parse_args(); plot_curves(a.csv, a.output_prefix)
if __name__ == "__main__": main()
