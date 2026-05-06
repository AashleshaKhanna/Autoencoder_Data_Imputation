"""Run hyperparameter experiments for Adult autoencoder imputation."""
from __future__ import annotations
import pandas as pd
from src.train import train
EXPERIMENTS=[
    {"name":"baseline","h1":40,"h2":32,"learning_rate":1e-3,"batch_size":64,"epochs":30},
    {"name":"lower_lr","h1":40,"h2":32,"learning_rate":5e-4,"batch_size":64,"epochs":30},
    {"name":"larger_model","h1":50,"h2":35,"learning_rate":1e-3,"batch_size":64,"epochs":30},
    {"name":"larger_batch","h1":40,"h2":32,"learning_rate":1e-3,"batch_size":128,"epochs":30},
]
def main():
    results=[]
    for exp in EXPERIMENTS:
        print("="*80); print(f"Running experiment: {exp}"); print("="*80)
        hist=train(exp["epochs"], exp["learning_rate"], exp["batch_size"], exp["h1"], exp["h2"])
        results.append({**exp,"final_train_loss":hist["train_loss"][-1],"final_val_loss":hist["val_loss"][-1],"final_train_acc":hist["train_acc"][-1],"final_val_acc":hist["val_acc"][-1]})
    df=pd.DataFrame(results).sort_values("final_val_acc", ascending=False); df.to_csv("results/experiment_summary.csv", index=False); print(df)
if __name__ == "__main__": main()
