"""Most-common-category baseline for categorical imputation."""
from __future__ import annotations
from src.data import CATEGORICAL_COLS, preprocess_adult

def run_baseline(data_dir="data"):
    processed=preprocess_adult(data_dir=data_dir)
    train_df=processed.raw_no_missing.iloc[processed.train_idx]
    test_df=processed.raw_no_missing.iloc[processed.test_idx]
    most_common={col: train_df[col].mode()[0] for col in CATEGORICAL_COLS}
    total=correct=0
    for _, row in test_df.iterrows():
        for col in CATEGORICAL_COLS:
            correct += int(most_common[col] == row[col]); total += 1
    acc=correct/total
    print("Most common training values:")
    for col, value in most_common.items(): print(f"  {col}: {value}")
    print(f"Baseline test accuracy: {acc:.4f}"); print(f"Baseline test accuracy (%): {acc*100:.2f}")
    return acc
if __name__ == "__main__": run_baseline()
