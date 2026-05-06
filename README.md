# Adult Income Data Imputation with Autoencoders

A PyTorch project that uses a fully connected autoencoder to impute missing categorical values in the UCI Adult Income dataset.

This repository converts a data-imputation lab into a GitHub-ready ML engineering project with reusable data preprocessing, model training, hyperparameter tuning, baseline comparison, and test-set evaluation.

## Why this project is useful for engineering roles

- Cleans real tabular data with missing values
- Normalizes continuous features
- One-hot encodes categorical variables
- Trains a PyTorch autoencoder for tabular reconstruction
- Simulates missing categorical features during training
- Measures imputation accuracy on masked categorical values
- Compares against a simple most-common-category baseline
- Runs hyperparameter experiments and saves reproducible metrics

## Dataset

The project uses the UCI Adult dataset (`adult.data`). The original dataset is commonly used for income classification, but this project reframes it as a missing categorical-value imputation task.

The lab processed the dataset into 30,718 clean records and 57 features after removing missing records, normalizing continuous features, and one-hot encoding categorical features.

## Repository structure

```text
adult-income-autoencoder-data-imputation/
├── README.md
├── PROJECT_SUMMARY.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── baseline.py
│   ├── plot_curves.py
│   └── utils.py
├── experiments/
│   └── run_experiments.py
├── results/
├── checkpoints/
└── data/
```

## Setup

```bash
git clone <your-repo-url>
cd adult-income-autoencoder-data-imputation
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

## Train the autoencoder

```bash
python -m src.train --epochs 30 --learning-rate 0.001 --batch-size 64 --h1 50 --h2 35
```

## Evaluate test imputation accuracy

```bash
python -m src.evaluate --checkpoint checkpoints/best_autoencoder.pt
```

## Compare against baseline

```bash
python -m src.baseline
```

## Plot curves

```bash
python -m src.plot_curves --csv results/training_history.csv
```

## Run hyperparameter experiments

```bash
python experiments/run_experiments.py
```

## Representative lab results

| Model | Test imputation accuracy |
|---|---:|
| Most-common-category baseline | ~45.69% |
| Autoencoder | ~65.45% |

The autoencoder outperforms the simple baseline by learning relationships between demographic, employment, education, and relationship features instead of always selecting the most frequent category.
