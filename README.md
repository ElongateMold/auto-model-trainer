# Auto Model Trainer — Modular Machine Learning Framework

An object-oriented, modular Python platform for automated exploratory data analysis (EDA), leak-free preprocessing, and autonomous routing between **Supervised Learning** and **Unsupervised Clustering**.

---

## Quick Start from Terminal (Windows — No Admin or Python Required)

You can download and launch the portable executable directly into your user profile (`%LOCALAPPDATA%\ml_framework`) without administrator privileges or a local Python installation.

### 1. Interactive Menu Mode
Open **PowerShell** (standard user terminal) and run:
```powershell
irm [https://raw.githubusercontent.com/ElongateMold/auto-model-trainer/main/run.ps1](https://raw.githubusercontent.com/ElongateMold/auto-model-trainer/main/run.ps1) | iex

## Project Architecture

```text
ml_framework/
│
├── data/                       # Input tabular datasets (.csv, .tsv, .parquet)
├── reports/                    # Generated 300 DPI charts, CSV leaderboards & saved models
│
├── core/
│   ├── __init__.py
│   ├── loader.py               # Multi-format reader (utf-8/latin-1) & auto-target detection
│   ├── transformer.py          # Category spelling normalization (automatic or interactive)
│   ├── eda.py                  # Structural diagnostics & 300 DPI correlation heatmaps
│   ├── model_selector.py       # Feature pruning & supervised model family selection
│   └── preprocessor.py         # Conditional StandardScaler, imputation & One-Hot Encoding
│
├── models/
│   ├── __init__.py
│   ├── base.py                 # Abstract BaseModelTrainer (fit, predict, score, save) + CV
│   ├── supervised/
│   │   ├── __init__.py
│   │   ├── linear_models.py    # Logistic Regression (L1/L2), Ridge & Lasso
│   │   ├── tree_models.py      # CART (with variable importances) & Random Forest
│   │   ├── spline_models.py    # MARS (hinge basis + GCV pruning) & GAM (Penalized B-Splines)
│   │   └── prototype_models.py # Euclidean k-NN, LVQ1 & DANN (deformable local metric)
│   │
│   └── unsupervised/
│       ├── __init__.py
│       └── kmeans_engine.py    # K-Means audit (Geometric Elbow, Silhouette, CH, DB) & 2D PCA
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py              # Classification & regression leaderboard scoring
│   └── visualizer.py           # 300 DPI HD diagnostic plots
│
├── main.py                     # CLI and interactive entry point
├── README.md                   # Project documentation
└── requirements.txt            # Pinned dependencies