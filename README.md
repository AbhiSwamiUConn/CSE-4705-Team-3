# CSE-4705-Team-3: Musical Symbol Classification

This project implements a comparative study of three machine learning models (Logistic Regression, Random Forest, and CNN) to classify handwritten musical symbols (Whole, Half, and Quarter notes) using the HOMUS dataset.

---

## Project Directory

- **data/**
  - **raw/**: Sorted `.txt` stroke files from the HOMUS dataset.
  - **processed/**: Standardized 32x32 `.npy` arrays for model training.
  - `organize_data.py`: Automates sorting raw HOMUS files from a download directory.
- **src/**
  - `preprocess.py`: Renders strokes into binarized 32x32 images.
  - `verify_data.py`: Visual verification script to check rendered samples.
- **models/**: Individual directories for each model implementation.
- **results/**: Final comparison plots, confusion matrices, and inference timing.
- **requirements.txt**: Project dependencies (TensorFlow, Scikit-learn, OpenCV).

---

## Getting Started (Internal Team Workflow)

### 1. Environment Setup

Ensure you are using **Python 3.12**.

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Data Pipeline

The data is processed in a three-stage pipeline to ensure consistency across all models.

**Stage 1 — Organization**

Sort raw HOMUS files into class folders:

```bash
python data/organize_data.py
```

**Stage 2 — Preprocessing**

Render strokes to images, apply Adaptive Gaussian Thresholding, and normalize pixels to `[0.0, 1.0]`:

```bash
python src/preprocess.py
```

**Stage 3 — Outputs**

Four files are generated in `data/processed/`:

| File | Shape | Used By |
|---|---|---|
| `X_train_cnn.npy` / `X_test_cnn.npy` | `(N, 32, 32, 1)` | CNN |
| `X_train_flat.npy` / `X_test_flat.npy` | `(N, 1024)` | Logistic Regression, Random Forest |

---

## Model Comparison Assignments

To ensure a fair scientific comparison, each model must use the exact same `.npy` files from `data/processed/`.

| Model | Owner | Input Format | Key Metric |
|---|---|---|---|
| Logistic Regression | Abhinav | 2D Flattened (1024) | Baseline Accuracy |
| Random Forest | Justin | 2D Flattened (1024) | Overfitting Analysis |
| CNN | Krish | 4D Spatial (32x32x1) | Max Accuracy / GPU Perf |

---

## Usage

To verify the data is rendered correctly before training:

```bash
python src/verify_data.py
```
