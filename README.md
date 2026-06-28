# EE656_Project# Deepfake Detection with EfficientNet-B0

A binary image classification pipeline for detecting AI-generated (deepfake) faces in video content, built with PyTorch and EfficientNet-B0. The project covers the full ML workflow: data preprocessing, model training, evaluation, and robustness testing under image perturbations.

---

## Project Structure

| Notebook | Purpose |
|---|---|
| `preprocessing_fixed.ipynb` | Extract and prepare face crops from raw videos |
| `training.ipynb` | Fine-tune EfficientNet-B0 for real vs. fake classification |
| `evaluation.ipynb` | Evaluate the trained model on the test set |
| `perturbed.ipynb` | Test model robustness under various image distortions |

---

## Notebooks

### 1. `preprocessing_fixed.ipynb` — Data Preparation

Processes raw video files from the **Celeb-DF** dataset into face-cropped JPEG images ready for training.

- Loads real videos from `Celeb-real/` and synthetic videos from `Celeb-synthesis/`
- Balances classes by randomly sampling fake videos to match real count (seed 42)
- Splits data into train / val / test sets (80 / 10 / 10)
- Uses **MTCNN** (via `facenet-pytorch`) to detect and crop faces from video frames
- Samples one face every 15 frames, up to 5 faces per video, resized to 224×224
- Saves crops to `processed/{split}/{real|fake}/`

**Output structure:**
```
processed/
├── train/
│   ├── real/
│   └── fake/
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

---

### 2. `training.ipynb` — Model Training

Fine-tunes a pretrained **EfficientNet-B0** model for binary classification (real vs. fake).

- Replaces the final classifier head (1000 → 2 classes)
- Uses ImageNet pretrained weights as starting point
- Training augmentations: random horizontal flip, color jitter, Gaussian blur, Gaussian noise
- Optimizer: Adam (lr=1e-4), Loss: CrossEntropyLoss
- Trains for 20 epochs, saving the best checkpoint as `best_model.pth`
- Plots training/validation accuracy and loss curves

---

### 3. `evaluation.ipynb` — Model Evaluation

Evaluates `best_model.pth` on the held-out test set.

- Computes overall test accuracy
- Generates a **confusion matrix** (seaborn heatmap)
- Prints a full **classification report** (precision, recall, F1 per class)
- Plots the **ROC curve** with AUC score

---

### 4. `perturbed.ipynb` — Robustness Testing

Tests how well the model performs when the input images are corrupted, simulating real-world degradation.

Perturbations tested:

| Perturbation | Description |
|---|---|
| White Gaussian Noise | Random noise with mean ∈ [-0.3, 0.3], variance ∈ [0, 1] |
| Gaussian Blur | Random kernel size 11–21 (odd values) |
| JPEG Compression | Random quality factor 30–95 |
| Color Contrast | Enhancement factor ∈ [0.5, 2.5] |
| Color Saturation | Enhancement factor ∈ [0.0, 2.5] |

For each perturbation, the notebook reports AUC and plots the ROC curve.

---

## Requirements

```
torch
torchvision
opencv-python
Pillow
facenet-pytorch
scikit-learn
matplotlib
seaborn
tqdm
numpy
```

Install with:

```bash
pip install torch torchvision opencv-python Pillow facenet-pytorch scikit-learn matplotlib seaborn tqdm numpy
```

A CUDA-capable GPU is recommended for preprocessing and training.

---

## Usage

Run the notebooks in order:

```
1. preprocessing_fixed.ipynb   # prepare the dataset
2. training.ipynb              # train and save best_model.pth
3. evaluation.ipynb            # evaluate on the test set
4. perturbed.ipynb             # robustness analysis
```

---

## Dataset

This project uses the **Celeb-DF** dataset. Place the raw videos at:

```
~/Downloads/Celeb-real/        # real celebrity videos (.mp4)
~/Downloads/Celeb-synthesis/   # AI-synthesized videos (.mp4)
```

The dataset is not included in this repository and must be obtained separately.
