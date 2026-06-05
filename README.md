# LDCT Denoising — 2.5D U-Net

A clean, modular PyTorch pipeline for **Low-Dose CT (LDCT) denoising** using a pseudo-3D (2.5D) MONAI U-Net.

## Project Structure

```
LDCT/
├── config.py          # All hyperparameters, paths, and constants
├── download.py        # Parallel NBIA dataset downloader with resume support
├── dataset.py         # Pseudo-3D data pipeline (MONAI transforms + DataLoaders)
├── model.py           # MONAI U-Net builder
├── losses.py          # Hybrid loss (L1 + SSIM + Perceptual + Edge)
├── metrics.py         # PSNR, RMSE, VIF evaluation metrics
├── utils.py           # Reproducibility and DICOM sorting helpers
├── train.py           # Main training script
└── requirements.txt   # Dependencies
```

## Method

The model uses **residual learning** on pseudo-3D input:
- **Input**: 3 adjacent DICOM slices stacked as a 3-channel image `[prev, curr, next]`
- **Target**: The residual (noise) between LDCT input and NDCT (full-dose) ground truth
- **Output**: Denoised slice = input + predicted residual

### Loss Function

| Component | Weight | Purpose |
|---|---|---|
| L1 Loss | 1.0 | Pixel-wise accuracy |
| SSIM Loss | 0.5 | Structural similarity |
| VGG Perceptual Loss | 0.1 | High-level feature preservation |
| Sobel Edge Loss | 0.05 | Sharp edge recovery |

### Evaluation Metrics

- **PSNR** (↑) — Peak Signal-to-Noise Ratio
- **SSIM** (↑) — Structural Similarity Index
- **RMSE** (↓) — Root Mean Squared Error
- **VIF** (↑) — Visual Information Fidelity

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Download Data

Downloads 36 patients from the [LDCT-and-projection-data](https://www.cancerimagingarchive.net/collection/ldct-and-projection-data/) collection on TCIA:

```bash
python download.py
```

This creates:
```
dataset/   ← 30 patients for training (15 Chest + 15 Abdomen)
test/      ← 6 patients for testing   (3 Chest  + 3 Abdomen)
```

### 2. Train

```bash
python train.py
```

Training supports **automatic resume** from the last checkpoint. Logs are written to TensorBoard:

```bash
tensorboard --logdir FinalCT_2.5D-UNET-DATASET/logs
```

## Configuration

All settings are centralized in [`config.py`](config.py). Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `TOTAL_EPOCHS` | 200 | Training epochs |
| `LEARNING_RATE` | 1e-4 | AdamW learning rate |
| `TRAIN_BATCH_SIZE` | 32 | Training batch size |
| `SPATIAL_SIZE` | (256, 256) | Input patch size |
| `PATIENCE` | 10 | Early stopping patience |

## Dataset

Uses the **LDCT-and-projection-data** collection from The Cancer Imaging Archive (TCIA). Patients are split stratified by body type (Chest / Abdomen) and selected by smallest file size first.

> **Note**: You need a valid NBIA account or anonymous access may be available depending on the collection.

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0 with CUDA (recommended)
- See `requirements.txt` for the full list
