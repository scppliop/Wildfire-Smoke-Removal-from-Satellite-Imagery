# Wildfire Smoke Removal from Satellite Imagery

> Removing wildfire smoke from Sentinel-2 satellite imagery using Deep Graph Total Variation (DGTV) and AOD-Net.  
> *IE7615 Neural Networks / Deep Learning — April 2026*

---

## Overview

Dense wildfire smoke obscures satellite imagery used for evacuation coordination and damage assessment. This project explores two deep learning approaches to restore smoke-degraded Sentinel-2 imagery from British Columbia wildfire events.

**Key Results:**

| Model | PSNR | SSIM | Parameters |
|-------|------|------|------------|
| DGTV | 27.52 dB | 0.8282 | ~120K |
| AOD-Net | 27.03 dB | **0.9255** | ~10K |

---

## Methods

### Method 1: Deep Graph Total Variation (DGTV)
A hybrid approach combining Graph Signal Processing with deep feature learning.
- Lightweight 4-layer CNN builds an adaptive pixel similarity graph
- Analytical GTV low-pass filter derived from convex optimization with ℓ₁ prior
- 80% fewer parameters than DnCNN with competitive PSNR
- **Limitation:** Assumes AWGN noise model — fundamentally mismatched to smoke's physics (ASM)

### Method 2: AOD-Net (All-in-One Dehazing Network)
Physics-aware end-to-end CNN based on Li et al. (ICCV 2017), adapted for wildfire smoke.
- Directly encodes the Atmospheric Scattering Model (ASM): `y = x·(1−t) + A·t`
- Reformulates ASM so a single CNN estimates everything: `x̂ = K(y)·y − K(y) + 1`
- 5 conv layers with multi-scale feature concatenation (1×1, 3×3, 5×5, 7×7)
- Only ~10K parameters — outperforms DGTV in SSIM

---

## Dataset

**Source:** Copernicus Data Space — Sentinel-2 MSI (13 bands, 10m/20m/60m resolution)

| Split | Period | Description |
|-------|--------|-------------|
| Clean images (5) | May–June 2020–2025 | Pre-wildfire season, cloud cover < 5% |
| Smoky images | July–August 2023 | Peak wildfire activity — Kelowna, Shuswap, Interior BC |

**Preprocessing Pipeline:**
1. Load `.SAFE` files → extract RGB bands (B02, B03, B04)
2. Gamma correction & normalization
3. ROI extraction → 256×256 patches (from 1024×1024)
4. Physics-based smoke synthesis via ASM (Brown / White / Grey smoke profiles)
5. Quality filtering (mean intensity > 0.2)
6. Data augmentation → **9,600 clean/smoky pairs**

---

## Repository Structure

```
├── Preprocessing_Augmentation-final-code.ipynb   # Data loading, ASM synthesis, augmentation
├── DGTV.ipynb                                     # DGTV model, training, and inference
├── AOD_NET.ipynb                                  # AOD-Net model, training, and inference
└── Data/
    ├── Clean/          # Clean .SAFE Sentinel-2 scenes
    ├── Smoky/          # Real smoky .SAFE scenes (2023 BC wildfires)
    └── patches_A/      # Preprocessed patch arrays (.npy)
```

---

## Usage

### 1. Preprocessing & Data Augmentation
Run `Preprocessing_Augmentation-final-code.ipynb` to:
- Load Sentinel-2 `.SAFE` files
- Synthesize smoky/clean patch pairs using ASM
- Save augmented dataset as `.npy` arrays

### 2. Train & Evaluate DGTV
Run `DGTV.ipynb`:
- Experiments follow 3 stages: feature dim K → optimizer → GTV layers T
- Best config: K=3, T=2, SGD (lr=1e-4), batch=16, epochs=50

### 3. Train & Evaluate AOD-Net
Run `AOD_NET.ipynb`:
- Trains AOD-Net end-to-end on synthesized smoke pairs
- Includes inference on real smoky Sentinel-2 imagery

### Requirements
```bash
pip install torch torchvision rasterio numpy matplotlib
```
> Notebooks are designed to run on **Google Colab** (GPU recommended).

---

## Atmospheric Scattering Model (ASM)

The physical basis for smoke simulation and AOD-Net's design:

```
I(x) = J(x) · t(x) + A · (1 − t(x))
```

| Variable | Description |
|----------|-------------|
| `I(x)` | Observed smoky image (input) |
| `J(x)` | Clean scene radiance (target) |
| `t(x)` | Transmission map |
| `A` | Global atmospheric light |

---

## References

1. Vu, H., Cheung, G., & Eldar, Y. C. (2021). *Unrolling of Deep Graph Total Variation for Image Denoising.* arXiv:2010.11290.
2. Li, B., et al. (2017). *AOD-Net: All-in-One Dehazing Network.* IEEE ICCV.
3. Zhang, K., et al. (2017). *Beyond a Gaussian Denoiser.* IEEE TIP.
4. ESA / Copernicus. [Sentinel-2 MSI](https://dataspace.copernicus.eu)
5. BC Wildfire Service. (2023). *2023 Wildfire Season Summary.*

---

## Author

**Juwon Park** — IE7615 Neural Networks / Deep Learning, April 2026
