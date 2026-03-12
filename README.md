# Ship Track Detection with U-Net

Pixel-level segmentation of ship tracks in MODIS satellite imagery using a small U-Net.

Ship tracks are bright linear cloud features caused by ship aerosol emissions. This project trains a U-Net to localise them in false-colour MODIS composites from the CloudTracks dataset.

## Dataset

The sample images included here are just enough to verify the pipeline runs. The full dataset (1,780 images, 12,000+ annotations) can be downloaded from:

https://zenodo.org/records/10042922

Each sample is a 3-channel false-colour PNG (MODIS channels 1, 20, 32) paired with a LabelMe JSON containing polyline annotations. Native resolution is 1354x2030; images are resized to 512x512 for training.

## Model

Two-level U-Net with skip connections. See `model_architecture.png` for a diagram.

Each conv block is two layers of Conv3x3 → BatchNorm → ReLU. Upsampling uses transposed convolutions. Masks are rasterised from polyline annotations with 10px width per the original paper.

## Pre-trained Results

The `pre_trained_results/` folder has outputs from a longer training run:

| Setting | Value |
|---|---|
| Images | 600 |
| Epochs | 85 |
| Batch size | 4 |
| Resolution | 512x512 |
| Loss | BCEWithLogitsLoss |
| Optimiser | Adam, lr=1e-3 |

## Quick Start

```
pip install -r requirements.txt
python shiptrack_unet.py
```

This trains on `data/train/` (10 samples, 10 epochs), runs inference on `data/test/`, and writes result images to `results/`.

To train on the full dataset, download from Zenodo, drop files into `data/train/images/` and `data/train/jsons/`, and bump `max_images` and `epochs` in the script.

## Structure

```
├── README.md
├── shiptrack_unet.py
├── requirements.txt
├── model_architecture.png
├── data/
│   ├── train/
│   │   ├── images/
│   │   └── jsons/
│   └── test/
│       ├── images/
│       └── jsons/
├── pre_trained_results/
└── results/
```

## References

- Chaudhry et al. (2024). CloudTracks: A Dataset for Localizing Ship Tracks in Satellite Images of Clouds. [arXiv:2401.14486](https://arxiv.org/abs/2401.14486)
- Ronneberger et al. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.
