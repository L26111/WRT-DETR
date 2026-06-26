
# WRT-DETR: Wavelet-Embedded Detection Transformer for Pavement Disease Detection

PyTorch implementation of **"WRT-DETR: Wavelet-Embedded Detection Transformer for Pavement Disease Detection"**.

## Overview

WRT-DETR enhances the RT-DETR architecture by integrating wavelet transform theory with multi-resolution analysis and directional selectivity. The proposed method introduces two core modules:

- **LDWConv (Low-frequency enhancement and Direction-adaptive Wavelet Convolution)**: Integrated into the backbone network to expand the effective receptive field while refining crack orientations through recursive low-frequency enhancement and direction-adaptive high-frequency refinement.
- **DWT-HiLo (Discrete Wavelet Transform-based HiLo Attention)**: Replaces the AIFI module in the feature fusion stage, employing lossless subband decomposition and combining local and global attention to preserve high-frequency details.

## Requirements

- Python &gt;= 3.10
- PyTorch &gt;= 2.3.0
- CUDA &gt;= 12.1
- torchvision
- numpy
- opencv-python
- pillow
- matplotlib
- pycocotools
- tqdm

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/WRT-DETR.git
cd WRT-DETR

# Create conda environment (recommended)
conda create -n wrt-detr python=3.10
conda activate wrt-detr

# Install dependencies
pip install -r requirements.txt


## Pretrained Weights

Download the best model: [WRT-DETR v1.0](https://github.com/L26111/WRT-DETR/releases/tag/v1.0)
