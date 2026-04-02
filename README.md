# **Visual-Textual Fusion via Cross-Attention in Dual-Stream CLIP for No-Reference Image Quality Assessment**

## Project Overview

This project implements a deep learning model for Image Quality Assessment (IQA) based on a dual-branch CLIP model and Cross-Attention mechanism. By fusing visual features and text description features, it achieves accurate prediction of image quality.
![framwork](methods.png)

---

## Directory Structure

```
.
├── iqa_clip_cross_attention.py    # Main model code
├── iqa_clip_cross_attention.md    # This documentation
└── datasets/                      # Dataset directory (text labels only)
└── requirements.txt                
```

---

## Supported Datasets


| Dataset   | Links                                                       |
| --------- | ----------------------------------------------------------- |
| TID2013      | http://www.ponomarenko.info/tid2013.htm     |
| CSIQ      | https://s2.smu.edu/~eclarson/csiq.html |
| AGIQA     |  https://github.com/lcysyzxdxc/AGIQA-3k-Database 
| LIVE      |  https://live.ece.utexas.edu/research/Quality/index.htm 


---

## Hyperparameter Configuration


| Parameter    | Default Value                                                                     | Description             |
| ------------ | --------------------------------------------------------------------------------- | ----------------------- |
| `model_name` | `"ViT-L-14-336"`                                                                  | CLIP model name         |
| `pretrained` | `"/home/user/model/open_clip_model.safetensors"（Yours pre-trained model address）` | Pretrained weights path |
| `batch_size` | `8`                                                                               | Batch size              |
| `lr`         | `1e-4`                                                                            | Learning rate           |


---

## Evaluation Metrics


| Metric              | Description                 |
| ------------------- | --------------------------- |
| **Spearman (SRCC)** | Measures rank correlation   |
| **Pearson (PLCC)**  | Measures linear correlation |
| **MSE**             | Mean Squared Error          |


---
## Downloading Pre-trained Models

This project uses OpenCLIP pre-trained models. You need to download the model weights before training.

### Using OpenCLIP Library (Recommended)

```python
import open_clip

# Download and cache the model
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-L-14-336', 
    pretrained='openai'
)
```

### Manual Download

If you prefer manual download, you can download the `.safetensors` file from the following sources:

| Model | Download Link | File |
|-------|--------------|------|
| **ViT-L-14-336** | [HuggingFace - open_clip](https://huggingface.co/huggingface/open_clip) | `open_clip_pytorch_model.safetensors` |
| **ViT-L-14** | [HuggingFace - open_clip](https://huggingface.co/huggingface/open_clip) | `open_clip_pytorch_model.safetensors` |

After downloading, update the `pretrained` parameter in the code:

```python
pretrained = "/path/to/your/open_clip_model.safetensors"
```

---

## Usage

### 1. Basic Training

```python
from iqa_clip_cross_attention import main

# Train on specified dataset
main(dataset_name="tid2013")
```

### 2. Modify Dataset

Modify the configuration at the beginning of the code:

```python
dataset = "agiqa"  # Options: agiqa, tid2013, csiq, live
```

### 3. Modify Model Configuration

```python
model_this = "my_custom_model"  # Custom model name
epochs = 60                      # Increase training epochs
batch_size = 8                  # Adjust batch size
lr = 1e-4                        # Adjust learning rate
```

---


## Dependencies

```
torch >= 1.9.0
torchvision
open_clip_torch
pillow
pandas
numpy
tqdm
scipy
scikit-learn
matplotlib
```

### Installing Dependencies

```bash
pip install -r requirements.txt
```

---

## Notes

1. **Data Paths**: Ensure dataset paths are configured correctly and image files exist
2. **CLIP Weights**: The pretrained model path should point to a valid `.safetensors` file
3. **GPU Memory**: The ViT-L-14-336 model is large, recommended to use at least 16GB VRAM
4. **Text Column Names**: Different datasets may have different column names, ensure CSV files contain required fields

