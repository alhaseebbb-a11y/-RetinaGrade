# 👁️ Diabetic Retinopathy Detection and Severity Grading

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![Colab](https://img.shields.io/badge/Google%20Colab-GPU-yellow?logo=googlecolab&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Training%20In%20Progress-blue)

**A deep learning pipeline for automated detection and severity grading of Diabetic Retinopathy using EfficientNet-B3 transfer learning.**

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [DR Severity Grades](#-dr-severity-grades)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Project Structure](#-project-structure)
- [Notebooks](#-notebooks)
- [Training Pipeline (Google Colab)](#-training-pipeline-google-colab)
- [Results](#-results-to-be-updated-after-training)
- [How to Run](#-how-to-run)
- [Requirements](#-requirements)

---

## 🔬 Overview

Diabetic Retinopathy (DR) is a leading cause of preventable blindness worldwide. Early and accurate grading of DR severity is critical for timely clinical intervention. This project implements a **production-quality deep learning pipeline** that:

- Automatically classifies fundus retinal images into **5 severity grades** (No DR → Proliferative DR)
- Uses **EfficientNet-B3** pretrained on ImageNet with a two-phase transfer learning strategy
- Provides a complete **Google Colab training pipeline** ready to run with GPU acceleration
- Includes **full evaluation** — confusion matrix, per-class metrics, error analysis, and inference function

---

## 🏥 DR Severity Grades

| Label | Grade | Description |
|-------|-------|-------------|
| **0** | No DR | No signs of diabetic retinopathy |
| **1** | Mild | Microaneurysms only |
| **2** | Moderate | More than just microaneurysms but less than severe |
| **3** | Severe | Extensive hemorrhages, venous beading, IRMA |
| **4** | Proliferative DR | Neovascularization, vitreous/pre-retinal hemorrhage |

---

## 📊 Dataset

### Full Dataset (used for training — not included in repo due to size)

| Split | Class 0 | Class 1 | Class 2 | Class 3 | Class 4 | **Total** |
|-------|---------|---------|---------|---------|---------|-----------|
| Train | 5,968 | 5,968 | 5,968 | 5,968 | 5,968 | **29,840** |
| Val | 1,520 | 1,305 | 1,399 | 1,278 | 1,294 | **6,796** |
| Test | 2,650 | 1,430 | 1,965 | 1,280 | 1,366 | **8,691** |
| **Total** | | | | | | **45,327** |

- **Training split is perfectly balanced** — 5,968 samples per class (augmented/oversampled during preprocessing)
- Images stored as RGB JPEG/PNG, resized to **300×300** during loading
- Dataset has gone through full preprocessing: quality filtering, resizing, normalization

### Sample Dataset (included in this repo)

`sample_dataset/` contains **8 representative images per class** (40 total) for demonstration and quick testing purposes.

```
sample_dataset/
├── 0/   # No DR       — 8 images
├── 1/   # Mild        — 8 images
├── 2/   # Moderate    — 8 images
├── 3/   # Severe      — 8 images
└── 4/   # Proliferative DR — 8 images
```

---

## 🏗️ Model Architecture

```
Input (300 × 300 × 3)
        │
        ▼
preprocess_input()          ← EfficientNet per-channel normalisation (inside model)
        │
        ▼
Data Augmentation           ← RandomFlip | RandomRotation(0.08) | RandomZoom(0.10) | RandomContrast(0.10)
   (training only)             Disabled automatically during val / test / inference
        │
        ▼
EfficientNet-B3 Backbone    ← ImageNet pretrained weights | 10,783,535 params
  (Functional, no top)         Phase 1: Frozen | Phase 2: Top 30% unfrozen
        │
        ▼
GlobalAveragePooling2D      → (None, 1536)
        │
        ▼
Dropout(0.4)
        │
        ▼
Dense(256, ReLU) + BatchNormalization
        │
        ▼
Dropout(0.3)
        │
        ▼
Dense(5, Softmax)           → DR grade probabilities
```

### Two-Phase Training Strategy

| Phase | Backbone | Learning Rate | Purpose |
|-------|----------|---------------|---------|
| **Phase 1 — Head Training** | Frozen | 1e-3 | Train custom classifier on ImageNet features |
| **Phase 2 — Fine-Tuning** | Top ~30% unfrozen | 1e-5 | Adapt backbone to retinal image features |

---

## 📁 Project Structure

```
├── 01_Data_Cleaning_EDA.ipynb                   # EDA and data quality analysis
├── 02_Image_Preprocessing.ipynb                 # Preprocessing pipeline
├── 03_Data_Augmentation_And_Lodaing.ipynb       # Augmentation and dataset balancing
├── 04_EfficientNet-B3_Transfer_Learning.ipynb   # Initial model experiments
├── 05_EfficientNetB3_Colab_Training_Pipeline.ipynb  # ✅ MAIN: Full Colab pipeline
│
├── sample_dataset/                              # 40 sample images (8 per class)
│   ├── 0/    # No DR
│   ├── 1/    # Mild
│   ├── 2/    # Moderate
│   ├── 3/    # Severe
│   └── 4/    # Proliferative DR
│
└── README.md
```

> **Note:** The full `split_dataset/` (45,327 images, ~several GB) is excluded from this repo.  
> Upload it to Google Drive and point `DATASET_ROOT` in the config cell to its path.

---

## 📓 Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 01 | `01_Data_Cleaning_EDA.ipynb` | Exploratory data analysis, quality checks, class distribution |
| 02 | `02_Image_Preprocessing.ipynb` | Resizing, normalization, CLAHE, quality filtering |
| 03 | `03_Data_Augmentation_And_Lodaing.ipynb` | Augmentation strategy, dataset balancing |
| 04 | `04_EfficientNet-B3_Transfer_Learning.ipynb` | Initial transfer learning experiments |
| **05** | **`05_EfficientNetB3_Colab_Training_Pipeline.ipynb`** | **✅ Complete production training pipeline** |

---

## 🚀 Training Pipeline (Google Colab)

The main notebook `05_EfficientNetB3_Colab_Training_Pipeline.ipynb` covers all 17 steps:

```
1.  Environment Setup & GPU Verification
2.  Imports
3.  Configuration
4.  Dataset Loading         ← tf.data pipeline with prefetch + cache
5.  Dataset Validation      ← shape, dtype, value range checks
6.  Dataset Visualization   ← sample grid per class
7.  Model Creation          ← EfficientNet-B3 with custom head
8.  Model Compilation       ← Adam + categorical_crossentropy
9.  Phase 1 Training        ← frozen backbone, train head only
10. Phase 2 Fine-Tuning     ← partial unfreeze, LR=1e-5
11. Training Curves         ← accuracy & loss plots
12. Test Evaluation         ← loss, accuracy, precision, recall, F1
13. Confusion Matrix        ← count + row-normalised heatmaps
14. Error Analysis          ← per-class metrics + misclassified samples
15. Model Saving            ← best_model.keras + all artifacts to Drive
16. Inference Test          ← standalone predict_single_image() function
17. Final Results Summary
```

### Quick Start

```python
# 1. Open notebook in Colab
# 2. Set runtime: Runtime → Change runtime type → T4 GPU
# 3. Update dataset path in Cell 3 (Configuration):
DATASET_ROOT = "/content/drive/MyDrive/split_dataset"

# 4. Run all cells: Runtime → Run all
```

### Saved Artifacts

After training, these files are saved to Google Drive (`/efficientnet_b3_output/`):

| File | Description |
|------|-------------|
| `best_model.keras` | Best checkpoint (by val_accuracy) |
| `class_names.json` | Label mapping |
| `training_history.csv` | Epoch-by-epoch metrics |
| `test_metrics.json` | Final test scores |
| `confusion_matrix.png` | Visual confusion matrix |
| `training_curves.png` | Loss & accuracy plots |
| `error_analysis.png` | Misclassified sample grid |
| `per_class_metrics.png` | Per-class P/R/F1 bar charts |

---

## 📈 Results *(to be updated after training)*

> **Training is currently in progress on Google Colab T4 GPU.**  
> This section will be updated with actual evaluation metrics after the run completes.

| Metric | Value |
|--------|-------|
| Test Accuracy | `—` |
| Macro Precision | `—` |
| Macro Recall | `—` |
| Macro F1-Score | `—` |
| Weighted F1 | `—` |
| Best Val Accuracy | `—` |
| Most Confused Classes | `—` |

### Confusion Matrix
*To be added after training*

### Training Curves
*To be added after training*

---

## ⚙️ How to Run

### 1. Clone the repo
```bash
git clone https://github.com/alhaseebbb-a11y/Diabetic-Retinopathy-Detection-and-Severity-Grading-.git
cd Diabetic-Retinopathy-Detection-and-Severity-Grading-
```

### 2. Prepare the full dataset
- Upload your full preprocessed `split_dataset/` folder to **Google Drive**
- Structure must be:
```
split_dataset/
├── train/{0,1,2,3,4}/
├── val/{0,1,2,3,4}/
└── test/{0,1,2,3,4}/
```

### 3. Open the main notebook in Colab
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alhaseebbb-a11y/Diabetic-Retinopathy-Detection-and-Severity-Grading-/blob/main/05_EfficientNetB3_Colab_Training_Pipeline.ipynb)

### 4. Configure and run
- Set runtime to **T4 GPU**: Runtime → Change runtime type → T4 GPU
- Update `DATASET_ROOT` in the Configuration cell
- **Runtime → Run all**

### 5. Run inference on new images
```python
from tensorflow.keras.models import load_model
import json, numpy as np, tensorflow as tf

model      = load_model("best_model.keras")
class_names = json.load(open("class_names.json"))

def predict(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(300, 300))
    arr = tf.expand_dims(tf.keras.utils.img_to_array(img), 0)
    probs = model(arr, training=False).numpy()[0]
    idx   = np.argmax(probs)
    return {
        "grade"      : class_names[idx],
        "confidence" : f"{probs[idx]:.2%}",
        "all_probs"  : dict(zip(class_names, probs.tolist()))
    }

result = predict("path/to/fundus_image.jpg")
print(result)
# → {'grade': '2', 'confidence': '87.34%', 'all_probs': {...}}
```

---

## 📦 Requirements

```
tensorflow >= 2.12
numpy
matplotlib
seaborn
scikit-learn
Pillow
```

Install:
```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn Pillow
```

> In Google Colab all dependencies are pre-installed. The notebook runs `!pip install -q scikit-learn seaborn` automatically.

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [EfficientNet](https://arxiv.org/abs/1905.11946) — Tan & Le, 2019
- [Keras Applications](https://keras.io/api/applications/efficientnet/)
- Diabetic Retinopathy dataset (preprocessed from APTOS / Kaggle DR grading challenge)

---

<div align="center">
  <i>⭐ Star this repo if you find it useful! Results will be updated after training completes.</i>
</div>
