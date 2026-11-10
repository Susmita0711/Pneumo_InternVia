# PneumoScan – Pneumonia Detection from Chest X‑rays

**Organization:** Apex Medical Imaging Solutions  
**Category:** Intermediate – AI / Machine Learning  
**Framework:** TensorFlow / Keras  
**Frontend:** Streamlit  

---

## Table of Contents
- [Project Overview](#project-overview)
- [Problem Statement & Objective](#problem-statement--objective)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation (Windows)](#installation-windows)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Running the Streamlit App](#running-the-streamlit-app)
- [Explanation of the Model](#explanation-of-the-model)
- [Medical Disclaimer](#medical-disclaimer)
- [Limitations & Future Work](#limitations--future-work)
- [Screenshots](#screenshots)

---

## Project Overview
PneumoScan is an **educational prototype** that demonstrates how a convolutional neural network (CNN) can be applied to chest radiographs to classify them as **NORMAL** or **PNEUMONIA**. The project includes a full data‑loading pipeline, training/evaluation scripts, a Grad‑CAM visualisation tool, and a modern Streamlit UI.

---

## Problem Statement & Objective
Pneumonia remains a leading cause of mortality, especially among children and the elderly. Early detection via chest X‑rays can accelerate treatment. The goal of this project is to showcase a reproducible end‑to‑end workflow for building a deep‑learning classifier on a publicly available dataset, while providing an interactive UI for exploring predictions.

---

## Features
- **Custom CNN** architecture (and optional MobileNetV2 transfer learning).
- **Data preprocessing** with OpenCV (resize to 224×224, RGB conversion, normalization).
- **Data augmentation** applied only to the training set (small rotations, zoom, translation, optional horizontal flip).
- **Training utilities** – early stopping, checkpointing, LR reduction, reproducible seeds.
- **Comprehensive evaluation** – accuracy, precision, recall (sensitivity), specificity, F1‑score, ROC‑AUC, confusion matrix, classification report.
- **Grad‑CAM** heat‑map visualisation for model attention.
- **Streamlit dashboard** with:
  - X‑ray upload & prediction.
  - Model performance plots.
  - Session‑based prediction history.
  - Project description & medical disclaimer.
- **Unit tests** for preprocessing and prediction error handling.

---

## Technology Stack
| Category | Library / Tool |
|---|---|
| Language | Python 3.10 |
| Deep Learning | TensorFlow 2.13, Keras |
| Image Processing | OpenCV, Pillow |
| Data Handling | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Evaluation | Scikit‑learn |
| UI | Streamlit |
| Testing | pytest |

---

## Dataset
- **Source:** [Chest X‑Ray Images (Pneumonia) – Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **License:** CC‑0 (public domain)
- **Structure expected by the code:**
```
data/
├─ train/
│   ├─ NORMAL/
│   └─ PNEUMONIA/
├─ val/
│   ├─ NORMAL/
│   └─ PNEUMONIA/
└─ test/
    ├─ NORMAL/
    └─ PNEUMONIA/
```
The `train/` set is used for model fitting, `val/` for early‑stopping & LR‑schedule, and `test/` remains untouched for final evaluation.

---

## Project Structure
```
PneumoScan/
├─ .gitignore
├─ requirements.txt
├─ README.md
├─ app.py
├─ assets/            # generated plots, placeholder logo
├─ data/              # *download and extract here*
├─ models/            # checkpoints will be saved here
├─ src/
│   ├─ __init__.py
│   ├─ preprocessing.py
│   ├─ dataset_loader.py
│   ├─ model.py
│   ├─ train.py
│   ├─ evaluate.py
│   ├─ predict.py
│   └─ gradcam.py
├─ pages/
│   ├─ 1_Xray_Analysis.py
│   ├─ 2_Model_Performance.py
│   ├─ 3_Prediction_History.py
│   └─ 4_About.py
└─ tests/
    ├─ test_preprocessing.py
    └─ test_prediction.py
```
---

## Installation (Windows)
1. **Clone / copy the repository** to a location of your choice (already done in the workspace).
2. **Create a virtual environment** and install dependencies:
```powershell
cd C:\Users\soumi\OneDrive\Pictures\SUSMITA HAZRA\Pneumo_InternVia\PneumoScan
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
3. **Download the dataset** from Kaggle and extract it so the `data/` folder follows the structure shown above.
4. (Optional) **Place a custom logo** in `assets/logo.png` – the placeholder can be kept.

---

## Training the Model
Run the training script. You can choose the custom CNN (`cnn`) or the MobileNetV2 transfer‑learning version (`mobilenet`).
```powershell
python src/train.py --epochs 20 --batch-size 32 --model cnn
```
*Key behaviours:* 
- Deterministic seed (`42`).
- EarlyStopping (patience 5) restores the best weights.
- ModelCheckpoint saves `models/best_cnn.h5` (or `best_mobilenetv2.h5`).
- `assets/training_history.csv` stores loss/accuracy per epoch for the UI.

---

## Evaluation
After training finishes, evaluate on the untouched test set:
```powershell
python src/evaluate.py --model cnn   # or --model mobilenet
```
The script prints a detailed metric table and creates:
- `assets/confusion_matrix.png`
- `assets/roc_curve.png`
- `assets/test_metrics.csv`
These assets are automatically displayed on the **Model Performance** page.

---

## Running the Streamlit App
```powershell
streamlit run app.py
```
The sidebar navigation lets you switch between:
- **Home / X‑ray Analysis** – upload an image, see the prediction, confidence, and optional Grad‑CAM heatmap.
- **Model Performance** – view training curves, test metrics, confusion matrix, ROC curve.
- **Prediction History** – session‑based list of previous predictions.
- **About** – project description, methodology, and medical disclaimer.

---

## Explanation of the Model
**Custom CNN (`PneumoCNN`)**
- 3 Conv2D blocks (32 → 64 → 128 filters) with BatchNorm and ReLU.
- Max‑Pooling after each block.
- GlobalAveragePooling → Dense(128) → Dropout(0.5) → Sigmoid output.
- Binary cross‑entropy loss, Adam optimizer.

**MobileNetV2 (optional)**
- ImageNet‑pretrained base, last 100 layers frozen by default.
- GlobalAveragePooling + Dropout(0.4) + Sigmoid head.
- Smaller learning rate (1e‑4).

Both models output a single probability; a threshold of 0.5 yields the class label.

---

## Medical Disclaimer
> **This application is an educational Deep Learning prototype and is not a medical diagnostic system.**
> Its predictions may be incorrect and must not replace professional radiological interpretation or medical advice. Never use this result to make treatment decisions or delay medical care.

---

## Limitations & Future Work
- Trained on a single public dataset; limited diversity of imaging equipment and patient populations.
- No external clinical validation – metrics are indicative only.
- Grad‑CAM is an approximate attention visualiser, not a validated diagnostic aid.
- **Future improvements** could include multi‑center data, probability calibration, advanced explainability (SHAP/LIME), secure deployment, and integration with a PACS system.

---

## Screenshots
*(Add screenshots of each Streamlit page here after first run – e.g., `assets/home_screenshot.png`)*

---

*Happy researching! If you need any UI theme tweaks, additional pages, or integration with a plugin, just let me know.*

