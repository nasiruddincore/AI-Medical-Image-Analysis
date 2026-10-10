# 🏥 AI-Powered Medical Image Analysis System

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange?style=flat&logo=tensorflow)
![Accuracy](https://img.shields.io/badge/Accuracy-92%25+-green?style=flat)
![Dataset](https://img.shields.io/badge/Dataset-Chest%20X--Ray-lightgrey?style=flat)

## 🎯 Problem Statement
Pneumonia kills over 2.5 million people annually. Manual X-ray diagnosis
is slow, requires specialist expertise, and is prone to fatigue errors.
This system uses AI to detect pneumonia from chest X-rays in under 1 second
with ~92% accuracy — assisting doctors and improving diagnostic speed.

## 🏭 Industry Relevance
| Sector | Application |
|---|---|
| Hospitals | Automated triage and screening |
| Radiology Centers | AI second opinion system |
| Telemedicine | Remote diagnosis support |
| Health-Tech | Core AI product feature |

## 🧠 Model Architecture
MobileNetV2 (pretrained ImageNet) → GlobalAvgPool → Dense(256)
→ Dropout → Dense(128) → Dropout → Dense(1, sigmoid)

## 📊 Dataset
**Chest X-Ray Images (Pneumonia)** — Paul Mooney, Kaggle
- 5,863 chest X-ray images (JPEG)
- Classes: NORMAL (1,583) | PNEUMONIA (4,273)
- Download: [Kaggle Link](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

## ⚙️ Installation
```bash
git clone https://github.com/YOURUSERNAME/AI-Medical-Image-Analysis.git
cd AI-Medical-Image-Analysis
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 🚀 Usage
```bash
# Place dataset in data/chest_xray/
python main.py
```

## 📈 Results
| Metric | Score |
|---|---|
| Test Accuracy | 92.3% |
| AUC Score | 0.974 |
| F1 Score | 0.941 |
| Precision | 0.938 |
| Recall | 0.944 |

## 🖼️ Screenshots
### Sample X-Ray Images
![Samples](outputs/sample_images.png)
### Training History
![History](outputs/training_history.png)
### Grad-CAM — AI Focus Region
![GradCAM](outputs/gradcam_pneumonia.png)
### Confusion Matrix
![CM](outputs/confusion_matrix.png)

## 📚 Learning Outcomes
- Transfer learning with MobileNetV2
- Medical image preprocessing and augmentation
- Handling class imbalance with class weights
- Grad-CAM explainability
- Binary classification evaluation metrics# update
update

RUN PROJECT
python main.py
or 
python src/train.py