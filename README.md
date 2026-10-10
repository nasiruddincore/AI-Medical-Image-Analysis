# 🩺 AI-Powered Medical Image Analysis System

![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![UI](https://img.shields.io/badge/UI-Streamlit-FF4B4B)

An industry-grade Deep Learning pipeline that automates the classification of medical radiographs using MobileNetV2 Transfer Learning and provides transparent decision-support via Grad-CAM explainable AI.

---

## 📋 Table of Contents
1. [Overview & Healthcare Relevance](#-overview--healthcare-relevance)
2. [Problem Statement](#-problem-statement)
3. [System Objectives](#-system-objectives)
4. [Architecture & Data Flow](#-architecture--data-flow)
5. [Tech Stack](#-tech-stack)
6. [Folder Structure](#-folder-structure)
7. [Installation & Setup Guide](#-installation--setup-guide)
8. [Usage Instructions](#-usage-instructions)
9. [Evaluation Metrics & Results](#-evaluation-metrics--results)
10. [Author & License](#-author--license)

---

## 📌 Overview & Healthcare Relevance
Medical imaging is a cornerstone of modern diagnostics, but high workloads often lead to fatigue-induced errors and delayed reporting. This project models an enterprise health-tech system that processes chest X-rays to detect abnormalities like pneumonia instantly, serving as a reliable clinical second opinion.

---

## ⚠️ Problem Statement
* **High Diagnostic Workloads:** Radiologists face massive queues of scans daily, increasing turnaround times.
* **Interpretation Fatigue:** Manual inspection leaves room for human error in subtle anomaly detection.
* **Lack of Transparency:** "Black-box" AI models are often distrusted in clinical settings without visual explainability.

---

## 🎯 System Objectives
1. **Automated Classification:** Rapidly categorize chest X-rays as Normal or Pneumonia using deep learning.
2. **Transfer Learning Optimization:** Utilize MobileNetV2 to achieve high accuracy without heavy GPU infrastructure.
3. **Clinical Decision Support:** Deploy an interactive Streamlit web interface for real-time inference and visualization.
4. **Explainable AI Integration:** Use Grad-CAM to highlight specific regions influencing the model's prediction.

---

## 📐 Architecture & Data Flow
```text
[ Raw Chest X-Ray Scan ] ──► [ Preprocessing & Resize (224x224) ]
                                          │
                                          ▼
[ Streamlit Clinical UI ] ◄──(Confidence Score & Grad-CAM)── [ MobileNetV2 Feature Extraction ]
