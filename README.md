# CKD Multi-Stage Classification

This repository contains the full implementation of my MSc Data Science final project:

**Title**: *Multi-Stage Prediction of Chronic Kidney Disease Using Interpretable Machine Learning and Deep Learning Models: A Comparative Study*

---

## Project Overview

The aim of this project is to develop and evaluate predictive models that classify Chronic Kidney Disease (CKD) into multiple stages based on clinical and demographic features. The pipeline includes traditional machine learning models (Random Forest, SVM, Gradient Boosting) and a deep neural network, along with interpretability techniques using SHAP.

---

## Repository Contents

| File/Folder                            | Description |
|----------------------------------------|-------------|
| `CKD_MultiStage_Classification_Final.ipynb` | Complete notebook with all preprocessing, modeling, and evaluation steps |
| `models.py`                            | Script containing training logic for ML models |
| `Data_Preprocessing_and_EDA.py`       | Script with preprocessing, feature engineering, and EDA |
| `Feature_SHAP.py`                      | SHAP interpretability code for Random Forest |
| `data/`                                | Folder for dataset (not included here due to privacy policies) |
| `README.md`                            | This file |
| `LICENSE`                              | MIT license file |

---

## Tools and Technologies Used

- Python 3.10
- NumPy, pandas, seaborn, matplotlib
- Scikit-learn
- TensorFlow/Keras
- SHAP (SHapley Additive exPlanations)

---

## Dataset

- **Source**: [UCI Machine Learning Repository – Chronic Kidney Disease Dataset](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease)
- **Description**: The dataset contains clinical and biochemical features of CKD patients and was used to build classification models for CKD stage prediction.

---

## Running the Code

1. Clone the repository:
   ```bash
   git clone https://github.com/Srikardayala/ckd-multistage-classification.git
   cd ckd-multistage-classification
