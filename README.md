# CKD Multi-Stage Classification

This repository contains the full implementation of my MSc Data Science final project:

**Title**: *Multi-Stage Prediction of Chronic Kidney Disease Using Interpretable Machine Learning and Deep Learning Models: A Comparative Study*

---

## Project Overview

The aim of this project is to develop and evaluate predictive models that classify Chronic Kidney Disease (CKD) into multiple stages based on clinical and demographic features.  
The pipeline includes traditional machine learning models (Random Forest, SVM, Gradient Boosting) and a deep neural network, along with interpretability techniques using SHAP.

---

## Repository Contents

| File/Folder                  | Description |
|-------------------------------|-------------|
| [`CKD_final_version.ipynb`](https://github.com/Srikardayala/ckd-multistage-classification/blob/main/CKD_final_version.ipynb) | Final notebook containing all preprocessing, modeling, evaluation, and SHAP interpretability |
| `models.py`                   | Script containing training logic for ML models |
| `Data_Preprocessing_and_EDA.py` | Script with preprocessing, feature engineering, and EDA |
| `Feature_SHAP.py`              | SHAP interpretability code for Random Forest model |
| `README.md`                    | This file |
| `LICENSE`                      | MIT license file |

---

## Tools and Technologies Used

- Python 3.10
- NumPy, pandas, seaborn, matplotlib
- Scikit-learn
- TensorFlow/Keras
- SHAP (SHapley Additive exPlanations)

---

## Dataset

- **Source**: [Kaggle – Chronic Kidney Disease Dataset (by mansoordaku)](https://www.kaggle.com/datasets/mansoordaku/ckdisease)
- **Original Source**: [UCI Machine Learning Repository – CKD Dataset](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease)

### Notes on the Dataset:
- The version used is a **cleaned CSV file** downloaded from Kaggle.
- 400 patient records with 24 features + 1 class label.
- The target variable was redefined into **five CKD stages** based on serum creatinine thresholds.
- Missing values were imputed using **KNN imputation**.
- Categorical features were encoded using **label encoding**.

---

## Running the Code

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Srikardayala/ckd-multistage-classification.git
    cd ckd-multistage-classification
    ```

2. **Run the final notebook**:  
   Open [`CKD_final_version.ipynb`](https://github.com/Srikardayala/ckd-multistage-classification/blob/main/CKD_final_version.ipynb) in Jupyter Notebook or Google Colab.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
