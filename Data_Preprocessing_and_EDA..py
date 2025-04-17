# SECTION 1 : Importing Libraries

# Standard Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings

# Suppress warnings for clearer output
warnings.filterwarnings("ignore")

# Scikit-learn modules for data preparation and modeling
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc
)

# TensorFlow/Keras modules for building neural networks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Set Seaborn style for consistent plots
sns.set(style="whitegrid")

# SECTION 2: Data Preparation & EDA

# Load dataset
file_path = '/content/drive/MyDrive/kidney_disease.csv'
df = pd.read_csv(file_path)
print(f"Original dataset shape: {df.shape}")

# Data Cleaning: Replace invalid entries with NaN
df.replace(to_replace=r'\t\?', value=np.nan, regex=True, inplace=True)
df.replace('?', np.nan, inplace=True)
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Convert numeric columns from object to numeric
numeric_columns = ['age', 'bp', 'sg', 'al', 'su', 'pcv', 'wc', 'rc', 'sc']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Encode categorical features using LabelEncoder
categorical_features = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
label_enc = LabelEncoder()
for col in categorical_features:
    df[col] = label_enc.fit_transform(df[col].astype(str))  # convert all to string first to handle NaNs

# Impute missing values using KNN Imputer
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)

# Drop non-numeric / irrelevant columns BEFORE imputation
df_clean = df.drop(columns=['id', 'classification'], errors='ignore')

# Apply KNN Imputer on numeric + encoded categorical data
imputer = KNNImputer(n_neighbors=5)
df_imputed_array = imputer.fit_transform(df_clean)
df_imputed = pd.DataFrame(df_imputed_array, columns=df_clean.columns)

# Define CKD stages based on serum creatinine (sc)
def assign_ckd_stage(sc):
    if sc < 1.5:
        return 1
    elif 1.5 <= sc < 2.0:
        return 2
    elif 2.0 <= sc < 3.5:
        return 3
    elif 3.5 <= sc < 5.0:
        return 4
    else:
        return 5

# Apply CKD stage assignment
df_imputed['ckd_stage'] = df_imputed['sc'].apply(assign_ckd_stage)
print("CKD stage distribution:\n", df_imputed['ckd_stage'].value_counts())

# Encode target variable
y = label_enc.fit_transform(df_imputed['ckd_stage'])

# Drop irrelevant columns and prepare features
X = df_imputed.drop(columns=['id', 'classification', 'ckd_stage'], errors='ignore')

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Feature scaling using StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Exploratory Data Analysis (EDA) Plots ---

import matplotlib.pyplot as plt
import seaborn as sns

# Grid of Histograms/KDE plots for numeric columns
num_cols = 3
num_rows = (len(numeric_columns) + num_cols - 1) // num_cols
fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 12))
axes = axes.flatten()

for i, col in enumerate(numeric_columns):
    sns.histplot(df_imputed[col], kde=True, color='steelblue', bins=30, ax=axes[i])
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Count')
    axes[i].grid(True)

# Remove unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# Grid of Box Plots: Numeric Features vs CKD Stage
num_cols = 3
num_rows = (len(numeric_columns) + num_cols - 1) // num_cols
fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 12))
axes = axes.flatten()

for i, col in enumerate(numeric_columns):
    sns.boxplot(x='ckd_stage', y=col, data=df_imputed, palette='Set2', ax=axes[i])
    axes[i].set_title(f'{col} vs CKD Stage')
    axes[i].set_xlabel('CKD Stage')
    axes[i].set_ylabel(col)
    axes[i].grid(True)

# Remove unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# CKD stage distribution bar chart with percentages
plt.figure(figsize=(7, 5))
stage_counts = df_imputed['ckd_stage'].value_counts().sort_index()
stage_labels = stage_counts.index
stage_values = stage_counts.values
stage_perc = (stage_values / stage_values.sum()) * 100

sns.barplot(
    x=stage_labels,
    y=stage_values,
    palette='pastel',
    edgecolor='gray',
    linewidth=1.5
)

for i, val in enumerate(stage_values):
    plt.text(i, val + 2, f'{stage_perc[i]:.1f}%', ha='center', va='bottom',
             fontsize=11, fontweight='bold', color='dimgray')

plt.title('CKD Stage Distribution', fontsize=16, fontweight='bold')
plt.xlabel('CKD Stage', fontsize=12)
plt.ylabel('Number of Patients', fontsize=12)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
