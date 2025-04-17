# SECTION 3: Hyperparameter Tuning

# Hyperparameter grid for Random Forest
rf_params = {
    'n_estimators': [100, 150],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2]
}

# GridSearchCV for Random Forest
rf_grid = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=rf_params,
    scoring='f1_weighted',
    cv=3,
    n_jobs=-1
)
rf_grid.fit(X_train_scaled, y_train)
rf_best = rf_grid.best_estimator_
print("Best Random Forest Parameters:", rf_grid.best_params_)

# Hyperparameter grid for Support Vector Machine (SVM)
svm_params = {
    'C': [1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

# GridSearchCV for SVM
svm_grid = GridSearchCV(
    estimator=SVC(probability=True, random_state=42),
    param_grid=svm_params,
    scoring='f1_weighted',
    cv=3,
    n_jobs=-1
)
svm_grid.fit(X_train_scaled, y_train)
svm_best = svm_grid.best_estimator_
print("Best SVM Parameters:", svm_grid.best_params_)

# Hyperparameter grid for Gradient Boosting
gb_params = {
    'n_estimators': [100, 150],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5]
}

# GridSearchCV for Gradient Boosting
gb_grid = GridSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_grid=gb_params,
    scoring='f1_weighted',
    cv=3,
    n_jobs=-1
)
gb_grid.fit(X_train_scaled, y_train)
gb_best = gb_grid.best_estimator_
print("Best Gradient Boosting Parameters:", gb_grid.best_params_)

# Predictions on the test set using the best models
rf_pred = rf_best.predict(X_test_scaled)
svm_pred = svm_best.predict(X_test_scaled)
gb_pred = gb_best.predict(X_test_scaled)

# SECTION 4: Deep Learning with EarlyStopping and Class Weights

print("Training Deep Neural Network with Class Weights...")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt

# One-hot encode labels for multi-class classification
y_train_cat = to_categorical(y_train, num_classes=5)
y_test_cat = to_categorical(y_test, num_classes=5)

# Compute class weights to handle imbalance
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# Build a lightweight MLP model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(5, activation='softmax')  # 5 output classes for CKD stages
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Configure EarlyStopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Train the model
history = model.fit(
    X_train_scaled, y_train_cat,
    validation_data=(X_test_scaled, y_test_cat),
    epochs=100,
    batch_size=16,
    class_weight=class_weight_dict,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate and make predictions
dl_probs = model.predict(X_test_scaled)
dl_pred = np.argmax(dl_probs, axis=1)

# Plot training accuracy and loss
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# SECTION 5: Evaluation Metrics

print("\n Model Performance Comparison:")

# Store predictions from all models
model_predictions = {
    "Random Forest": rf_pred,
    "Support Vector Machine": svm_pred,
    "Gradient Boosting": gb_pred,
    "Deep Neural Network": dl_pred
}

# Initialize results list
results = []

# Compute evaluation metrics for each model
for model_name, y_pred in model_predictions.items():
    results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred, average='weighted'),
        "Precision": precision_score(y_test, y_pred, average='weighted'),
        "Recall": recall_score(y_test, y_pred, average='weighted')
    })

# Create results dataframe
results_df = pd.DataFrame(results)
print(results_df)

# Plot model performance comparison
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df.melt(id_vars='Model'),
            x='Model', y='value', hue='variable', palette='Set2')

plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.legend(title="Metric")
plt.grid(axis='y')
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# SECTION 6: Confusion Matrices

print("\n Confusion Matrices (Raw and Normalized):")

# Loop through each model to display its confusion matrix
for model_name, y_pred in model_predictions.items():
    # Raw confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Normalized confusion matrix (row-wise)
    cm_norm = confusion_matrix(y_test, y_pred, normalize='true')

    plt.figure(figsize=(12, 4))

    # Plot raw confusion matrix
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} - Confusion Matrix (Counts)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Plot normalized confusion matrix
    plt.subplot(1, 2, 2)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues')
    plt.title(f'{model_name} - Confusion Matrix (Normalized)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    plt.tight_layout()
    plt.show()

