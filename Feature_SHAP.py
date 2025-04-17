# SECTION 7: Feature Importances

print("\n Feature Importances from Tree-Based Models:")

# Zip model names and model objects
for model_name, model in zip(["Random Forest", "Gradient Boosting"], [rf_best, gb_best]):
    # Ensure the model has feature_importances_ attribute
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_names = X.columns
        indices = np.argsort(importances)[::-1]  # Descending order

        # Create a sorted dataframe for easy viewing
        sorted_features = pd.DataFrame({
            'Feature': feature_names[indices],
            'Importance': importances[indices]
        })

        print(f"\nTop Features - {model_name}:\n", sorted_features.head(10))

        # Plot the feature importances
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=sorted_features.head(10), palette='viridis')
        plt.title(f"Top 10 Important Features - {model_name}")
        plt.xlabel("Importance Score")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.show()

print("\nSHAP Summary Plot for Random Forest")

# Fit a simpler model for interpretability
rf_shap = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_shap.fit(X_train, y_train)

# Create test DataFrame
X_test_df = pd.DataFrame(X_test, columns=X.columns)

# TreeExplainer for multiclass
explainer = shap.TreeExplainer(rf_shap)
shap_values = explainer.shap_values(X_test_df)  # shape: (samples, features, classes)

# Combine across classes
shap_values_combined = np.mean(np.abs(shap_values), axis=2)

# Confirm matching shape
print("Combined SHAP shape:", shap_values_combined.shape)
print("X_test_df shape:", X_test_df.shape)

# Plot summary
shap.summary_plot(shap_values_combined, X_test_df, plot_type='bar')

# SECTION 9: Cross-Validation Scores

print("\n 5-Fold Cross-Validation Accuracy (on Training Set):")

# Define models and labels
cv_models = {
    "Random Forest": rf_best,
    "Gradient Boosting": gb_best,
    "Support Vector Machine": svm_best
}

# Compute and display mean ± std of accuracy for each model
for model_name, model in cv_models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"{model_name}: {scores.mean():.4f} ± {scores.std():.4f}")


