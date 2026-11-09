import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("data/creditcard.csv")

# Display dataset information
print("Dataset loaded successfully!")

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nFraud Transaction Count:")
print(df["Class"].value_counts())

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nStatistical Summary:")
print(df.describe())


# Display both graphs in one window

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Graph 1: Transaction Class Distribution
sns.countplot(x="Class", data=df, ax=axes[0])

axes[0].set_title("Normal vs Fraudulent Transactions")
axes[0].set_xlabel("Transaction Class")
axes[0].set_ylabel("Number of Transactions")
axes[0].set_xticks([0, 1], labels=["Normal", "Fraud"])


# Graph 2: Transaction Amount Analysis
sns.boxplot(x="Class", y="Amount", data=df, ax=axes[1])

axes[1].set_title("Transaction Amount Analysis")
axes[1].set_xlabel("Transaction Class")
axes[1].set_ylabel("Transaction Amount")
axes[1].set_xticks([0, 1], labels=["Normal", "Fraud"])


# Display graphs
plt.tight_layout()
plt.show()
# Separate Features and Target

X = df.drop("Class", axis=1)
y = df["Class"]

print("\nFeatures Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)
from sklearn.model_selection import train_test_split

# Split dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

print("\nTraining Target Distribution:")
print(y_train.value_counts())

print("\nTesting Target Distribution:")
print(y_test.value_counts())
from sklearn.preprocessing import StandardScaler

# Feature Scaling

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nScaled Training Data Shape:")
print(X_train_scaled.shape)

print("\nScaled Testing Data Shape:")
print(X_test_scaled.shape)
from sklearn.linear_model import LogisticRegression

# Create Logistic Regression Model

model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

# Train the Model

model.fit(X_train_scaled, y_train)

print("\nLogistic Regression Model Trained Successfully!")
# Make Predictions

y_pred = model.predict(X_test_scaled)

print("\nFirst 10 Actual Values:")
print(y_test.iloc[:10].values)

print("\nFirst 10 Predicted Values:")
print(y_pred[:10])
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Model Evaluation

print("\nModel Evaluation Results")

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Precision:", precision_score(y_test, y_pred))

print("Recall:", recall_score(y_test, y_pred))

print("F1 Score:", f1_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# Confusion Matrix Visualization

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Fraud"],
    yticklabels=["Normal", "Fraud"]
)

plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()
plt.show()
from sklearn.ensemble import RandomForestClassifier

# Create Random Forest Model

rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

# Train the Model

rf_model.fit(X_train, y_train)

print("\nRandom Forest Model Trained Successfully!")

# Make Predictions

rf_pred = rf_model.predict(X_test)

# Evaluate Model

print("\nRandom Forest Model Evaluation")

print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred))
print("Recall:", recall_score(y_test, rf_pred))
print("F1 Score:", f1_score(y_test, rf_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))
# Model Performance Comparison

model_comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy": [
        accuracy_score(y_test, y_pred),
        accuracy_score(y_test, rf_pred)
    ],
    "Precision": [
        precision_score(y_test, y_pred),
        precision_score(y_test, rf_pred)
    ],
    "Recall": [
        recall_score(y_test, y_pred),
        recall_score(y_test, rf_pred)
    ],
    "F1 Score": [
        f1_score(y_test, y_pred),
        f1_score(y_test, rf_pred)
    ]
})

print("\nModel Performance Comparison:")
print(model_comparison)

# Plot Comparison Graph

model_comparison.set_index("Model").plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Machine Learning Model Performance Comparison")
plt.ylabel("Score")
plt.xlabel("ML Models")
plt.ylim(0, 1.1)
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
# Random Forest Threshold Tuning

y_prob = rf_model.predict_proba(X_test)[:, 1]

thresholds = [0.5, 0.4, 0.3, 0.2, 0.1]

print("\nThreshold Performance Comparison")

for threshold in thresholds:

    y_threshold_pred = (y_prob >= threshold).astype(int)

    print("\nThreshold:", threshold)

    print("Precision:", precision_score(y_test, y_threshold_pred))

    print("Recall:", recall_score(y_test, y_threshold_pred))

    print("F1 Score:", f1_score(y_test, y_threshold_pred))
    # Final Model Using Best Threshold

best_threshold = 0.3

final_pred = (y_prob >= best_threshold).astype(int)

print("\nFinal Random Forest Model Results")

print("Selected Threshold:", best_threshold)

print("Accuracy:", accuracy_score(y_test, final_pred))

print("Precision:", precision_score(y_test, final_pred))

print("Recall:", recall_score(y_test, final_pred))

print("F1 Score:", f1_score(y_test, final_pred))

# Final Confusion Matrix

final_cm = confusion_matrix(y_test, final_pred)

print("\nFinal Confusion Matrix:")
print(final_cm)

# Confusion Matrix Graph

plt.figure(figsize=(8, 6))

sns.heatmap(
    final_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Fraud"],
    yticklabels=["Normal", "Fraud"]
)

plt.title("Final Random Forest Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.show()
import joblib

# Save Final Model

model_artifacts = {
    "model": rf_model,
    "threshold": best_threshold,
    "feature_names": list(X_train.columns)
}

joblib.dump(
    model_artifacts,
    "models/fraud_detection_model.pkl"
)

print("\nFinal Model Saved Successfully!")