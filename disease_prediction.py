"""
Disease Prediction from Medical Data
CodeAlpha Machine Learning Internship - Task 4

Dataset:
Breast Cancer Wisconsin Diagnostic dataset from scikit-learn.

Target:
Diagnosis - Malignant or Benign
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


DATASET_PATH = "dataset.csv"
TARGET_COLUMN = "Diagnosis"


def load_and_clean_data():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            "dataset.csv not found. Keep dataset.csv in the same folder as this script."
        )

    df = pd.read_csv(DATASET_PATH)
    df = df.drop_duplicates()

    for column in df.columns:
        if df[column].dtype == "object":
            df[column] = df[column].fillna(df[column].mode()[0])
        else:
            df[column] = df[column].fillna(df[column].median())

    return df


def save_accuracy_chart(results_df):
    os.makedirs("images", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.bar(results_df["Model"], results_df["Accuracy"])
    plt.title("Model Accuracy Comparison")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig("images/accuracy_comparison.png", dpi=200)
    plt.close()


def save_confusion_matrix(y_test, y_pred, model_name):
    os.makedirs("images", exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.colorbar()
    plt.xticks([0, 1], ["Malignant", "Benign"], rotation=20)
    plt.yticks([0, 1], ["Malignant", "Benign"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("images/confusion_matrix.png", dpi=200)
    plt.close()


def save_roc_curve(y_test, y_probability, model_name):
    os.makedirs("images", exist_ok=True)
    fpr, tpr, _ = roc_curve(y_test, y_probability)
    auc_score = roc_auc_score(y_test, y_probability)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig("images/roc_curve.png", dpi=200)
    plt.close()

    return auc_score


def save_feature_importance(model, feature_names):
    os.makedirs("images", exist_ok=True)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False).head(15)

    plt.figure(figsize=(10, 7))
    plt.barh(importance_df["Feature"], importance_df["Importance"])
    plt.title("Top 15 Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("images/feature_importance.png", dpi=200)
    plt.close()


def main():
    print("Loading dataset...")
    df = load_and_clean_data()

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found.")

    print("Dataset shape:", df.shape)
    print(df.head())

    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN].map({"Malignant": 0, "Benign": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
    }

    results = []

    for model_name, model in models.items():
        if model_name in ["Logistic Regression", "SVM"]:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_probability = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_probability = model.predict_proba(X_test)[:, 1]

        results.append({
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_probability)
        })

    results_df = pd.DataFrame(results)
    print("\nModel comparison:")
    print(results_df)

    best_model_name = results_df.sort_values("Accuracy", ascending=False).iloc[0]["Model"]
    best_model = models[best_model_name]

    if best_model_name in ["Logistic Regression", "SVM"]:
        y_pred_best = best_model.predict(X_test_scaled)
        y_probability_best = best_model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_pred_best = best_model.predict(X_test)
        y_probability_best = best_model.predict_proba(X_test)[:, 1]

    print("\nBest Model:", best_model_name)
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred_best, target_names=["Malignant", "Benign"]
    ))

    save_accuracy_chart(results_df)
    save_confusion_matrix(y_test, y_pred_best, best_model_name)
    save_roc_curve(y_test, y_probability_best, best_model_name)
    save_feature_importance(best_model, X.columns)

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/disease_prediction_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(list(X.columns), "models/feature_names.pkl")

    print("\nFiles saved successfully.")
    print("Model: models/disease_prediction_model.pkl")
    print("Scaler: models/scaler.pkl")
    print("Features: models/feature_names.pkl")
    print("Graphs: images/")


if __name__ == "__main__":
    main()
