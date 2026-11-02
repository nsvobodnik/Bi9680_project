import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

try:
    df_full = pd.read_csv("train.csv")
    sample_eval = df_full.sample(n=10, random_state=42)
    X_sample = sample_eval.drop(columns=["label"])
    y_true = sample_eval["label"]
except FileNotFoundError:
    print("Error: 'train.csv' not found. Ensure the dataset is present in the working directory.")
    raise

try:
    model = joblib.load("model.joblib")
    y_pred = model.predict(X_sample)
except FileNotFoundError:
    print("Error: 'model.joblib' not found. Ensure your pipeline is trained and saved.")
    raise

print("=" * 55)
print("             PROJECT MODEL EVALUATION             ")
print("=" * 55)
print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}  (Required: >= 0.84)")
print(f"MCC:       {matthews_corrcoef(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
print(f"F1-Score:  {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
print("-" * 55)
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print("\nDetailed Classification Report:")
print(classification_report(y_true, y_pred, zero_division=0))
print("=" * 55)
