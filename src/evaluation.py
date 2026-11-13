import joblib
import json
from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_auc_score,
    ConfusionMatrixDisplay
)


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "02_data.joblib"
MODEL_PATH = BASE_DIR / "models" / "best_svm_model.joblib"
REPORTS_DIR = BASE_DIR / "reports"


def evaluate_model():

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load test data
    data = joblib.load(DATA_PATH)

    x_test = data["x_test"]
    y_test = data["y_test"]

    # Load optimized model
    model = joblib.load(MODEL_PATH)

    # Predictions
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)

    # Calculate metrics
    metrics = {
        "accuracy": float(
            accuracy_score(y_test, y_pred)
        ),
        "precision": float(
            precision_score(
                y_test, y_pred,
                average="macro",
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_test, y_pred,
                average="macro",
                zero_division=0
            )
        ),
        "f1_score": float(
            f1_score(
                y_test, y_pred,
                average="macro",
                zero_division=0
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_test,
                y_proba,
                multi_class="ovr",
                average="macro"
            )
        )
    }

    print("Evaluation Metrics:")
    print(metrics)

    # Classification report
    report = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    print("\nClassification Report:")
    print(report)

    # Save metrics
    with open(REPORTS_DIR / "metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    # Save classification report
    with open(REPORTS_DIR / "classification_report.txt", "w") as file:
        file.write(report)

    # Confusion matrix
    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        cmap="Blues"
    )

    plt.title("Optimized SVM - Confusion Matrix")

    plt.savefig(
        REPORTS_DIR / "confusion_matrix.png",
        bbox_inches="tight"
    )

    plt.close()

    print("Evaluation completed.")


if __name__ == "__main__":
    evaluate_model()