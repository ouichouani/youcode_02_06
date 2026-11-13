import joblib
import json
from pathlib import Path

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "02_data.joblib"
MODEL_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"


def train_model():

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load preprocessed data
    data = joblib.load(DATA_PATH)

    x_train = data["x_train"]
    y_train = data["y_train"]

    # Initialize SVM
    svm = SVC(
        probability=True,
        random_state=100
    )

    # Hyperparameter grid
    param_grid = {
        "C": [0.1, 1, 10, 100],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto", 0.01, 0.1, 1],
        "class_weight": [None, "balanced"]
    }

    # GridSearchCV
    grid_search = GridSearchCV(
        estimator=svm,
        param_grid=param_grid,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1
    )

    # Train and optimize
    grid_search.fit(x_train, y_train)

    best_svm = grid_search.best_estimator_

    print("Best Parameters:", grid_search.best_params_)
    print("Best CV F1-score:", grid_search.best_score_)

    # Save optimized model
    joblib.dump(
        best_svm,
        MODEL_DIR / "best_svm_model.joblib"
    )

    # Save optimization results
    with open(REPORTS_DIR / "training_results.json", "w") as file:
        json.dump(
            {
                "best_params": grid_search.best_params_,
                "best_cv_f1": float(grid_search.best_score_)
            },
            file,
            indent=4
        )

    print("Training completed.")


if __name__ == "__main__":
    train_model()