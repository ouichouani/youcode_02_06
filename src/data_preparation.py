import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def prepare_data():

    data = pd.read_csv(
        BASE_DIR / "data" / "student_mental_health.csv"
    )

    data = data.drop(columns=["Student_ID"])

    joblib.dump(
        data,
        BASE_DIR / "data" / "01_data.joblib"
    )

    print("Data preparation completed.")


if __name__ == "__main__":
    prepare_data()