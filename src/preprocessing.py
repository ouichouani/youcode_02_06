import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
    LabelEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


BASE_DIR = Path(__file__).resolve().parent.parent


def preprocess_data():

    # Load prepared data
    data = joblib.load(
        BASE_DIR / "data" / "01_data.joblib"
    )

    # Separate features and target
    target = data["Mental_Health_Risk"]

    X = data.drop(
        columns=["Student_ID", "Age", "Mental_Health_Risk"],
        errors="ignore"
    )

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        target,
        test_size=0.2,
        stratify=target,
        random_state=100
    )

    # Feature types
    nominal_features = [
        "Gender",
        "Part_Time_Job",
        "Previous_Mental_Health_History"
    ]

    ordinal_features = [
        "Year_of_Study",
        "Physical_Activity",
        "Diet_Quality"
    ]

    numerical_features = [
        "Sleep_Hours",
        "Study_Hours_Per_Week",
        "GPA",
        "Social_Media_Hours",
        "Screen_Time_Hours",
        "Financial_Stress",
        "Family_Support",
        "Attendance_Rate"
    ]

    ordinal_categories = [
        [1, 2, 3, 4, 5],
        ["Low", "Moderate", "High"],
        ["Poor", "Average", "Good"]
    ]

    # Nominal preprocessing
    preprocessing_nominal = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Ordinal preprocessing
    preprocessing_ordinal = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=ordinal_categories,
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ))
    ])

    # Numerical preprocessing
    preprocessing_num = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Column transformer
    preprocessor = ColumnTransformer([
        ("nominal", preprocessing_nominal, nominal_features),
        ("ordinal", preprocessing_ordinal, ordinal_features),
        ("numerical", preprocessing_num, numerical_features)
    ])

    # Fit and transform features
    x_train_processed = preprocessor.fit_transform(x_train)
    x_test_processed = preprocessor.transform(x_test)

    # Encode target
    target_encoder = LabelEncoder()

    y_train_processed = target_encoder.fit_transform(y_train)
    y_test_processed = target_encoder.transform(y_test)

    # Save processed data and fitted transformers
    joblib.dump(
        {
            "x_train": x_train_processed,
            "x_test": x_test_processed,
            "y_train": y_train_processed,
            "y_test": y_test_processed,
            "preprocessor": preprocessor,
            "target_encoder": target_encoder
        },
        BASE_DIR / "data" / "02_data.joblib"
    )

    print("Preprocessing completed.")


if __name__ == "__main__":
    preprocess_data()