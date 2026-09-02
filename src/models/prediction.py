import json
from pathlib import Path

import joblib
import pandas as pd


# Get project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Model and metadata paths
MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "random_forest.joblib"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "models"
    / "model_metadata.json"
)


def load_model():
    """
    Load the trained Random Forest model.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def load_metadata():
    """
    Load model metadata.
    """

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_PATH}"
        )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def validate_features(
    features: pd.DataFrame,
    expected_features: list
):
    """
    Validate that prediction input contains
    all expected ML features and preserve
    the training feature order.
    """

    if not isinstance(features, pd.DataFrame):
        raise TypeError(
            "Features must be a pandas DataFrame."
        )

    missing_features = [
        feature
        for feature in expected_features
        if feature not in features.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing required features: "
            f"{missing_features}"
        )

    return features[expected_features]


def predict_incident(
    features: pd.DataFrame
):
    """
    Predict the probability of a future
    network incident.

    Returns:
        incident probability,
        production threshold,
        prediction,
        and status.
    """

    # Load trained model
    model = load_model()

    # Load model metadata
    metadata = load_metadata()

    # Validate metadata
    if "feature_names" not in metadata:
        raise ValueError(
            "feature_names not found in model metadata."
        )

    # Get expected feature names
    expected_features = metadata[
        "feature_names"
    ]

    # Get production threshold
    threshold = float(
        metadata.get(
            "production_threshold",
            0.05
        )
    )

    # Validate threshold
    if not 0 <= threshold <= 1:
        raise ValueError(
            "production_threshold must be "
            "between 0 and 1."
        )

    # Validate and order features
    features = validate_features(
        features,
        expected_features
    )

    # Predict incident probability
    probability = float(
        model.predict_proba(features)[0][1]
    )

    # Apply production threshold
    prediction = int(
        probability >= threshold
    )

    # Return API response
    return {
        "incident_probability": round(
            probability,
            4
        ),
        "production_threshold": threshold,
        "prediction": prediction,
        "status": (
            "INCIDENT RISK"
            if prediction == 1
            else "NORMAL"
        )
    }


if __name__ == "__main__":
    print(
        "prediction.py loaded successfully"
    )