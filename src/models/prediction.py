import json
from pathlib import Path

import joblib
import pandas as pd


# Get project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Model and metadata paths
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest.joblib"

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

    model = joblib.load(MODEL_PATH)

    return model


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
        "r"
    ) as file:
        metadata = json.load(file)

    return metadata


def validate_features(
    features: pd.DataFrame,
    expected_features: list
):
    """
    Validate that prediction input contains
    the expected ML features.
    """

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

    # Keep only expected features
    # and preserve training column order
    features = features[expected_features]

    return features


def predict_incident(
    features: pd.DataFrame
):
    """
    Predict the probability of a future
    network incident.

    Returns prediction probability,
    threshold, prediction and status.
    """

    # Load trained model
    model = load_model()

    # Load model metadata
    metadata = load_metadata()

    # Get expected feature names
    expected_features = metadata["feature_names"]

    # Validate and order features
    features = validate_features(
        features,
        expected_features
    )

    # Predict incident probability
    probability = model.predict_proba(
        features
    )[0][1]

    # Get production threshold
    threshold = metadata[
        "production_threshold"
    ]

    # Final prediction
    prediction = int(
        probability >= threshold
    )

    # Return result
    return {
        "incident_probability": round(
            float(probability),
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