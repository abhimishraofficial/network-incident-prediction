from src.data.ingestion import load_network_data
from src.models.data_preparation import prepare_ml_data
from src.models.feature_importance import (
    get_feature_importance
)
from src.models.model_utils import load_model


INPUT_FILE = (
    "data/processed/network_incident_dataset.csv"
)

MODEL_PATH = (
    "models/random_forest.joblib"
)


def main():

    print("=" * 60)
    print("NETWORK INCIDENT FEATURE IMPORTANCE")
    print("=" * 60)

    # ------------------------------------------
    # LOAD DATA
    # ------------------------------------------

    print("\nLoading dataset...")

    df = load_network_data(
        INPUT_FILE
    )

    # ------------------------------------------
    # PREPARE FEATURES
    # ------------------------------------------

    print(
        "\nPreparing feature names..."
    )

    X, y = prepare_ml_data(
        df
    )

    feature_names = list(
        X.columns
    )

    # ------------------------------------------
    # LOAD MODEL
    # ------------------------------------------

    print(
        "\nLoading trained Random Forest..."
    )

    model = load_model(
        MODEL_PATH
    )

    print(
        "Model loaded successfully."
    )

    # ------------------------------------------
    # FEATURE IMPORTANCE
    # ------------------------------------------

    importance_df = get_feature_importance(
        model,
        feature_names
    )

    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE RANKING")
    print("=" * 60)

    print(
        importance_df.to_string(
            index=False
        )
    )

    print("\n" + "=" * 60)
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("=" * 60)

    print(
        importance_df.head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()