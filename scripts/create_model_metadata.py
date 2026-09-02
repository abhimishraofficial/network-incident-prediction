from src.data.ingestion import load_network_data
from src.models.data_preparation import prepare_ml_data
from src.models.train_test_split import split_train_test
from src.models.model_utils import load_model
from src.models.evaluate_model import evaluate_model
from src.models.model_metadata import save_model_metadata


INPUT_FILE = (
    "data/processed/network_incident_dataset.csv"
)

MODEL_PATH = (
    "models/random_forest.joblib"
)

METADATA_PATH = (
    "models/model_metadata.json"
)

PRODUCTION_THRESHOLD = 0.20


def main():

    print("=" * 60)
    print("MODEL METADATA GENERATION")
    print("=" * 60)

    # ------------------------------------------
    # LOAD DATA
    # ------------------------------------------

    print("\n[1/5] Loading dataset...")

    df = load_network_data(
        INPUT_FILE
    )

    print(
        f"Records loaded: {len(df)}"
    )

    # ------------------------------------------
    # PREPARE ML DATA
    # ------------------------------------------

    print("\n[2/5] Preparing ML data...")

    X, y = prepare_ml_data(
        df
    )

    feature_names = list(
        X.columns
    )

    print(
        f"Features found: {len(feature_names)}"
    )

    # ------------------------------------------
    # SPLIT DATA
    # ------------------------------------------

    print("\n[3/5] Creating test dataset...")

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_train_test(
        X,
        y
    )

    print(
        f"Testing samples: {len(X_test)}"
    )

    # ------------------------------------------
    # LOAD MODEL
    # ------------------------------------------

    print("\n[4/5] Loading model...")

    model = load_model(
        MODEL_PATH
    )

    print(
        "Model loaded successfully."
    )

    # ------------------------------------------
    # EVALUATE MODEL
    # ------------------------------------------

    print(
        "\n[5/5] Evaluating model..."
    )

    metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    # ------------------------------------------
    # SAVE METADATA
    # ------------------------------------------

    save_model_metadata(
        model_name="Random Forest",
        model_version="1.0.0",
        threshold=PRODUCTION_THRESHOLD,
        metrics=metrics,
        feature_names=feature_names,
        output_path=METADATA_PATH
    )

    print("\n" + "=" * 60)
    print("MODEL METADATA SAVED")
    print("=" * 60)

    print(
        f"\nModel: Random Forest"
    )

    print(
        f"Version: 1.0.0"
    )

    print(
        f"Production threshold: "
        f"{PRODUCTION_THRESHOLD}"
    )

    print(
        f"\nMetadata saved to: "
        f"{METADATA_PATH}"
    )


if __name__ == "__main__":
    main()