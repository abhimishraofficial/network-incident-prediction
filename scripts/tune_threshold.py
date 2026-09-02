from src.data.ingestion import load_network_data
from src.models.data_preparation import prepare_ml_data
from src.models.model_utils import load_model
from src.models.threshold_tuning import (
    tune_threshold,
    get_best_threshold
)
from src.models.train_test_split import split_train_test


INPUT_FILE = (
    "data/processed/network_incident_dataset.csv"
)

MODEL_PATH = (
    "models/random_forest.joblib"
)


def main():

    print("=" * 60)
    print("NETWORK INCIDENT THRESHOLD TUNING")
    print("=" * 60)

    # ------------------------------------------
    # LOAD DATA
    # ------------------------------------------

    print("\nLoading dataset...")

    df = load_network_data(
        INPUT_FILE
    )

    print(
        f"Records loaded: {len(df)}"
    )

    # ------------------------------------------
    # PREPARE ML DATA
    # ------------------------------------------

    print("\nPreparing ML data...")

    X, y = prepare_ml_data(
        df
    )

    # ------------------------------------------
    # SPLIT DATA
    # ------------------------------------------

    print("\nCreating test dataset...")

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

    print(
        f"Actual incidents: {y_test.sum()}"
    )

    # ------------------------------------------
    # LOAD MODEL
    # ------------------------------------------

    print("\nLoading trained model...")

    model = load_model(
        MODEL_PATH
    )

    print("Model loaded successfully.")

    # ------------------------------------------
    # TUNE THRESHOLD
    # ------------------------------------------

    print("\nTesting thresholds...")

    results_df = tune_threshold(
        model,
        X_test,
        y_test
    )

    print("\n" + "=" * 90)
    print("THRESHOLD RESULTS")
    print("=" * 90)

    print(
        results_df.to_string(
            index=False
        )
    )

    # ------------------------------------------
    # BEST THRESHOLD
    # ------------------------------------------

    best = get_best_threshold(
        results_df
    )

    print("\n" + "=" * 60)
    print("BEST THRESHOLD")
    print("=" * 60)

    print(
        f"Threshold: "
        f"{best['threshold']:.2f}"
    )

    print(
        f"Precision: "
        f"{best['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{best['recall']:.4f}"
    )

    print(
        f"F1 Score: "
        f"{best['f1_score']:.4f}"
    )

    print(
        f"False Positives: "
        f"{int(best['false_positive'])}"
    )

    print(
        f"False Negatives: "
        f"{int(best['false_negative'])}"
    )


if __name__ == "__main__":
    main()