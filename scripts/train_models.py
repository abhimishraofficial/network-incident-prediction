from src.data.ingestion import load_network_data
from src.models.data_preparation import prepare_ml_data
from src.models.evaluate_model import evaluate_model
from src.models.model_utils import save_model
from src.models.train_model import train_models
from src.models.train_test_split import split_train_test


INPUT_FILE = (
    "data/processed/network_incident_dataset.csv"
)

MODEL_DIRECTORY = "models"


def main():

    print("=" * 60)
    print("NETWORK INCIDENT MODEL TRAINING")
    print("=" * 60)

    # ------------------------------------------
    # STEP 1: LOAD DATA
    # ------------------------------------------

    print("\n[1/5] Loading processed dataset...")

    df = load_network_data(
        INPUT_FILE
    )

    print(
        f"Records loaded: {len(df)}"
    )

    # ------------------------------------------
    # STEP 2: PREPARE ML DATA
    # ------------------------------------------

    print("\n[2/5] Preparing ML data...")

    X, y = prepare_ml_data(
        df
    )

    print(
        f"Total ML samples: {len(X)}"
    )

    # ------------------------------------------
    # STEP 3: SPLIT DATA
    # ------------------------------------------

    print("\n[3/5] Splitting train/test data...")

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
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )

    # ------------------------------------------
    # STEP 4: TRAIN MODELS
    # ------------------------------------------

    print("\n[4/5] Training models...")

    trained_models = train_models(
        X_train,
        y_train
    )

    # ------------------------------------------
    # STEP 5: EVALUATE MODELS
    # ------------------------------------------

    print("\n[5/5] Evaluating models...")

    results = {}

    for model_name, model in trained_models.items():

        print("\n" + "-" * 60)
        print(model_name)
        print("-" * 60)

        metrics = evaluate_model(
            model,
            X_test,
            y_test
        )

        results[model_name] = metrics

        for metric_name, value in metrics.items():

            if isinstance(
                value,
                float
            ):
                print(
                    f"{metric_name}: {value:.4f}"
                )
            else:
                print(
                    f"{metric_name}: {value}"
                )

    # ------------------------------------------
    # SELECT BEST MODEL
    # ------------------------------------------

    best_model_name = max(
        results,
        key=lambda name: results[name]["f1_score"]
    )

    best_model = trained_models[
        best_model_name
    ]

    best_metrics = results[
        best_model_name
    ]

    print("\n" + "=" * 60)
    print("BEST MODEL")
    print("=" * 60)

    print(
        f"\nModel: {best_model_name}"
    )

    print(
        f"F1 Score: "
        f"{best_metrics['f1_score']:.4f}"
    )

    print(
        f"Recall: "
        f"{best_metrics['recall']:.4f}"
    )

    print(
        f"PR-AUC: "
        f"{best_metrics['pr_auc']:.4f}"
    )

    # ------------------------------------------
    # SAVE BEST MODEL
    # ------------------------------------------

    model_filename = (
        best_model_name
        .lower()
        .replace(" ", "_")
        + ".joblib"
    )

    model_path = (
        f"{MODEL_DIRECTORY}/{model_filename}"
    )

    print("\nSaving best model...")

    save_model(
        best_model,
        model_path
    )

    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()