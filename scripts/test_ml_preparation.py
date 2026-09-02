from src.data.ingestion import load_network_data
from src.models.data_preparation import prepare_ml_data
from src.models.train_test_split import split_train_test


INPUT_FILE = (
    "data/processed/network_incident_dataset.csv"
)


def main():

    print("=" * 60)
    print("ML DATA PREPARATION TEST")
    print("=" * 60)

    # ------------------------------------------
    # LOAD DATA
    # ------------------------------------------

    print("\nLoading processed dataset...")

    df = load_network_data(
        INPUT_FILE
    )

    print(
        f"Records loaded: {len(df)}"
    )

    # ------------------------------------------
    # PREPARE DATA
    # ------------------------------------------

    print("\nPreparing ML features and target...")

    X, y = prepare_ml_data(
        df
    )

    print("ML data prepared successfully.")

    # ------------------------------------------
    # SPLIT DATA
    # ------------------------------------------

    print("\nSplitting train and test data...")

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_train_test(
        X,
        y
    )

    # ------------------------------------------
    # RESULTS
    # ------------------------------------------

    print("\n" + "=" * 60)
    print("ML DATASET SUMMARY")
    print("=" * 60)

    print(f"\nTotal samples: {len(X)}")

    print(f"\nNumber of features: {len(X.columns)}")

    print("\nFeatures:")

    for column in X.columns:
        print(f"- {column}")

    print("\nTrain/Test Split:")

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    print("\nTarget distribution:")

    print(y.value_counts())

    print("\nTraining incident distribution:")

    print(y_train.value_counts())

    print("\nTesting incident distribution:")

    print(y_test.value_counts())

    print("\nFirst 5 ML features:")

    print(X.head())

    print("\nFirst 5 targets:")

    print(y.head())


if __name__ == "__main__":
    main()