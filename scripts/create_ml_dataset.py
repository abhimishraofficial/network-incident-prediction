import os

from src.data.ingestion import load_network_data
from src.data.validation import validate_network_data
from src.features.feature_engineering import create_network_features


INPUT_FILE = "data/raw/network_metrics.csv"

OUTPUT_FILE = (
    "data/processed/network_incident_dataset.csv"
)


def main():

    print("=" * 60)
    print("NETWORK INCIDENT ML DATASET PIPELINE")
    print("=" * 60)

    # ------------------------------------------
    # STEP 1: LOAD DATA
    # ------------------------------------------

    print("\n[1/3] Loading network data...")

    df = load_network_data(
        INPUT_FILE
    )

    print(
        f"Loaded {len(df)} records."
    )

    # ------------------------------------------
    # STEP 2: VALIDATE DATA
    # ------------------------------------------

    print("\n[2/3] Validating data...")

    df = validate_network_data(
        df
    )

    print("Validation successful.")

    # ------------------------------------------
    # STEP 3: CREATE FEATURES
    # ------------------------------------------

    print(
        "\n[3/3] Creating network features..."
    )

    df = create_network_features(
        df
    )

    print(
        "Feature engineering successful."
    )

    # ------------------------------------------
    # SAVE DATASET
    # ------------------------------------------

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(
        f"\nTotal records: {len(df)}"
    )

    print(
        f"Total columns: {len(df.columns)}"
    )

    print(
        f"\nDataset saved to: {OUTPUT_FILE}"
    )

    print("\nIncident distribution:")

    print(
        df["incident"]
        .value_counts()
        .sort_index()
    )

    print("\nIncident percentage:")

    print(
        (
            df["incident"]
            .value_counts(normalize=True)
            * 100
        )
        .sort_index()
        .round(3)
    )

    print("\nFirst 5 records:")

    print(
        df.head()
    )


if __name__ == "__main__":
    main()