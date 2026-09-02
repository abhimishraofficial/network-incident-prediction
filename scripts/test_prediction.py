import pandas as pd

from src.models.data_preparation import prepare_ml_data
from src.models.prediction import predict_incident


DATA_PATH = "data/processed/network_incident_dataset.csv"


def main():

    print("=" * 60)
    print("NETWORK INCIDENT PREDICTION TEST")
    print("=" * 60)

    # Load processed dataset
    print("\n[1/4] Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    print(
        f"Records loaded: {len(df)}"
    )

    # Prepare ML features
    print("\n[2/4] Preparing ML features...")

    X, y = prepare_ml_data(df)

    print(
        f"Features prepared: {X.shape[1]}"
    )

    # Find normal and incident samples
    print("\n[3/4] Selecting test samples...")

    normal_index = y[y == 0].index[0]
    incident_index = y[y == 1].index[0]

    normal_sample = X.loc[
        [normal_index]
    ]

    incident_sample = X.loc[
        [incident_index]
    ]

    # Predict normal sample
    print("\n[4/4] Running predictions...")

    print("\n" + "-" * 60)
    print("NORMAL NETWORK SAMPLE")
    print("-" * 60)

    normal_result = predict_incident(
        normal_sample
    )

    for key, value in normal_result.items():
        print(f"{key}: {value}")

    # Predict incident sample
    print("\n" + "-" * 60)
    print("INCIDENT NETWORK SAMPLE")
    print("-" * 60)

    incident_result = predict_incident(
        incident_sample
    )

    for key, value in incident_result.items():
        print(f"{key}: {value}")

    # Actual values
    print("\n" + "=" * 60)
    print("ACTUAL RESULTS")
    print("=" * 60)

    print(
        f"Normal sample actual incident: "
        f"{y.loc[normal_index]}"
    )

    print(
        f"Incident sample actual incident: "
        f"{y.loc[incident_index]}"
    )

    print("\nPrediction test completed successfully!")


if __name__ == "__main__":
    main()