import pandas as pd
import requests

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


DATA_PATH = "data/processed/network_incident_dataset.csv"
API_URL = "http://127.0.0.1:8000/predict"

# Number of samples from each class
SAMPLES_PER_CLASS = 50


# Load dataset
df = pd.read_csv(DATA_PATH)

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])


def get_history_for_row(row):
    """
    Get exactly 3 readings for the same site,
    ordered from oldest to newest.
    """

    site_id = row["site_id"]
    timestamp = row["timestamp"]

    history = df[
        (df["site_id"] == site_id)
        & (df["timestamp"] <= timestamp)
    ].sort_values("timestamp").tail(3)

    if len(history) != 3:
        return None

    metric_columns = [
        "cpu_usage",
        "memory_usage",
        "latency_ms",
        "packet_loss",
        "throughput_mbps"
    ]

    return history[
        metric_columns
    ].to_dict(orient="records")


# Get real incident and normal samples
incident_samples = df[
    df["incident"] == 1
].head(SAMPLES_PER_CLASS)

normal_samples = df[
    df["incident"] == 0
].head(SAMPLES_PER_CLASS)


# Combine samples
test_samples = pd.concat(
    [incident_samples, normal_samples]
)

# Store actual and predicted values
y_true = []
y_pred = []

print("\n" + "=" * 60)
print("API EVALUATION STARTED")
print("=" * 60)

print(
    f"Testing {len(incident_samples)} incident samples "
    f"and {len(normal_samples)} normal samples"
)


for index, row in test_samples.iterrows():

    history = get_history_for_row(row)

    if history is None:
        print(
            f"Skipping row {index}: "
            "not enough history"
        )
        continue

    payload = {
        "metrics_history": history
    }

    try:

        response = requests.post(
            API_URL,
            json=payload,
            timeout=10
        )

        response.raise_for_status()

        result = response.json()

        actual = int(row["incident"])
        predicted = int(result["prediction"])

        y_true.append(actual)
        y_pred.append(predicted)

        print(
            f"Row {index} | "
            f"Actual: {actual} | "
            f"Predicted: {predicted} | "
            f"Probability: "
            f"{result['incident_probability']:.4f}"
        )

    except requests.exceptions.RequestException as error:

        print(
            f"API error for row {index}: {error}"
        )


print("\n" + "=" * 60)
print("API EVALUATION RESULTS")
print("=" * 60)

if len(y_true) == 0:

    print("No samples were successfully evaluated.")

else:

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    matrix = confusion_matrix(
        y_true,
        y_pred
    )

    correct_predictions = sum(
        actual == predicted
        for actual, predicted
        in zip(y_true, y_pred)
    )

    print(
        f"\nTotal evaluated samples: "
        f"{len(y_true)}"
    )

    print(
        f"Correct predictions: "
        f"{correct_predictions}"
    )

    print(
        f"Incorrect predictions: "
        f"{len(y_true) - correct_predictions}"
    )

    print(
        f"\nAccuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "NORMAL",
                "INCIDENT"
            ],
            zero_division=0
        )
    )

print("\n" + "=" * 60)
print("API EVALUATION COMPLETED")
print("=" * 60)