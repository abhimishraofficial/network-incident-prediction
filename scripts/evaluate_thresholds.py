import pandas as pd
import requests

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


DATA_PATH = "data/processed/network_incident_dataset.csv"
API_URL = "http://127.0.0.1:8000/predict"

RANDOM_STATE = 42
SAMPLES_PER_CLASS = 50

THRESHOLDS = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50
]


def create_payload(history):

    metrics_history = []

    for _, row in history.iterrows():

        metrics_history.append(
            {
                "cpu_usage": float(row["cpu_usage"]),
                "memory_usage": float(row["memory_usage"]),
                "latency_ms": float(row["latency_ms"]),
                "packet_loss": float(row["packet_loss"]),
                "throughput_mbps": float(
                    row["throughput_mbps"]
                )
            }
        )

    return {
        "metrics_history": metrics_history
    }


def main():

    print("=" * 70)
    print("RANDOM THRESHOLD EVALUATION STARTED")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)

    print()
    print(f"Total dataset rows: {len(df)}")

    incident_data = df[
        df["incident"] == 1
    ]

    normal_data = df[
        df["incident"] == 0
    ]

    print(f"Total incident rows: {len(incident_data)}")
    print(f"Total normal rows: {len(normal_data)}")

    # Randomly select incident samples
    incident_rows = incident_data.sample(
        n=min(SAMPLES_PER_CLASS, len(incident_data)),
        random_state=RANDOM_STATE
    )

    # Randomly select normal samples
    normal_rows = normal_data.sample(
        n=min(SAMPLES_PER_CLASS, len(normal_data)),
        random_state=RANDOM_STATE
    )

    # Combine and shuffle samples
    test_rows = pd.concat(
        [
            incident_rows,
            normal_rows
        ]
    ).sample(
        frac=1,
        random_state=RANDOM_STATE
    )

    actual_values = []
    probabilities = []

    print()
    print(
        f"Randomly selected {len(test_rows)} "
        f"samples for evaluation..."
    )

    for index, row in test_rows.iterrows():

        # Need current row + previous 2 rows
        if index < 2:
            continue

        history = df.iloc[
            index - 2:index + 1
        ]

        # Ensure exactly 3 readings
        if len(history) != 3:
            continue

        # Very important:
        # All three readings must belong to the same site
        if history["site_id"].nunique() != 1:
            continue

        payload = create_payload(history)

        try:

            response = requests.post(
                API_URL,
                json=payload,
                timeout=10
            )

            response.raise_for_status()

            result = response.json()

            probability = float(
                result["incident_probability"]
            )

            actual_values.append(
                int(row["incident"])
            )

            probabilities.append(
                probability
            )

        except Exception as error:

            print(
                f"Error testing row {index}: {error}"
            )

    print()
    print(
        f"Successfully evaluated: "
        f"{len(actual_values)} samples"
    )

    if len(actual_values) == 0:

        print()
        print("No valid samples were evaluated.")
        print(
            "Check whether the API is running "
            "and whether valid 3-reading histories exist."
        )

        return

    print()
    print("=" * 70)
    print("THRESHOLD COMPARISON")
    print("=" * 70)

    results = []

    for threshold in THRESHOLDS:

        predictions = [
            1 if probability >= threshold else 0
            for probability in probabilities
        ]

        accuracy = accuracy_score(
            actual_values,
            predictions
        )

        precision = precision_score(
            actual_values,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            actual_values,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            actual_values,
            predictions,
            zero_division=0
        )

        cm = confusion_matrix(
            actual_values,
            predictions,
            labels=[0, 1]
        )

        results.append(
            {
                "threshold": threshold,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "confusion_matrix": cm
            }
        )

        print()
        print("-" * 70)
        print(f"THRESHOLD: {threshold}")
        print("-" * 70)

        print(f"Accuracy : {accuracy:.2%}")
        print(f"Precision: {precision:.2%}")
        print(f"Recall   : {recall:.2%}")
        print(f"F1 Score : {f1:.4f}")

        print("Confusion Matrix:")
        print(cm)

    best_result = max(
        results,
        key=lambda x: x["f1_score"]
    )

    best_threshold = best_result["threshold"]

    best_predictions = [
        1 if probability >= best_threshold else 0
        for probability in probabilities
    ]

    print()
    print("=" * 70)
    print("BEST THRESHOLD BY F1 SCORE")
    print("=" * 70)

    print(
        f"Best Threshold: {best_threshold}"
    )

    print(
        f"Accuracy: "
        f"{best_result['accuracy']:.2%}"
    )

    print(
        f"Precision: "
        f"{best_result['precision']:.2%}"
    )

    print(
        f"Recall: "
        f"{best_result['recall']:.2%}"
    )

    print(
        f"F1 Score: "
        f"{best_result['f1_score']:.4f}"
    )

    print()
    print("Best Threshold Confusion Matrix:")
    print(best_result["confusion_matrix"])

    print()
    print("Classification Report:")

    print(
        classification_report(
            actual_values,
            best_predictions,
            target_names=["NORMAL", "INCIDENT"],
            zero_division=0
        )
    )

    print("=" * 70)
    print("RANDOM THRESHOLD EVALUATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()