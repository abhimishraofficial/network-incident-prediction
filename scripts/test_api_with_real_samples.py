import pandas as pd
import requests


DATA_PATH = "data/processed/network_incident_dataset.csv"
API_URL = "http://127.0.0.1:8000/predict"


# Load dataset
df = pd.read_csv(DATA_PATH)

# Convert timestamp to datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])


def get_history_for_row(row):
    """
    Get the current row and previous 2 readings
    for the same site.

    Returns readings ordered from oldest to newest.
    """

    site_id = row["site_id"]
    timestamp = row["timestamp"]

    history = df[
        (df["site_id"] == site_id) &
        (df["timestamp"] <= timestamp)
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

    return history[metric_columns].to_dict(
        orient="records"
    )


# Select one real incident sample
incident_samples = df[df["incident"] == 1]

if incident_samples.empty:
    print("No incident samples found.")
else:

    row = incident_samples.iloc[0]

    history = get_history_for_row(row)

    if history is None:
        print(
            "Not enough historical readings "
            "for incident sample."
        )

    else:

        payload = {
            "metrics_history": history
        }

        print("\n" + "=" * 60)
        print("TEST 1: REAL INCIDENT SAMPLE")
        print("=" * 60)

        print("\nActual incident:", row["incident"])
        print("Site ID:", row["site_id"])
        print("Timestamp:", row["timestamp"])

        print("\nRequest payload:")
        print(payload)

        try:

            response = requests.post(
                API_URL,
                json=payload,
                timeout=10
            )

            print("\nHTTP Status Code:")
            print(response.status_code)

            print("\nAPI Response:")
            print(response.json())

        except requests.exceptions.ConnectionError:

            print(
                "\nERROR: Could not connect to API."
            )
            print(
                "Make sure Uvicorn is running:"
            )
            print(
                "uvicorn api.main:app --reload"
            )


# Select one real normal sample
normal_samples = df[df["incident"] == 0]

if normal_samples.empty:
    print("No normal samples found.")
else:

    row = normal_samples.iloc[2]

    history = get_history_for_row(row)

    if history is None:
        print(
            "Not enough historical readings "
            "for normal sample."
        )

    else:

        payload = {
            "metrics_history": history
        }

        print("\n" + "=" * 60)
        print("TEST 2: REAL NORMAL SAMPLE")
        print("=" * 60)

        print("\nActual incident:", row["incident"])
        print("Site ID:", row["site_id"])
        print("Timestamp:", row["timestamp"])

        print("\nRequest payload:")
        print(payload)

        try:

            response = requests.post(
                API_URL,
                json=payload,
                timeout=10
            )

            print("\nHTTP Status Code:")
            print(response.status_code)

            print("\nAPI Response:")
            print(response.json())

        except requests.exceptions.ConnectionError:

            print(
                "\nERROR: Could not connect to API."
            )
            print(
                "Make sure Uvicorn is running:"
            )
            print(
                "uvicorn api.main:app --reload"
            )


print("\n" + "=" * 60)
print("REAL SAMPLE API TEST COMPLETED")
print("=" * 60)