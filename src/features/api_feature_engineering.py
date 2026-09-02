from datetime import datetime

import pandas as pd


def create_api_features(metrics_history):
    """
    Convert 3 historical network readings into
    the 23 features required by the trained model.

    The readings must be ordered from oldest
    to newest.
    """

    df = pd.DataFrame(metrics_history)

    metric_columns = [
        "cpu_usage",
        "memory_usage",
        "latency_ms",
        "packet_loss",
        "throughput_mbps"
    ]

    # Latest network reading
    current = df.iloc[-1]

    # Previous network reading
    previous = df.iloc[-2]

    now = datetime.now()

    hour = now.hour
    day_of_week = now.weekday()

    is_peak_hour = int(
        9 <= hour <= 21
    )

    features = {
        # Current metrics
        "cpu_usage": current["cpu_usage"],
        "memory_usage": current["memory_usage"],
        "latency_ms": current["latency_ms"],
        "packet_loss": current["packet_loss"],
        "throughput_mbps": current["throughput_mbps"],

        # Time features
        "hour": hour,
        "day_of_week": day_of_week,
        "is_peak_hour": is_peak_hour
    }

    # Historical features
    for column in metric_columns:

        # Previous reading
        features[f"{column}_lag_1"] = previous[column]

        # Change from previous reading
        features[f"{column}_change"] = (
            current[column] - previous[column]
        )

        # Rolling mean of all 3 readings
        features[f"{column}_rolling_mean_3"] = (
            df[column].mean()
        )

    return pd.DataFrame([features])