import pandas as pd

from src.features.temporal_features import (
    create_temporal_features
)


FEATURE_COLUMNS = [
    # Current metrics
    "cpu_usage",
    "memory_usage",
    "latency_ms",
    "packet_loss",
    "throughput_mbps",

    # Time features
    "hour",
    "day_of_week",
    "is_peak_hour",

    # CPU historical features
    "cpu_usage_lag_1",
    "cpu_usage_change",
    "cpu_usage_rolling_mean_3",

    # Memory historical features
    "memory_usage_lag_1",
    "memory_usage_change",
    "memory_usage_rolling_mean_3",

    # Latency historical features
    "latency_ms_lag_1",
    "latency_ms_change",
    "latency_ms_rolling_mean_3",

    # Packet loss historical features
    "packet_loss_lag_1",
    "packet_loss_change",
    "packet_loss_rolling_mean_3",

    # Throughput historical features
    "throughput_mbps_lag_1",
    "throughput_mbps_change",
    "throughput_mbps_rolling_mean_3"
]


def prepare_ml_data(
    df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare network metrics for incident prediction.
    """

    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df = df.sort_values(
        by=["site_id", "timestamp"]
    )

    # ------------------------------------------
    # CREATE TEMPORAL FEATURES
    # ------------------------------------------

    df = create_temporal_features(
        df
    )

    # ------------------------------------------
    # CREATE FUTURE INCIDENT TARGET
    # ------------------------------------------

    df["future_incident"] = (
        df.groupby("site_id")["incident"]
        .shift(-1)
    )

    # Remove rows without a future target
    df = df.dropna(
        subset=["future_incident"]
    )

    df["future_incident"] = (
        df["future_incident"]
        .astype(int)
    )

    # Input features
    X = df[FEATURE_COLUMNS]

    # Target
    y = df["future_incident"]

    return X, y