import pandas as pd


def create_network_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional features from network metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Validated network metrics data.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional engineered features.
    """

    # Create a copy to avoid changing the original DataFrame
    df = df.copy()

    # Convert timestamp from string to datetime
    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    # ------------------------------------------
    # TIME-BASED FEATURES
    # ------------------------------------------

    df["hour"] = df["timestamp"].dt.hour

    df["day_of_week"] = (
        df["timestamp"].dt.dayofweek
    )

    # Peak hours: 9 AM to 9 PM
    df["is_peak_hour"] = (
        (
            df["hour"] >= 9
        )
        &
        (
            df["hour"] <= 21
        )
    ).astype(int)

    # ------------------------------------------
    # NETWORK RISK FEATURES
    # ------------------------------------------

    # High CPU usage
    df["high_cpu"] = (
        df["cpu_usage"] >= 80
    ).astype(int)

    # High memory usage
    df["high_memory"] = (
        df["memory_usage"] >= 85
    ).astype(int)

    # High latency
    df["high_latency"] = (
        df["latency_ms"] >= 100
    ).astype(int)

    # High packet loss
    df["high_packet_loss"] = (
        df["packet_loss"] >= 2
    ).astype(int)

    # Low throughput
    df["low_throughput"] = (
        df["throughput_mbps"] <= 80
    ).astype(int)

    return df