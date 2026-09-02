import pandas as pd


def create_temporal_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create historical and trend-based features
    for each network site.
    """

    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df = df.sort_values(
        by=["site_id", "timestamp"]
    )

    metric_columns = [
        "cpu_usage",
        "memory_usage",
        "latency_ms",
        "packet_loss",
        "throughput_mbps"
    ]

    for column in metric_columns:

        # Previous monitoring value
        df[f"{column}_lag_1"] = (
            df.groupby("site_id")[column]
            .shift(1)
        )

        # Change from previous value
        df[f"{column}_change"] = (
            df[column]
            - df[f"{column}_lag_1"]
        )

        # Rolling average of previous 3 intervals
        df[f"{column}_rolling_mean_3"] = (
            df.groupby("site_id")[column]
            .transform(
                lambda x: x.rolling(
                    window=3,
                    min_periods=1
                ).mean()
            )
        )

    # Drop first row of every site because lag features
    # are unavailable there
    df = df.dropna()

    return df