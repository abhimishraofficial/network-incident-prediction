import pandas as pd


REQUIRED_COLUMNS = [
    "timestamp",
    "site_id",
    "cpu_usage",
    "memory_usage",
    "latency_ms",
    "packet_loss",
    "throughput_mbps",
    "incident"
]


def validate_network_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate network metrics data before processing.
    """

    # Check whether the DataFrame is empty
    if df.empty:
        raise ValueError(
            "Network DataFrame is empty."
        )

    # Check required columns
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Check missing values
    missing_values = df.isnull().sum()

    if missing_values.any():
        print("\nMissing values found:")
        print(
            missing_values[
                missing_values > 0
            ]
        )

        raise ValueError(
            "Data contains missing values."
        )

    # Check duplicate records
    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        print(
            f"\nWarning: Found {duplicate_count} duplicate records."
        )

        df = df.drop_duplicates()

        print(
            f"Duplicates removed. Remaining records: {len(df)}"
        )

    return df