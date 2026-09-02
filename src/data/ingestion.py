from pathlib import Path

import pandas as pd


def load_network_data(
    file_path: str
) -> pd.DataFrame:
    """
    Load network metrics data from a CSV file.

    Parameters
    ----------
    file_path : str
        Path of the network metrics CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded network metrics data.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Network data file not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError(
            "Network data file is empty."
        )

    return df