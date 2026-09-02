import pandas as pd


def generate_incident_labels(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate network incident labels based on
    network risk conditions.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing network features.

    Returns
    -------
    pd.DataFrame
        Data with risk_score and incident columns.
    """

    df = df.copy()

    # ------------------------------------------
    # CALCULATE NETWORK RISK SCORE
    # ------------------------------------------

    df["risk_score"] = (
        df["high_cpu"] * 2
        + df["high_memory"] * 2
        + df["high_latency"] * 3
        + df["high_packet_loss"] * 4
        + df["low_throughput"] * 2
    )

    # ------------------------------------------
    # GENERATE INCIDENT LABEL
    # ------------------------------------------

    df["incident"] = (
        df["risk_score"] >= 5
    ).astype(int)

    return df