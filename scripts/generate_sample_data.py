from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42

NUM_SITES = 20

DAYS = 30

FREQUENCY = "5min"

OUTPUT_FILE = (
    "data/raw/network_metrics.csv"
)


def generate_site_data(
    site_number: int,
    timestamps: pd.DatetimeIndex,
    rng: np.random.Generator
) -> pd.DataFrame:
    """
    Generate realistic network metrics for one site.

    The site normally operates within a healthy range.
    Before selected incidents, network degradation begins
    gradually so ML models can learn early warning patterns.
    """

    num_records = len(timestamps)

    # ------------------------------------------
    # BASELINE NETWORK METRICS
    # ------------------------------------------

    cpu_usage = rng.normal(
        loc=55,
        scale=8,
        size=num_records
    )

    memory_usage = rng.normal(
        loc=60,
        scale=10,
        size=num_records
    )

    latency_ms = rng.normal(
        loc=35,
        scale=12,
        size=num_records
    )

    packet_loss = rng.normal(
        loc=0.4,
        scale=0.25,
        size=num_records
    )

    throughput_mbps = rng.normal(
        loc=150,
        scale=25,
        size=num_records
    )

    # Keep values realistic
    cpu_usage = np.clip(cpu_usage, 5, 100)
    memory_usage = np.clip(memory_usage, 5, 100)
    latency_ms = np.clip(latency_ms, 1, 500)
    packet_loss = np.clip(packet_loss, 0, 20)
    throughput_mbps = np.clip(
        throughput_mbps,
        1,
        500
    )

    # ------------------------------------------
    # CREATE INCIDENT PERIODS
    # ------------------------------------------

    incident = np.zeros(
        num_records,
        dtype=int
    )

    # Around 8-12 incidents per site
    num_incidents = int(
        rng.integers(8, 13)
    )

    possible_positions = list(
        range(
            30,
            num_records - 10
        )
    )

    rng.shuffle(
        possible_positions
    )

    incident_positions = []

    for position in possible_positions:

        # Prevent incident periods from overlapping
        if all(
            abs(position - previous) > 20
            for previous in incident_positions
        ):
            incident_positions.append(
                position
            )

        if (
            len(incident_positions)
            >= num_incidents
        ):
            break

    # ------------------------------------------
    # SIMULATE GRADUAL DEGRADATION
    # ------------------------------------------

    for incident_index in incident_positions:

        # Warning period:
        # 6 intervals = 30 minutes before incident
        warning_start = max(
            0,
            incident_index - 6
        )

        warning_indices = np.arange(
            warning_start,
            incident_index + 1
        )

        warning_length = len(
            warning_indices
        )

        # Gradual increase from 0 to 1
        severity = np.linspace(
            0,
            1,
            warning_length
        )

        # CPU gradually increases
        cpu_usage[
            warning_indices
        ] += severity * rng.uniform(
            25,
            40
        )

        # Memory gradually increases
        memory_usage[
            warning_indices
        ] += severity * rng.uniform(
            15,
            30
        )

        # Latency increases strongly
        latency_ms[
            warning_indices
        ] += severity * rng.uniform(
            70,
            150
        )

        # Packet loss increases
        packet_loss[
            warning_indices
        ] += severity * rng.uniform(
            2,
            5
        )

        # Throughput decreases
        throughput_mbps[
            warning_indices
        ] -= severity * rng.uniform(
            50,
            100
        )

        # Mark the final point as incident
        incident[
            incident_index
        ] = 1

    # Final clipping after degradation
    cpu_usage = np.clip(
        cpu_usage,
        0,
        100
    )

    memory_usage = np.clip(
        memory_usage,
        0,
        100
    )

    latency_ms = np.clip(
        latency_ms,
        1,
        500
    )

    packet_loss = np.clip(
        packet_loss,
        0,
        20
    )

    throughput_mbps = np.clip(
        throughput_mbps,
        1,
        500
    )

    # ------------------------------------------
    # CREATE DATAFRAME
    # ------------------------------------------

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "site_id": (
                f"SITE_{site_number:03d}"
            ),
            "cpu_usage": np.round(
                cpu_usage,
                2
            ),
            "memory_usage": np.round(
                memory_usage,
                2
            ),
            "latency_ms": np.round(
                latency_ms,
                2
            ),
            "packet_loss": np.round(
                packet_loss,
                3
            ),
            "throughput_mbps": np.round(
                throughput_mbps,
                2
            ),
            "incident": incident
        }
    )

    return df


def main():

    print("=" * 60)
    print("REALISTIC NETWORK DATA GENERATION")
    print("=" * 60)

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    timestamps = pd.date_range(
        start="2026-01-01",
        periods=(
            DAYS * 24 * 12
        ),
        freq=FREQUENCY
    )

    all_sites = []

    print(
        f"\nGenerating data for {NUM_SITES} sites..."
    )

    for site_number in range(
        1,
        NUM_SITES + 1
    ):

        site_data = generate_site_data(
            site_number,
            timestamps,
            rng
        )

        all_sites.append(
            site_data
        )

    df = pd.concat(
        all_sites,
        ignore_index=True
    )

    # Ensure output directory exists
    Path(
        OUTPUT_FILE
    ).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nNetwork data generated successfully!")

    print(
        f"Total records: {len(df)}"
    )

    print(
        f"Total sites: {NUM_SITES}"
    )

    print(
        f"Records per site: "
        f"{len(timestamps)}"
    )

    print(
        f"Total incidents: "
        f"{df['incident'].sum()}"
    )

    print(
        "\nIncident distribution:"
    )

    print(
        df["incident"]
        .value_counts()
        .sort_index()
    )

    print(
        f"\nSaved file: {OUTPUT_FILE}"
    )

    print(
        "\nFirst 5 records:"
    )

    print(
        df.head()
    )


if __name__ == "__main__":
    main()