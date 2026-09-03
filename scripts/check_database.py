import sys
from pathlib import Path


# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


from src.database.database import get_connection


def main():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM predictions
        ORDER BY id DESC
        """
    )

    rows = cursor.fetchall()

    print()
    print("=" * 70)
    print("SAVED PREDICTIONS")
    print("=" * 70)

    print(f"\nTotal predictions: {len(rows)}\n")

    for row in rows:

        print(f"ID: {row['id']}")
        print(f"Timestamp: {row['timestamp']}")
        print(f"CPU Usage: {row['cpu_usage']}")
        print(f"Memory Usage: {row['memory_usage']}")
        print(f"Latency: {row['latency_ms']}")
        print(f"Packet Loss: {row['packet_loss']}")
        print(f"Throughput: {row['throughput_mbps']}")
        print(
            f"Probability: "
            f"{row['incident_probability']}"
        )
        print(
            f"Production Threshold: "
            f"{row['production_threshold']}"
        )
        print(f"Prediction: {row['prediction']}")
        print(f"Status: {row['status']}")

        print("-" * 70)

    connection.close()


if __name__ == "__main__":
    main()