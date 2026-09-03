import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATABASE_PATH = (
    PROJECT_ROOT
    / "data"
    / "app.db"
)


def get_connection():
    """
    Create and return a SQLite database connection.
    """

    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


def initialize_database():
    """
    Create the predictions table if it does not exist.
    """

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            cpu_usage REAL NOT NULL,
            memory_usage REAL NOT NULL,
            latency_ms REAL NOT NULL,
            packet_loss REAL NOT NULL,
            throughput_mbps REAL NOT NULL,
            incident_probability REAL NOT NULL,
            production_threshold REAL NOT NULL,
            prediction INTEGER NOT NULL,
            status TEXT NOT NULL
        )
        """
    )

    connection.commit()

    connection.close()


def save_prediction(
    timestamp,
    cpu_usage,
    memory_usage,
    latency_ms,
    packet_loss,
    throughput_mbps,
    incident_probability,
    production_threshold,
    prediction,
    status
):
    """
    Save a network prediction to SQLite.
    """

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO predictions (
            timestamp,
            cpu_usage,
            memory_usage,
            latency_ms,
            packet_loss,
            throughput_mbps,
            incident_probability,
            production_threshold,
            prediction,
            status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            timestamp,
            cpu_usage,
            memory_usage,
            latency_ms,
            packet_loss,
            throughput_mbps,
            incident_probability,
            production_threshold,
            prediction,
            status
        )
    )

    connection.commit()

    connection.close()


def get_predictions(
    limit=10,
    incident_only=False
):
    """
    Retrieve saved predictions from SQLite.
    """

    connection = get_connection()

    cursor = connection.cursor()

    query = """
        SELECT *
        FROM predictions
    """

    parameters = []

    if incident_only:

        query += """
            WHERE prediction = 1
        """

    query += """
        ORDER BY id DESC
        LIMIT ?
    """

    parameters.append(limit)

    cursor.execute(
        query,
        parameters
    )

    rows = cursor.fetchall()

    connection.close()

    return [
        dict(row)
        for row in rows
    ]


def get_prediction_statistics():
    """
    Calculate prediction statistics from SQLite.
    """

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            COUNT(*) AS total_predictions,

            SUM(
                CASE
                    WHEN prediction = 1
                    THEN 1
                    ELSE 0
                END
            ) AS incident_predictions,

            SUM(
                CASE
                    WHEN prediction = 0
                    THEN 1
                    ELSE 0
                END
            ) AS normal_predictions,

            AVG(
                incident_probability
            ) AS average_incident_probability

        FROM predictions
        """
    )

    row = cursor.fetchone()

    connection.close()

    total_predictions = row["total_predictions"]

    incident_predictions = (
        row["incident_predictions"] or 0
    )

    normal_predictions = (
        row["normal_predictions"] or 0
    )

    average_probability = (
        row["average_incident_probability"] or 0
    )

    incident_rate = 0

    if total_predictions > 0:

        incident_rate = round(
            (
                incident_predictions
                / total_predictions
            )
            * 100,
            2
        )

    return {
        "total_predictions": total_predictions,
        "incident_predictions": incident_predictions,
        "normal_predictions": normal_predictions,
        "incident_rate": incident_rate,
        "average_incident_probability": round(
            average_probability,
            4
        )
    }