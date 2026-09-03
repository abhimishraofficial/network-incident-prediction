from typing import List

from pydantic import BaseModel, Field


class NetworkMetrics(BaseModel):
    """
    A single network metrics reading.
    """

    cpu_usage: float = Field(
        ...,
        ge=0,
        le=100,
        description="CPU usage percentage"
    )

    memory_usage: float = Field(
        ...,
        ge=0,
        le=100,
        description="Memory usage percentage"
    )

    latency_ms: float = Field(
        ...,
        ge=0,
        description="Network latency in milliseconds"
    )

    packet_loss: float = Field(
        ...,
        ge=0,
        le=100,
        description="Packet loss percentage"
    )

    throughput_mbps: float = Field(
        ...,
        ge=0,
        description="Network throughput in Mbps"
    )


class NetworkMetricsRequest(BaseModel):
    """
    Request containing recent network metric history.
    """

    metrics_history: List[NetworkMetrics] = Field(
        ...,
        min_length=3,
        max_length=3,
        description=(
            "Exactly 3 network readings ordered "
            "from oldest to newest"
        )
    )


class PredictionResponse(BaseModel):
    """
    Response returned after incident prediction.
    """

    incident_probability: float
    production_threshold: float
    prediction: int
    status: str


class SavedPrediction(BaseModel):
    """
    A prediction saved in the SQLite database.
    """

    id: int
    timestamp: str
    cpu_usage: float
    memory_usage: float
    latency_ms: float
    packet_loss: float
    throughput_mbps: float
    incident_probability: float
    production_threshold: float
    prediction: int
    status: str


class PredictionsResponse(BaseModel):
    """
    Response containing saved predictions.
    """

    total_predictions: int
    predictions: List[SavedPrediction]


class StatisticsResponse(BaseModel):
    """
    Response containing prediction statistics.
    """

    total_predictions: int
    incident_predictions: int
    normal_predictions: int
    incident_rate: float
    average_incident_probability: float


class HealthResponse(BaseModel):
    """
    Response containing application health status.
    """

    status: str
    model_available: bool
    metadata_available: bool
    database_available: bool