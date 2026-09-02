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