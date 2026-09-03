from datetime import datetime

from fastapi import APIRouter, HTTPException

from api.schemas import (
    NetworkMetricsRequest,
    PredictionResponse,
    PredictionsResponse,
    StatisticsResponse,
    HealthResponse
)

from src.features.api_feature_engineering import (
    create_api_features
)

from src.models.prediction import (
    predict_incident,
    MODEL_PATH,
    METADATA_PATH
)

from src.database.database import (
    save_prediction,
    get_predictions,
    get_prediction_statistics,
    get_connection
)


router = APIRouter()


@router.get("/")
def home():
    """
    Basic API endpoint.
    """

    return {
        "message": "Network Incident Prediction API is running",
        "status": "healthy"
    }


@router.get(
    "/health",
    response_model=HealthResponse
)
def health_check():
    """
    Check application dependencies.
    """

    model_available = MODEL_PATH.exists()

    metadata_available = METADATA_PATH.exists()

    database_available = False

    try:

        connection = get_connection()

        connection.execute(
            "SELECT 1"
        )

        connection.close()

        database_available = True

    except Exception:

        database_available = False

    overall_status = (
        "healthy"
        if (
            model_available
            and metadata_available
            and database_available
        )
        else "unhealthy"
    )

    return {
        "status": overall_status,
        "model_available": model_available,
        "metadata_available": metadata_available,
        "database_available": database_available
    }


@router.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(
    request: NetworkMetricsRequest
):
    """
    Predict network incident risk using
    recent network metrics history and
    save the prediction in SQLite.
    """

    try:

        metrics_history = [
            metric.model_dump()
            for metric in request.metrics_history
        ]

        features = create_api_features(
            metrics_history
        )

        result = predict_incident(features)

        latest_metrics = metrics_history[-1]

        save_prediction(
            timestamp=datetime.now().isoformat(),
            cpu_usage=latest_metrics["cpu_usage"],
            memory_usage=latest_metrics["memory_usage"],
            latency_ms=latest_metrics["latency_ms"],
            packet_loss=latest_metrics["packet_loss"],
            throughput_mbps=latest_metrics["throughput_mbps"],
            incident_probability=result["incident_probability"],
            production_threshold=result["production_threshold"],
            prediction=result["prediction"],
            status=result["status"]
        )

        return result

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


@router.get(
    "/predictions",
    response_model=PredictionsResponse
)
def get_saved_predictions(
    limit: int = 10,
    incident_only: bool = False
):
    """
    Retrieve recent saved predictions.

    Set incident_only=true to retrieve only
    incident risk predictions.
    """

    try:

        predictions = get_predictions(
            limit=limit,
            incident_only=incident_only
        )

        return {
            "total_predictions": len(predictions),
            "predictions": predictions
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


@router.get(
    "/statistics",
    response_model=StatisticsResponse
)
def get_statistics():
    """
    Retrieve prediction statistics.
    """

    try:

        statistics = get_prediction_statistics()

        return statistics

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )