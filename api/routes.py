from fastapi import APIRouter, HTTPException

from api.schemas import (
    NetworkMetricsRequest,
    PredictionResponse
)

from src.features.api_feature_engineering import (
    create_api_features
)

from src.models.prediction import predict_incident


router = APIRouter()


@router.get("/")
def home():
    """
    Basic API health endpoint.
    """

    return {
        "message": "Network Incident Prediction API is running",
        "status": "healthy"
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
    recent network metrics history.
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

        return result

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )