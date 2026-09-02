from fastapi import FastAPI

from api.routes import router


app = FastAPI(
    title="Network Incident Prediction API",
    description=(
        "API for predicting potential network "
        "incidents using a Random Forest model."
    ),
    version="1.0.0"
)


app.include_router(router)