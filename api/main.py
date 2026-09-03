from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes import router
from src.database.database import initialize_database


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize application resources on startup.
    """

    initialize_database()

    yield


app = FastAPI(
    title="Network Incident Prediction API",
    description=(
        "API for predicting potential network "
        "incidents using a Random Forest model."
    ),
    version="1.0.0",
    lifespan=lifespan
)


app.include_router(router)