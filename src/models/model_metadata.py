import json
import os
from datetime import datetime


def save_model_metadata(
    model_name: str,
    model_version: str,
    threshold: float,
    metrics: dict,
    feature_names: list,
    output_path: str
) -> None:
    """
    Save model information and configuration
    as a JSON file.
    """

    metadata = {
        "model_name": model_name,
        "model_version": model_version,
        "created_at": datetime.now().isoformat(),
        "production_threshold": threshold,
        "metrics": metrics,
        "feature_names": feature_names
    }

    directory = os.path.dirname(
        output_path
    )

    if directory:
        os.makedirs(
            directory,
            exist_ok=True
        )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )


def load_model_metadata(
    metadata_path: str
) -> dict:
    """
    Load model metadata from a JSON file.
    """

    if not os.path.exists(
        metadata_path
    ):
        raise FileNotFoundError(
            f"Metadata file not found: "
            f"{metadata_path}"
        )

    with open(
        metadata_path,
        "r",
        encoding="utf-8"
    ) as file:

        metadata = json.load(
            file
        )

    return metadata