import os

import joblib


def save_model(
    model,
    model_path: str
) -> None:
    """
    Save a trained machine learning model.
    """

    directory = os.path.dirname(
        model_path
    )

    if directory:
        os.makedirs(
            directory,
            exist_ok=True
        )

    joblib.dump(
        model,
        model_path
    )

    print(
        f"Model saved successfully: {model_path}"
    )


def load_model(
    model_path: str
):
    """
    Load a previously trained machine learning model.
    """

    model = joblib.load(
        model_path
    )

    return model