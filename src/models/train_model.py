import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_models() -> dict:
    """
    Create and return the machine learning models
    used for network incident prediction.
    """

    logistic_regression = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42
                )
            )
        ]
    )

    random_forest = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    models = {
        "Logistic Regression": logistic_regression,
        "Random Forest": random_forest
    }

    return models


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> dict:
    """
    Train all machine learning models.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.

    y_train : pd.Series
        Training target.

    Returns
    -------
    dict
        Dictionary containing trained models.
    """

    models = get_models()

    trained_models = {}

    for model_name, model in models.items():

        print(
            f"\nTraining {model_name}..."
        )

        model.fit(
            X_train,
            y_train
        )

        trained_models[model_name] = model

        print(
            f"{model_name} trained successfully!"
        )

    return trained_models