import pandas as pd

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)


def evaluate_threshold(
    y_true,
    probabilities,
    threshold: float
) -> dict:
    """
    Evaluate model performance at one
    probability threshold.
    """

    predictions = (
        probabilities >= threshold
    ).astype(int)

    cm = confusion_matrix(
        y_true,
        predictions
    )

    return {
        "threshold": threshold,
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "f1_score": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "true_negative": int(cm[0][0]),
        "false_positive": int(cm[0][1]),
        "false_negative": int(cm[1][0]),
        "true_positive": int(cm[1][1])
    }


def tune_threshold(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> pd.DataFrame:
    """
    Test multiple probability thresholds
    and return the evaluation results.
    """

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    thresholds = [
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90
    ]

    results = []

    for threshold in thresholds:

        metrics = evaluate_threshold(
            y_test,
            probabilities,
            threshold
        )

        results.append(
            metrics
        )

    results_df = pd.DataFrame(
        results
    )

    return results_df


def get_best_threshold(
    results_df: pd.DataFrame
) -> dict:
    """
    Select the threshold with the highest
    F1 score.
    """

    best_row = results_df.loc[
        results_df["f1_score"].idxmax()
    ]

    return best_row.to_dict()