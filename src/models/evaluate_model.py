import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> dict:
    """
    Evaluate a trained machine learning model.

    Returns important metrics for an imbalanced
    network incident prediction problem.
    """

    # Predict classes
    y_pred = model.predict(
        X_test
    )

    # Predict probabilities
    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(
            y_test,
            y_pred
        ),
        "precision": precision_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "recall": recall_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "f1_score": f1_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_test,
            y_prob
        ),
        "pr_auc": average_precision_score(
            y_test,
            y_prob
        )
    }

    # Create confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred
    )

    metrics["true_negative"] = int(
        cm[0][0]
    )

    metrics["false_positive"] = int(
        cm[0][1]
    )

    metrics["false_negative"] = int(
        cm[1][0]
    )

    metrics["true_positive"] = int(
        cm[1][1]
    )

    return metrics