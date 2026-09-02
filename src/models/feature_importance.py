import pandas as pd


def get_feature_importance(
    model,
    feature_names: list
) -> pd.DataFrame:
    """
    Extract and sort feature importance values
    from a trained tree-based model.
    """

    if not hasattr(
        model,
        "feature_importances_"
    ):
        raise ValueError(
            "This model does not support feature importance."
        )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_
        }
    )

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    )

    importance_df = importance_df.reset_index(
        drop=True
    )

    return importance_df