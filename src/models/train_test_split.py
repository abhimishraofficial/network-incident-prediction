import pandas as pd


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series
]:
    """
    Split ML data into training and testing sets.

    Uses chronological ordering instead of a
    random split.
    """

    split_index = int(
        len(X) * (1 - test_size)
    )

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )