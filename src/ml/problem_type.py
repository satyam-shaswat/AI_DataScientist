import pandas as pd
from pandas.api import types as ptypes

from src.analysis.profiler import classify_column

# A numerical target with at most this many distinct values is treated as
# classification (e.g. a 0/1 flag or a 1-5 rating).
MAX_CLASSES_FOR_NUMERIC = 10


def detect_problem_type(series: pd.Series) -> str:
    """Return 'classification' or 'regression' for a target column."""
    if ptypes.is_bool_dtype(series) or not ptypes.is_numeric_dtype(series):
        return "classification"

    if series.nunique() <= MAX_CLASSES_FOR_NUMERIC:
        return "classification"

    return "regression"


def check_target(series: pd.Series):
    """Return an error message if the column cannot be a target, else None."""
    if series.notna().sum() == 0:
        return "This column has no values."

    if series.nunique() < 2:
        return "This column has only one distinct value, so there is nothing to predict."

    if classify_column(series) == "identifier":
        return "This column looks like an identifier, so it cannot be predicted meaningfully."

    return None
# Classification with more classes than this is almost certainly a mistake.
MAX_CLASSIFICATION_CLASSES = 20


def check_problem_choice(series: pd.Series, problem_type: str):
    """Return an error message if the chosen problem type does not fit."""
    if problem_type == "classification" and series.nunique() > MAX_CLASSIFICATION_CLASSES:
        return (
            f"This column has {series.nunique()} distinct values, which is too many "
            f"for classification (limit {MAX_CLASSIFICATION_CLASSES}). "
            "Choose regression instead."
        )

    if problem_type == "regression" and not pd.api.types.is_numeric_dtype(series):
        return "Regression needs a numerical target. Choose classification instead."

    return None