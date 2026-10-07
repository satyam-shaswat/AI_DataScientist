import pandas as pd
from pandas.api import types as ptypes

# An integer/text column where at least this share of values is unique
# is treated as a potential identifier.
ID_UNIQUE_RATIO = 0.95
# With very few rows everything looks unique, so skip the ID check.
MIN_ROWS_FOR_ID_CHECK = 20


def classify_column(series: pd.Series) -> str:
    """Classify a column as 'datetime', 'identifier', 'numerical' or 'categorical'."""
    if ptypes.is_datetime64_any_dtype(series):
        return "datetime"

    if ptypes.is_bool_dtype(series):
        return "categorical"

    non_null = series.dropna()
    if len(non_null) >= MIN_ROWS_FOR_ID_CHECK:
        unique_ratio = non_null.nunique() / len(non_null)
        integer_or_text = ptypes.is_integer_dtype(series) or not ptypes.is_numeric_dtype(series)
        if unique_ratio >= ID_UNIQUE_RATIO and integer_or_text:
            return "identifier"

    if ptypes.is_numeric_dtype(series):
        return "numerical"

    return "categorical"


def get_dataset_overview(df: pd.DataFrame) -> dict:
    """Return dataset-level information and the type of every column."""
    column_types = {col: classify_column(df[col]) for col in df.columns}

    def columns_of(kind):
        return [col for col, t in column_types.items() if t == kind]

    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 3),
        "column_types": column_types,
        "numerical_columns": columns_of("numerical"),
        "categorical_columns": columns_of("categorical"),
        "datetime_columns": columns_of("datetime"),
        "identifier_columns": columns_of("identifier"),
    }
def get_column_profile(data: pd.DataFrame) -> pd.DataFrame:
    """One row per column:type,missing values,unique values."""
    n_rows = len(data)
    rows = []
    for column in data.columns:
        series = data[column]
        missing = int(series.isna().sum())
        unique = int(series.nunique())

        rows.append({
            "column": column,
            "Type": classify_column(series),
            "Data Type": str(series.dtype),
            "Missing %": round(missing / n_rows * 100, 2) if n_rows else 0.0,
            "Unique": unique,
            "Unique %": round(unique / n_rows * 100, 2) if n_rows else 0.0,
        })
    return pd.DataFrame(rows)
def get_numerical_statistics(data: pd.DataFrame, numerical_columns: list) -> pd.DataFrame:
    """Summary statistics for each numerical column (one row per column)."""
    if not numerical_columns:
        return pd.DataFrame()

    data = data[numerical_columns]

    stats = data.agg(["mean", "median", "std", "min", "max"]).T
    stats["Q1"] = data.quantile(0.25)
    stats["Q3"] = data.quantile(0.75)

    stats = stats[["min", "Q1", "median", "mean", "Q3", "max", "std"]]
    stats.columns = ["Min", "Q1", "Median", "Mean", "Q3", "Max", "Std dev"]

    return stats.round(2)