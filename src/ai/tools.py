import pandas as pd
from pandas.api import types as ptypes

from src.analysis.profiler import classify_column, get_dataset_overview
from src.analysis.quality import IQR_MULTIPLIER

MAX_GROUPS = 20
MAX_EXAMPLES = 5
AGGREGATIONS = ["mean", "median", "sum", "min", "max", "count"]


# ---------- helpers ----------

def _py(value):
    """Convert NumPy/Pandas values to plain Python so they are JSON-safe."""
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, (int, str, bool)):
        return value
    return str(value)


def _resolve_column(df, column):
    """Return (real_column_name, None) or (None, error_message)."""
    if column in df.columns:
        return column, None
    matches = [c for c in df.columns if str(c).lower() == str(column).lower()]
    if len(matches) == 1:
        return matches[0], None
    return None, f"Column '{column}' not found. Available columns: {list(df.columns)}"


def _is_numeric(series):
    return ptypes.is_numeric_dtype(series) and not ptypes.is_bool_dtype(series)


# ---------- tools ----------

def get_dataset_summary(df):
    """Overview of the whole dataset."""
    overview = get_dataset_overview(df)
    missing = df.isna().sum()
    return {
        "rows": overview["rows"],
        "columns": overview["columns"],
        "numerical_columns": overview["numerical_columns"],
        "categorical_columns": overview["categorical_columns"],
        "datetime_columns": overview["datetime_columns"],
        "identifier_columns": overview["identifier_columns"],
        "columns_with_missing_values": {c: int(n) for c, n in missing.items() if n > 0},
        "duplicate_rows": int(df.duplicated().sum()),
    }


def get_column_statistics(df, column):
    """Statistics for one column (numbers for numerical, top values otherwise)."""
    col, error = _resolve_column(df, column)
    if error:
        return {"error": error}

    series = df[col]
    non_missing = int(series.notna().sum())
    result = {
        "column": col,
        "type": classify_column(series),
        "non_missing": non_missing,
        "missing": int(series.isna().sum()),
    }
    if non_missing == 0:
        result["note"] = "This column has no values."
        return result

    if result["type"] == "numerical":
        result.update({
            "mean": _py(series.mean()),
            "median": _py(series.median()),
            "std": _py(series.std()),
            "min": _py(series.min()),
            "q1": _py(series.quantile(0.25)),
            "q3": _py(series.quantile(0.75)),
            "max": _py(series.max()),
        })
    else:
        result["unique_values"] = int(series.nunique())
        result["top_values"] = [
            {"value": _py(v), "count": int(n), "percent": _py(n / non_missing * 100)}
            for v, n in series.value_counts().head(5).items()
        ]
    return result


def group_by_column(df, group_column, value_column, aggregation="mean"):
    """Compare groups. Numeric value column: aggregate it per group.
    Categorical value column: percentage of each value within each group."""
    group_col, error = _resolve_column(df, group_column)
    if error:
        return {"error": error}
    value_col, error = _resolve_column(df, value_column)
    if error:
        return {"error": error}
    if group_col == value_col:
        return {"error": "group_column and value_column must be different columns."}

    data = df[[group_col, value_col]].dropna()
    if data.empty:
        return {"error": "No rows remain after removing missing values."}
    if data[group_col].nunique() > MAX_GROUPS:
        return {"error": (
            f"'{group_col}' has {data[group_col].nunique()} distinct values, which is "
            f"too many to compare (limit {MAX_GROUPS}). Choose a column with fewer groups."
        )}

    if _is_numeric(data[value_col]):
        if aggregation not in AGGREGATIONS:
            return {"error": f"Unknown aggregation '{aggregation}'. Use one of {AGGREGATIONS}."}

        grouped = data.groupby(group_col)[value_col].agg(aggregation).to_frame("value")
        grouped["rows"] = data.groupby(group_col).size()
        grouped = grouped.sort_values("value", ascending=False)

        results = [
            {"group": _py(g), "value": _py(row["value"]), "rows": int(row["rows"])}
            for g, row in grouped.iterrows()
        ]
        return {
            "group_column": group_col,
            "value_column": value_col,
            "aggregation": aggregation,
            "results": results,
            "highest_group": results[0]["group"],
            "lowest_group": results[-1]["group"],
        }

    if data[value_col].nunique() > MAX_GROUPS:
        return {"error": f"'{value_col}' has too many distinct values to break down."}

    shares = pd.crosstab(data[group_col], data[value_col], normalize="index") * 100
    counts = data.groupby(group_col).size()
    results = [
        {
            "group": _py(g),
            "rows": int(counts[g]),
            "percent": {str(k): _py(v) for k, v in row.items()},
        }
        for g, row in shares.iterrows()
    ]
    return {
        "group_column": group_col,
        "value_column": value_col,
        "note": f"For each group, the percent of its rows having each value of '{value_col}'.",
        "results": results,
    }


def calculate_correlation(df, column1, column2):
    """Pearson correlation between two numerical columns."""
    col1, error = _resolve_column(df, column1)
    if error:
        return {"error": error}
    col2, error = _resolve_column(df, column2)
    if error:
        return {"error": error}
    if not (_is_numeric(df[col1]) and _is_numeric(df[col2])):
        return {"error": "Both columns must be numerical to calculate a correlation."}

    data = df[[col1, col2]].dropna()
    if len(data) < 3:
        return {"error": "Not enough rows with values in both columns."}
    if data[col1].nunique() < 2 or data[col2].nunique() < 2:
        return {"error": "One of the columns is constant, so a correlation is undefined."}

    r = data[col1].corr(data[col2])
    size = abs(r)
    if size < 0.1:
        strength = "negligible"
    elif size < 0.3:
        strength = "weak"
    elif size < 0.5:
        strength = "moderate"
    elif size < 0.8:
        strength = "strong"
    else:
        strength = "very strong"

    return {
        "column1": col1,
        "column2": col2,
        "correlation": _py(r),
        "strength": strength,
        "direction": "positive" if r > 0 else "negative",
        "rows_used": int(len(data)),
        "note": "Pearson correlation measures linear association, not causation.",
    }


def detect_outliers(df, column):
    """Potential outliers in a numerical column using the IQR method."""
    col, error = _resolve_column(df, column)
    if error:
        return {"error": error}
    if not _is_numeric(df[col]):
        return {"error": f"Column '{col}' is not numerical, so the IQR method does not apply."}

    values = df[col].dropna()
    if values.nunique() <= 2:
        return {"column": col, "note": "Two or fewer distinct values; the IQR method is not meaningful here."}

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return {"column": col, "note": "The IQR is 0, so outlier fences cannot be defined."}

    lower = q1 - IQR_MULTIPLIER * iqr
    upper = q3 + IQR_MULTIPLIER * iqr
    outliers = values[(values < lower) | (values > upper)]
    median = values.median()
    examples = sorted(outliers, key=lambda v: abs(v - median), reverse=True)[:MAX_EXAMPLES]

    return {
        "column": col,
        "method": f"IQR ({IQR_MULTIPLIER} x IQR beyond the quartiles)",
        "q1": _py(q1),
        "q3": _py(q3),
        "lower_fence": _py(lower),
        "upper_fence": _py(upper),
        "outliers_found": int(len(outliers)),
        "percent_of_values": _py(len(outliers) / len(values) * 100),
        "most_extreme_examples": [_py(v) for v in examples],
    }


# ---------- the controlled entry point ----------

TOOLS = {
    "get_dataset_summary": get_dataset_summary,
    "get_column_statistics": get_column_statistics,
    "group_by_column": group_by_column,
    "calculate_correlation": calculate_correlation,
    "detect_outliers": detect_outliers,
}


def call_tool(df, name, arguments=None):
    """Run one allowed tool. Never raises: problems come back as {'error': ...}."""
    arguments = arguments or {}
    if name not in TOOLS:
        return {"error": f"Unknown tool '{name}'. Available tools: {list(TOOLS)}"}
    try:
        return TOOLS[name](df, **arguments)
    except TypeError as e:
        return {"error": f"Invalid arguments for {name}: {e}"}
    except Exception as e:
        return {"error": f"Tool {name} failed: {e}"}