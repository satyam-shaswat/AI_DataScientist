import pandas as pd

from src.analysis.profiler import classify_column

# Missing-value findings at or above this percentage are warnings;
# smaller ones are reported as info.
MISSING_WARNING_PCT = 5

# Standard IQR rule: values beyond 1.5 x IQR from the quartiles.
IQR_MULTIPLIER = 1.5

# A categorical column with more distinct values than this is "high cardinality".
HIGH_CARDINALITY_UNIQUE = 50


def make_finding(check, message, column=None, value=None, severity="warning"):
    """Build one finding as a dictionary."""
    return {
        "check": check,
        "severity": severity,
        "column": column,
        "value": value,
        "message": message,
    }


def check_missing_values(df: pd.DataFrame) -> list:
    findings = []
    n_rows = len(df)

    for col in df.columns:
        missing = int(df[col].isna().sum())
        if missing > 0:
            pct = round(missing / n_rows * 100, 2)
            severity = "warning" if pct >= MISSING_WARNING_PCT else "info"
            findings.append(make_finding(
                "missing_values",
                f"Column '{col}' contains {pct}% missing values "
                f"({missing} of {n_rows} rows).",
                column=col,
                value=pct,
                severity=severity,
            ))
    return findings


def check_duplicates(df: pd.DataFrame) -> list:
    n_duplicates = int(df.duplicated().sum())
    if n_duplicates == 0:
        return []

    pct = round(n_duplicates / len(df) * 100, 2)
    return [make_finding(
        "duplicate_rows",
        f"The dataset contains {n_duplicates} duplicate rows ({pct}% of rows).",
        value=n_duplicates,
    )]


def check_constant_columns(df: pd.DataFrame) -> list:
    findings = []
    for col in df.columns:
        if df[col].nunique(dropna=True) <= 1:
            findings.append(make_finding(
                "constant_column",
                f"Column '{col}' has at most one distinct value, "
                "so it carries no information for analysis.",
                column=col,
            ))
    return findings


def check_outliers(df: pd.DataFrame) -> list:
    findings = []

    for col in df.columns:
        if classify_column(df[col]) != "numerical":
            continue

        values = df[col].dropna()
        if values.nunique() <= 2:
            continue                      # binary-like column: skip

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue                      # fences cannot be defined

        lower = q1 - IQR_MULTIPLIER * iqr
        upper = q3 + IQR_MULTIPLIER * iqr
        n_outliers = int(((values < lower) | (values > upper)).sum())

        if n_outliers > 0:
            pct = round(n_outliers / len(values) * 100, 2)
            findings.append(make_finding(
                "outliers",
                f"Column '{col}' contains {n_outliers} potential outliers "
                f"({pct}% of values) outside the range "
                f"{lower:.2f} to {upper:.2f} (IQR method).",
                column=col,
                value=n_outliers,
            ))
    return findings


def check_cardinality(df: pd.DataFrame) -> list:
    findings = []

    for col in df.columns:
        kind = classify_column(df[col])
        n_unique = int(df[col].nunique())

        if kind == "identifier":
            findings.append(make_finding(
                "potential_id",
                f"Column '{col}' looks like an identifier (almost every value "
                "is unique). It should be excluded from modelling.",
                column=col,
                value=n_unique,
                severity="info",
            ))
        elif kind == "categorical" and n_unique > HIGH_CARDINALITY_UNIQUE:
            findings.append(make_finding(
                "high_cardinality",
                f"Column '{col}' has very high cardinality "
                f"({n_unique} distinct values), which makes it hard to "
                "encode and chart.",
                column=col,
                value=n_unique,
            ))
    return findings

def run_quality_checks(df: pd.DataFrame) -> list:
    """Run every quality check and return all findings in one list."""
    findings = []
    findings += check_missing_values(df)
    findings += check_duplicates(df)
    findings += check_constant_columns(df)
    findings += check_outliers(df)
    findings += check_cardinality(df)
    return findings