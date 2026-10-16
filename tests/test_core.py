"""Tests for the deterministic core of AI Data Scientist.

Run from the project root with:  python -m pytest
Every expected value below was worked out by hand, not copied from the code's output.
"""
import numpy as np
import pandas as pd
import pytest

from src.ai.tools import call_tool
from src.analysis.profiler import classify_column
from src.analysis.quality import check_outliers
from src.data.loader import convert_numeric_text_columns
from src.ml.preprocessing import build_preprocessor, prepare_data
from src.ml.problem_type import check_problem_choice, check_target, detect_problem_type


@pytest.fixture
def small_df():
    return pd.DataFrame({
        "contract": ["Monthly"] * 4 + ["Yearly"] * 4,
        "churn": ["Yes", "Yes", "Yes", "No", "No", "No", "No", "Yes"],
        "charges": [70, 80, 90, 100, 20, 30, 40, 50],
        "tenure": [1, 2, 3, 4, 40, 50, 60, 70],
    })


# ---------- loader ----------

def test_numeric_text_conversion():
    df = pd.DataFrame({
        "Name": ["Asha", "Ravi", "Meena"],
        "Charges": ["100.5", " ", "200"],   # blank is allowed -> missing value
        "Score": ["5", "x", "9"],           # a real non-number -> must NOT convert
    })
    out, converted = convert_numeric_text_columns(df)
    assert converted == ["Charges"]
    assert out["Charges"].isna().sum() == 1
    assert pd.api.types.is_numeric_dtype(out["Charges"])
    assert not pd.api.types.is_numeric_dtype(out["Score"])
    assert not pd.api.types.is_numeric_dtype(out["Name"])


# ---------- profiler ----------

def test_classify_column_types():
    assert classify_column(pd.Series(range(1, 31))) == "identifier"
    assert classify_column(pd.Series([30000.5 + i * 100 for i in range(30)])) == "numerical"
    assert classify_column(pd.Series(["Delhi", "Pune", "Mumbai"] * 10)) == "categorical"
    assert classify_column(pd.Series(pd.date_range("2024-01-01", periods=30))) == "datetime"


# ---------- data quality ----------

def test_outlier_found_with_iqr():
    # Q1 = 11, Q3 = 13, IQR = 2, fences = 11 - 3 = 8 and 13 + 3 = 16
    df = pd.DataFrame({"value": [10 + i % 5 for i in range(99)] + [500]})
    findings = check_outliers(df)
    assert len(findings) == 1
    assert findings[0]["value"] == 1
    assert "8.00 to 16.00" in findings[0]["message"]


def test_binary_column_is_not_flagged_as_outliers():
    df = pd.DataFrame({"flag": [0] * 80 + [1] * 20})
    assert check_outliers(df) == []


# ---------- problem type ----------

def test_detect_problem_type():
    assert detect_problem_type(pd.Series(["Yes", "No", "Yes"])) == "classification"
    assert detect_problem_type(pd.Series([0, 1, 0, 1])) == "classification"
    assert detect_problem_type(pd.Series(range(100), dtype=float)) == "regression"


def test_check_target_rejects_unusable_columns():
    assert check_target(pd.Series([1, 1, 1])) is not None            # constant
    assert check_target(pd.Series(range(1, 31))) is not None         # identifier
    assert check_target(pd.Series(["Yes", "No"] * 10)) is None       # fine


def test_check_problem_choice_rejects_wrong_choice():
    assert check_problem_choice(pd.Series(range(500)), "classification") is not None
    assert check_problem_choice(pd.Series(["Yes", "No"] * 10), "regression") is not None
    assert check_problem_choice(pd.Series(["Yes", "No"] * 10), "classification") is None


# ---------- AI analyst tools ----------

def test_group_by_percent_and_mean(small_df):
    split = call_tool(small_df, "group_by_column",
                      {"group_column": "contract", "value_column": "churn"})
    assert split["results"][0]["group"] == "Monthly"
    assert split["results"][0]["percent"]["Yes"] == 75.0

    means = call_tool(small_df, "group_by_column",
                      {"group_column": "contract", "value_column": "charges", "aggregation": "mean"})
    assert means["highest_group"] == "Monthly"
    assert means["results"][0]["value"] == 85.0


def test_correlation_tool(small_df):
    result = call_tool(small_df, "calculate_correlation",
                       {"column1": "charges", "column2": "tenure"})
    assert result["correlation"] == pytest.approx(-0.7822, abs=1e-3)
    assert result["strength"] == "strong"
    assert result["direction"] == "negative"


def test_tools_return_errors_instead_of_crashing(small_df):
    bad_column = call_tool(small_df, "group_by_column",
                           {"group_column": "nope", "value_column": "churn"})
    assert "error" in bad_column
    unknown_tool = call_tool(small_df, "delete_everything", {})
    assert "Unknown tool" in unknown_tool["error"]


# ---------- preprocessing ----------

def test_preprocessor_imputes_scales_and_encodes():
    # np.nan (what read_csv produces) is imputed; a Python None in a text column is not
    df = pd.DataFrame({"age": [20, 30, None, 40], "city": ["Delhi", "Pune", "Delhi", np.nan]})
    out = build_preprocessor(["age"], ["city"]).fit_transform(df)
    expected = np.array([[-1.41, 1, 0], [0, 0, 1], [0, 1, 0], [1.41, 1, 0]])
    assert np.allclose(out, expected, atol=0.01)


def test_prepare_data_drops_ids_and_stratifies():
    df = pd.DataFrame({
        "id": range(1, 21),
        "age": [20 + i % 5 for i in range(20)],
        "city": ["Delhi", "Pune", "Mumbai", "Delhi"] * 5,
        "churn": ["Yes", "No"] * 10,
    })
    d = prepare_data(df, "churn", "classification")
    assert d["dropped_cols"] == ["id"]
    assert d["numeric_cols"] == ["age"]
    assert d["categorical_cols"] == ["city"]
    assert (len(d["X_train"]), len(d["X_test"])) == (16, 4)
    assert d["y_test"].value_counts().to_dict() == {"No": 2, "Yes": 2}