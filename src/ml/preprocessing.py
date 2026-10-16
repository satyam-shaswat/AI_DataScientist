from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.analysis.profiler import classify_column

TEST_SIZE = 0.2
RANDOM_STATE = 42


def prepare_data(df, target, problem_type):
    """Drop rows with no target, pick feature columns and split train/test."""
    data = df.dropna(subset=[target])
    y = data[target]
    X = data.drop(columns=[target])

    numeric_cols, categorical_cols, dropped_cols = [], [], []
    for col in X.columns:
        kind = classify_column(X[col])
        if kind == "numerical":
            numeric_cols.append(col)
        elif kind == "categorical":
            categorical_cols.append(col)
        else:                       # identifier or datetime
            dropped_cols.append(col)

    stratify = None
    if problem_type == "classification" and y.value_counts().min() >= 2:
        stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X[numeric_cols + categorical_cols],
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "dropped_cols": dropped_cols,
    }


def build_preprocessor(numeric_cols, categorical_cols):
    """Unfitted preprocessing recipe: impute then scale / one-hot encode."""
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])