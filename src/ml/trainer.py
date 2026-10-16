import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.ml.preprocessing import build_preprocessor, RANDOM_STATE


def get_classification_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, eval_metric="logloss", random_state=RANDOM_STATE
        ),
    }


def train_classification(prepared):
    """Train every classification model and return a comparison table."""
    # Turn labels such as "No"/"Yes" into 0/1 (XGBoost needs numbers).
    encoder = LabelEncoder()
    encoder.fit(pd.concat([prepared["y_train"], prepared["y_test"]]))
    y_train = encoder.transform(prepared["y_train"])
    y_test = encoder.transform(prepared["y_test"])

    n_classes = len(encoder.classes_)
    average = "binary" if n_classes == 2 else "weighted"

    rows = []
    fitted = {}

    for name, model in get_classification_models().items():
        pipeline = Pipeline([
            ("prep", build_preprocessor(
                prepared["numeric_cols"], prepared["categorical_cols"]
            )),
            ("model", model),
        ])
        pipeline.fit(prepared["X_train"], y_train)

        predictions = pipeline.predict(prepared["X_test"])
        probabilities = pipeline.predict_proba(prepared["X_test"])

        try:
            if n_classes == 2:
                auc = roc_auc_score(y_test, probabilities[:, 1])
            else:
                auc = roc_auc_score(
                    y_test, probabilities, multi_class="ovr", average="weighted"
                )
        except ValueError:
            auc = None

        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, average=average, zero_division=0),
            "Recall": recall_score(y_test, predictions, average=average, zero_division=0),
            "F1": f1_score(y_test, predictions, average=average, zero_division=0),
            "ROC-AUC": auc,
        })
        fitted[name] = pipeline

    results = pd.DataFrame(rows).round(3)

    # What a "model" that always predicts the most common class would score.
    majority_class = pd.Series(y_train).mode()[0]
    baseline_accuracy = round(float((y_test == majority_class).mean()), 3)

    return {
        "results": results,
        "models": fitted,
        "classes": list(encoder.classes_),
        "baseline_accuracy": baseline_accuracy,
    }