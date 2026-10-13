"""Temporal late-delivery experiments at the post-checkout prediction point."""
from pathlib import Path
import json

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["item_value", "freight_value", "item_count", "seller_count", "mean_weight_g", "interstate_share", "promised_days", "purchase_month", "purchase_weekday"]
CATEGORICAL = ["category", "customer_state", "seller_state"]
FEATURES = NUMERIC + CATEGORICAL


def features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    dates = pd.to_datetime(result.order_purchase_timestamp)
    result["purchase_month"] = dates.dt.month
    result["purchase_weekday"] = dates.dt.dayofweek
    return result[FEATURES]


def load_orders(db_path: Path) -> pd.DataFrame:
    with duckdb.connect(str(db_path), read_only=True) as con:
        return con.execute("SELECT * FROM fact_orders ORDER BY order_purchase_timestamp, order_id").df()


def train_model(db_path: Path, artifact_dir: Path) -> dict:
    all_orders = load_orders(db_path)
    snapshot = all_orders.order_purchase_timestamp.max()
    # Recent purchases are excluded to reduce right-censoring of the delivery label.
    eligible = all_orders.loc[all_orders.is_late.notna() & all_orders.item_count.gt(0)
                              & all_orders.order_purchase_timestamp.le(snapshot - pd.Timedelta(days=45))].copy()
    dates = eligible.order_purchase_timestamp
    if len(eligible) < 200 or eligible.is_late.nunique() < 2:
        raise ValueError("Need at least 200 matured labeled orders with both late and on-time outcomes")
    split = dates.quantile(.8)
    # Only outcomes already observed before the evaluation window are allowed in training.
    train = eligible.loc[(dates < split) & (eligible.order_delivered_customer_date < split)]
    test = eligible.loc[dates >= split]
    if min(len(train), len(test)) < 30 or train.is_late.nunique() < 2 or test.is_late.nunique() < 2:
        raise ValueError("Temporal train/test windows need at least 30 orders and both outcome classes")
    x_train, x_test = features(train), features(test)
    y_train, y_test = train.is_late.astype(int), test.is_late.astype(int)
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler())]), NUMERIC),
        ("category", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])
    candidates = {"prevalence_baseline": DummyClassifier(strategy="prior"),
                  "logistic_regression": LogisticRegression(max_iter=1500, random_state=42),
                  "random_forest": RandomForestClassifier(n_estimators=180, max_depth=9, min_samples_leaf=12, random_state=42, n_jobs=-1)}
    metrics, fitted = {}, {}
    for name, classifier in candidates.items():
        pipeline = Pipeline([("preprocess", preprocessing), ("classifier", classifier)])
        pipeline.fit(x_train, y_train)
        probability = pipeline.predict_proba(x_test)[:, 1]
        pred = probability >= .5
        metrics[name] = {"roc_auc": float(roc_auc_score(y_test, probability)),
                         "average_precision": float(average_precision_score(y_test, probability)),
                         "brier_score": float(brier_score_loss(y_test, probability)),
                         "precision_at_0.5": float(precision_score(y_test, pred, zero_division=0)),
                         "recall_at_0.5": float(recall_score(y_test, pred, zero_division=0))}
        fitted[name] = pipeline
    # Predeclared deployment candidate; do not select or tune using held-out outcomes.
    name = "logistic_regression"
    pipeline = fitted[name]
    importance = permutation_importance(pipeline, x_test, y_test, scoring="average_precision", n_repeats=5, random_state=42)
    report = {"candidate": name, "source": None, "train_orders": len(train), "test_orders": len(test),
              "split_timestamp": str(split), "train_start": str(train.order_purchase_timestamp.min()),
              "test_end": str(test.order_purchase_timestamp.max()), "test_late_rate": float(y_test.mean()),
              "threshold": .5, "metrics": metrics,
              "importance": sorted([{ "feature": col, "average_precision_drop": float(value)} for col, value in zip(FEATURES, importance.importances_mean)], key=lambda row: row["average_precision_drop"], reverse=True),
              "limitations": ["Delivered orders only: canceled and unresolved orders are not represented.",
                              "45-day maturity buffer reduces but does not eliminate censoring.",
                              "Scores are not guaranteed calibrated outside the historical population.",
                              "Feature importance is global, not causal or a per-order explanation.",
                              "Features assume item and promised-date information is available immediately after checkout."]}
    with duckdb.connect(str(db_path), read_only=True) as con:
        report["source"] = con.execute("SELECT source FROM metadata").fetchone()[0]
        report["database_built_at"] = con.execute("SELECT built_at FROM metadata").fetchone()[0]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, artifact_dir / "late_delivery.joblib")
    (artifact_dir / "model_report.json").write_text(json.dumps(report, indent=2))
    scored = test[["order_id", "order_purchase_timestamp", "is_late"]].copy()
    scored["late_probability"] = pipeline.predict_proba(x_test)[:, 1]
    scored.to_csv(artifact_dir / "test_predictions.csv", index=False)
    return report


def explain_order(pipeline, row: pd.DataFrame) -> pd.DataFrame:
    """Exact additive log-odds terms for the logistic candidate, not causal effects."""
    preprocess = pipeline.named_steps["preprocess"]
    transformed = preprocess.transform(features(row))
    dense = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
    contributions = dense[0] * pipeline.named_steps["classifier"].coef_[0]
    names = preprocess.get_feature_names_out()
    result = pd.DataFrame({"feature": names, "log_odds_contribution": contributions})
    return result.iloc[np.argsort(-np.abs(contributions))].head(8)
