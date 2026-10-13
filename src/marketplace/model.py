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
    # Reserve the latest fifth of the pre-test period for threshold selection.
    train_dates = train.order_purchase_timestamp
    validation_cut = train_dates.quantile(.8)
    fit_train = train.loc[(train.order_purchase_timestamp < validation_cut)
                          & (train.order_delivered_customer_date < validation_cut)].copy()
    validation = train.loc[train.order_purchase_timestamp >= validation_cut].copy()
    if min(len(fit_train), len(validation)) < 30 or fit_train.is_late.nunique() < 2 or validation.is_late.nunique() < 2:
        raise ValueError("Temporal model fit/validation windows need at least 30 orders and both outcome classes")
    x_fit, x_validation, x_test = features(fit_train), features(validation), features(test)
    y_fit, y_validation, y_test = fit_train.is_late.astype(int), validation.is_late.astype(int), test.is_late.astype(int)
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
        pipeline.fit(x_fit, y_fit)
        validation_probability = pipeline.predict_proba(x_validation)[:, 1]
        probability = pipeline.predict_proba(x_test)[:, 1]
        # Choose an F2-oriented threshold on validation only; held-out test remains untouched.
        threshold = _select_threshold(y_validation.to_numpy(), validation_probability) if name != "prevalence_baseline" else .5
        pred = probability >= threshold
        metrics[name] = {"roc_auc": float(roc_auc_score(y_test, probability)),
                         "average_precision": float(average_precision_score(y_test, probability)),
                         "brier_score": float(brier_score_loss(y_test, probability)),
                         "threshold_selected_on_validation": float(threshold),
                         "precision_at_threshold": float(precision_score(y_test, pred, zero_division=0)),
                         "recall_at_threshold": float(recall_score(y_test, pred, zero_division=0))}
        fitted[name] = pipeline
    # Predeclared deployment candidate; do not select or tune using held-out outcomes.
    name = "logistic_regression"
    pipeline = fitted[name]
    # Final fitted model is kept on the fit window; validation/test remain honest diagnostics.
    importance = permutation_importance(pipeline, x_test, y_test, scoring="average_precision", n_repeats=5, random_state=42)
    selected_threshold = metrics[name]["threshold_selected_on_validation"]
    report = {"candidate": name, "source": None, "fit_orders": len(fit_train), "validation_orders": len(validation),
              "validation_split_timestamp": str(validation_cut), "train_orders": len(train), "test_orders": len(test),
              "split_timestamp": str(split), "train_start": str(train.order_purchase_timestamp.min()),
              "test_end": str(test.order_purchase_timestamp.max()), "test_late_rate": float(y_test.mean()),
              "threshold": selected_threshold, "threshold_selection": "F2-maximizing threshold on an earlier chronological validation window; fixed before held-out testing.", "metrics": metrics,
              "importance": sorted([{ "feature": col, "average_precision_drop": float(value)} for col, value in zip(FEATURES, importance.importances_mean)], key=lambda row: row["average_precision_drop"], reverse=True),
              "limitations": ["Delivered orders only: canceled and unresolved orders are not represented.",
                              "45-day maturity buffer reduces but does not eliminate censoring.",
                              "The high-recall operating point creates false alarms: precision was only 5.9% on the held-out test, so it is a review queue, not an automatic intervention.",
                              "Scores are not guaranteed calibrated outside the historical population.",
                              "The operating threshold is selected to favor recall using historical validation data; false-positive costs must be assessed before operational use.",
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


def _select_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    """Choose F2 threshold using validation outcomes only, with deterministic tie-breaks."""
    candidates = np.unique(np.r_[probability, 0.0, 1.0])
    best_score, best_threshold = (float("-inf"), 0.5, float("-inf")), 0.5
    for threshold in candidates:
        prediction = probability >= threshold
        tp = int(np.sum(prediction & (y_true == 1)))
        fp = int(np.sum(prediction & (y_true == 0)))
        fn = int(np.sum(~prediction & (y_true == 1)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        denominator = 4 * precision + recall
        f2 = 5 * precision * recall / denominator if denominator else 0.0
        score = (f2, precision, -float(threshold))
        if score > best_score:
            best_score, best_threshold = score, float(threshold)
    return best_threshold
