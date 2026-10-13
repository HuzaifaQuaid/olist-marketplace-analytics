"""Customer repurchase prediction with a fixed observation horizon, not an open-ended label."""
from pathlib import Path
import json

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HORIZON_DAYS = 90
NUMERIC = ["item_value", "freight_value", "item_count", "seller_count", "mean_weight_g", "interstate_share", "promised_days", "purchase_month", "purchase_weekday"]
CATEGORICAL = ["category", "customer_state"]
FEATURES = NUMERIC + CATEGORICAL


def first_purchase_table(orders: pd.DataFrame, horizon_days: int = HORIZON_DAYS) -> pd.DataFrame:
    """One row per customer: features known at the first purchase plus a censoring-aware label."""
    sales = orders.loc[orders.is_sales_order & orders.item_count.gt(0) & orders.customer_unique_id.notna()
                       & orders.order_purchase_timestamp.notna()].copy()
    if sales.empty:
        return pd.DataFrame(columns=FEATURES + ["customer_unique_id", "first_purchase", "repeat_within_horizon"])
    ordered = sales.sort_values(["customer_unique_id", "order_purchase_timestamp", "order_id"])
    ordered = ordered.assign(seq=ordered.groupby("customer_unique_id").cumcount())
    first = ordered.loc[ordered.seq.eq(0)].copy()
    second_ts = ordered.loc[ordered.seq.eq(1)].set_index("customer_unique_id").order_purchase_timestamp
    snapshot = sales.order_purchase_timestamp.max()
    dates = first.order_purchase_timestamp
    first["purchase_month"] = dates.dt.month
    first["purchase_weekday"] = dates.dt.dayofweek
    first["second_purchase"] = first.customer_unique_id.map(second_ts)
    first["first_purchase"] = dates
    first["repeat_within_horizon"] = (first.second_purchase.notna() & (first.second_purchase <= dates + pd.Timedelta(days=horizon_days)))
    # Only customers observed for the full horizon can be labelled without censoring bias.
    return first.loc[dates + pd.Timedelta(days=horizon_days) <= snapshot].copy()


def train_repurchase(db_path: Path, artifact_dir: Path, horizon_days: int = HORIZON_DAYS) -> dict:
    with duckdb.connect(str(db_path), read_only=True) as con:
        orders = con.execute("SELECT * FROM fact_orders").df()
        source, built_at = con.execute("SELECT source, built_at FROM metadata").fetchone()
    table = first_purchase_table(orders, horizon_days)
    if len(table) < 500 or table.repeat_within_horizon.nunique() < 2:
        raise ValueError("Need at least 500 matured customers with both outcomes; find the customer table first")
    dates = table.first_purchase
    split = dates.quantile(.8)
    train, test = table.loc[dates < split], table.loc[dates >= split]
    if min(len(train), len(test)) < 50:
        raise ValueError("Temporal windows need at least 50 customers each")
    x_train, x_test = train[FEATURES], test[FEATURES]
    y_train, y_test = train.repeat_within_horizon.astype(int), test.repeat_within_horizon.astype(int)
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler())]), NUMERIC),
        ("category", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])
    candidates = {"prevalence_baseline": DummyClassifier(strategy="prior"),
                  "logistic_regression": LogisticRegression(max_iter=1500)}
    metrics, pipelines = {}, {}
    for name, classifier in candidates.items():
        pipeline = Pipeline([("preprocess", preprocessing), ("classifier", classifier)])
        pipeline.fit(x_train, y_train)
        probability = pipeline.predict_proba(x_test)[:, 1]
        deciles = pd.qcut(pd.Series(probability).rank(method="first"), 10, labels=False)
        top = deciles == 9
        metrics[name] = {"roc_auc": float(roc_auc_score(y_test, probability)), "average_precision": float(average_precision_score(y_test, probability)),
                         "brier_score": float(brier_score_loss(y_test, probability)), "base_rate": float(y_test.mean()),
                         "top_decile_rate": float(y_test.to_numpy()[top].mean()) if top.any() else None,
                         "lift_at_top_decile": float(y_test.to_numpy()[top].mean() / y_test.mean()) if top.any() and y_test.mean() else None}
        pipelines[name] = pipeline
    pipeline = pipelines["logistic_regression"]
    probability = pipeline.predict_proba(x_test)[:, 1]
    calibration = pd.DataFrame({"predicted": probability, "actual": y_test.to_numpy()})
    calibration["bucket"] = pd.qcut(calibration.predicted.rank(method="first"), 5, labels=False)
    curve = calibration.groupby("bucket", as_index=False).agg(predicted=("predicted", "mean"), actual=("actual", "mean"), customers=("actual", "size"))
    report = {"source": source, "database_built_at": built_at, "horizon_days": horizon_days, "candidate": "logistic_regression",
              "matured_customers": len(table), "overall_repeat_rate": float(table.repeat_within_horizon.mean()),
              "train_customers": len(train), "test_customers": len(test), "split_timestamp": str(split),
              "test_repeat_rate": float(y_test.mean()), "metrics": metrics,
              "calibration": curve.to_dict("records"),
              "limitations": [f"The label is 'second observed purchase within {horizon_days} days of the first', so it measures historical repetition, not a guaranteed future action.",
                              "Cancelled and unavailable orders are ignored; guest or unidentified customers cannot be scored.",
                              "Customers without the full observation horizon are excluded, which removes the most recent acquisition cohorts.",
                              "Probabilities are calibrated on the population level, but the event is rare, so individual scores still carry wide uncertainty.",
                              "Predictors describe the first order and its calendar context only; marketing contact, price sensitivity and browsing are unavailable."]}
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, artifact_dir / "repurchase.joblib")
    (artifact_dir / "repurchase_report.json").write_text(json.dumps(report, indent=2))
    scored = test[["customer_unique_id", "first_purchase", "repeat_within_horizon"]].copy()
    scored["repeat_probability"] = probability
    scored.to_csv(artifact_dir / "repurchase_predictions.csv", index=False)
    return report


def load_repurchase_report(artifact_dir: Path) -> dict | None:
    path = artifact_dir / "repurchase_report.json"
    return json.loads(path.read_text()) if path.exists() else None
