"""Monthly sales forecasting with rolling-origin backtests, never random splits."""
from pathlib import Path
import json

import duckdb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Naive, seasonal-naive and calendar/lag models share one feature definition.
MODEL_FEATURES = ["t", "sin", "cos", "lag1", "lag2", "lag3", "roll3", "lag12", "roll12"]
MODELS = ["naive_last_value", "seasonal_naive_12m", "calendar_lag_model"]
HORIZON = 1
MIN_COVERAGE = 0.9


def monthly_series(orders: pd.DataFrame) -> pd.DataFrame:
    """Complete calendar months of item sales; partial boundary months are dropped."""
    sales = orders.loc[orders.is_sales_order & orders.item_count.gt(0)].dropna(subset=["order_purchase_timestamp"]).copy()
    if sales.empty:
        return pd.DataFrame(columns=["month", "item_sales", "orders", "coverage"])
    sales["month"] = sales.order_purchase_timestamp.dt.to_period("M").dt.to_timestamp()
    sales["day"] = sales.order_purchase_timestamp.dt.normalize()
    grouped = sales.groupby("month").agg(item_sales=("item_value", "sum"), orders=("order_id", "count"), observed_days=("day", "nunique")).reset_index()
    grouped = grouped.sort_values("month").reset_index(drop=True)
    grouped["coverage"] = grouped.observed_days / grouped.month.dt.days_in_month
    # Leading and trailing months are usually partial; keeping them would distort trend and seasonality.
    while len(grouped) and grouped.coverage.iloc[-1] < MIN_COVERAGE:
        grouped = grouped.iloc[:-1]
    while len(grouped) and grouped.coverage.iloc[0] < MIN_COVERAGE:
        grouped = grouped.iloc[1:].reset_index(drop=True)
    return grouped[["month", "item_sales", "orders", "coverage"]].reset_index(drop=True)


def _row(values: np.ndarray, position: int) -> dict:
    """Features for the target at position, using only values strictly before it."""
    history = values[:position]
    return {"position": position, "t": float(position),
            "sin": float(np.sin(2 * np.pi * position / 12)), "cos": float(np.cos(2 * np.pi * position / 12)),
            "lag1": float(values[position - 1]), "lag2": float(values[position - 2]), "lag3": float(values[position - 3]),
            "roll3": float(np.mean(values[position - 3:position])),
            "lag12": float(values[position - 12]) if position >= 12 else np.nan,
            "roll12": float(np.mean(history[-12:])) if position >= 12 else np.nan}


def supervised(series: pd.DataFrame) -> pd.DataFrame:
    values = series.item_sales.to_numpy(dtype=float)
    rows = [_row(values, position) | {"y": float(values[position])} for position in range(3, len(values))]
    return pd.DataFrame(rows)


def _model() -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler()),
                     ("ridge", Ridge(alpha=1.0))])


def _predict(name: str, trained, history: np.ndarray, position: int, feature_row: pd.DataFrame) -> float:
    if name == "naive_last_value":
        return float(history[position - 1])
    if name == "seasonal_naive_12m":
        return float(history[position - 12]) if position >= 12 else float(history[position - 1])
    return float(trained.predict(feature_row[MODEL_FEATURES])[0])


def backtest(series: pd.DataFrame, min_train: int) -> pd.DataFrame:
    """Expanding-window, one-step-ahead evaluation across every available origin."""
    values = series.item_sales.to_numpy(dtype=float)
    feature_frame = supervised(series)
    records = []
    for origin in range(min_train, len(values)):
        train = feature_frame.loc[feature_frame.position < origin]
        target = feature_frame.loc[feature_frame.position.eq(origin)]
        if train.empty or target.empty:
            continue
        trained = _model()
        trained.fit(train[MODEL_FEATURES], train.y)
        for name in MODELS:
            records.append({"origin": origin, "month": series.month.iloc[origin], "model": name,
                            "actual": float(values[origin]),
                            "prediction": _predict(name, trained, values, origin, target)})
    return pd.DataFrame(records)


def _metrics(frame: pd.DataFrame) -> dict:
    actual, prediction = frame.actual.to_numpy(), frame.prediction.to_numpy()
    error = actual - prediction
    safe = actual != 0
    return {"folds": len(frame), "mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2))),
            "mape": float(np.mean(np.abs(error[safe] / actual[safe]))) if safe.any() else None}


def _interval(history_errors: np.ndarray, prediction: float) -> dict:
    return {"low_80": float(prediction + np.quantile(history_errors, .10)),
            "high_80": float(prediction + np.quantile(history_errors, .90)),
            "low_95": float(prediction + np.quantile(history_errors, .025)),
            "high_95": float(prediction + np.quantile(history_errors, .975))}


def train_forecast(db_path: Path, artifact_dir: Path) -> dict:
    with duckdb.connect(str(db_path), read_only=True) as con:
        orders = con.execute("SELECT * FROM fact_orders").df()
        metadata = con.execute("SELECT source, built_at FROM metadata").fetchone()
    series = monthly_series(orders)
    if len(series) < 12:
        raise ValueError(f"Need at least 12 complete sales months; found {len(series)}")
    min_train = 12 if len(series) >= 18 else max(6, len(series) - 4)
    folds = backtest(series, min_train)
    if folds.empty:
        raise ValueError("Not enough complete months to backtest a forecast")
    metrics, intervals = {}, {}
    for name in MODELS:
        subset = folds.loc[folds.model.eq(name)]
        metrics[name] = _metrics(subset)
        errors = (subset.actual - subset.prediction).to_numpy()
        coverage_80 = float(np.mean((subset.actual >= subset.prediction + np.quantile(errors, .10)) & (subset.actual <= subset.prediction + np.quantile(errors, .90))))
        metrics[name]["interval80_coverage"] = coverage_80
        intervals[name] = errors
    values = series.item_sales.to_numpy(dtype=float)
    next_position = len(values)
    next_month = (series.month.iloc[-1] + pd.offsets.MonthBegin(1))
    trained = _model()
    supervised_frame = supervised(series)
    trained.fit(supervised_frame[MODEL_FEATURES], supervised_frame.y)
    next_row = pd.DataFrame([_row(values, next_position)])
    forecast = {}
    for name in MODELS:
        point = _predict(name, trained, values, next_position, next_row)
        forecast[name] = {"point": point, **_interval(intervals[name], point)}
    report = {"source": metadata[0], "database_built_at": metadata[1], "horizon_months": HORIZON,
              "complete_months": len(series), "series_start": str(series.month.iloc[0].date()),
              "series_end": str(series.month.iloc[-1].date()), "min_train_months": min_train,
              "next_month": str(next_month.date()), "metrics": metrics, "forecast": forecast,
              "recent_actuals": [{"month": str(row.month.date()), "item_sales": float(row.item_sales), "orders": int(row.orders)}
                                 for row in series.tail(6).itertuples()],
              "limitations": ["Olist has under two years of complete months, so seasonality is estimated from at most one prior year and is weak evidence.",
                              "Intervals are empirical residual quantiles from backtests, not model-based predictive distributions.",
                              "Item sales exclude freight and canceled/unavailable orders, and are not a demand forecast for inventory or staffing.",
                              "Promotions, competitor activity, macro shocks and platform changes are not modeled.",
                              "Partial boundary months are excluded; the final historical month is therefore not the last calendar month in the raw data."]}
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "forecast_report.json").write_text(json.dumps(report, indent=2))
    folds.to_csv(artifact_dir / "forecast_backtest.csv", index=False)
    return report


def load_forecast(artifact_dir: Path) -> dict | None:
    path = artifact_dir / "forecast_report.json"
    return json.loads(path.read_text()) if path.exists() else None
