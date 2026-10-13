"""Adjusted investigation: does the late-delivery/review association survive controls?

Implemented with numpy/scipy/sklearn only so the dashboard has no heavyweight
stats dependency: OLS with HC1 robust standard errors and a near-unpenalised
logistic check, both on the same design matrix.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

MAX_CATEGORIES = 40
MAX_STATES = 27
LOGIT_SAMPLE = 30000
NUMERIC = ["late", "log_value", "log_freight"]


def _prepare(orders: pd.DataFrame) -> pd.DataFrame:
    frame = orders.dropna(subset=["review_score", "is_late"]).copy()
    frame = frame.loc[frame.item_value.gt(0) & frame.order_purchase_timestamp.notna()].copy()
    if frame.empty:
        return frame
    frame["late"] = frame.is_late.astype(float)
    frame["low_review"] = (frame.review_score <= 2).astype(float)
    frame["log_value"] = np.log1p(frame.item_value)
    frame["log_freight"] = np.log1p(frame.freight_value)
    frame["month"] = frame.order_purchase_timestamp.dt.month.astype(int)
    # Collapse rare levels so the design matrix stays stable and interpretable.
    for column, cap in [("category", MAX_CATEGORIES), ("customer_state", MAX_STATES)]:
        keep = frame[column].value_counts().head(cap).index
        frame[column] = np.where(frame[column].isin(keep), frame[column], "other")
    return frame


def _dummy_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(frame[["month", "category", "customer_state"]], drop_first=True, dtype=float)


def _design(frame: pd.DataFrame, reference: list | None = None) -> tuple[np.ndarray, list[str]]:
    """Intercept + scaled-free numerics + dummies, with constant columns dropped for identifiability."""
    dummies = _dummy_columns(frame)
    if reference is not None:
        dummies = dummies.reindex(columns=reference, fill_value=0.0)
    names = ["intercept", *NUMERIC, *dummies.columns]
    matrix = np.column_stack([np.ones(len(frame)), frame[NUMERIC].to_numpy(dtype=float), dummies.to_numpy(dtype=float)])
    keep = [index for index in range(matrix.shape[1]) if index < 1 + len(NUMERIC) or matrix[:, index].std() > 0]
    return matrix[:, keep], [names[index] for index in keep]


def _ols_hc1(matrix: np.ndarray, target: np.ndarray) -> dict:
    beta, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    residual = target - matrix @ beta
    n, k = matrix.shape
    gram_inverse = np.linalg.pinv(matrix.T @ matrix)
    meat = matrix.T @ (matrix * (residual ** 2)[:, None])
    covariance = gram_inverse @ meat @ gram_inverse * (n / (n - k))
    standard_error = np.sqrt(np.diag(covariance))
    t_stat = beta / standard_error
    p_value = 2 * (1 - norm.cdf(np.abs(t_stat)))
    total = float(((target - target.mean()) ** 2).sum())
    r_squared = 1 - float((residual ** 2).sum()) / total if total else 0.0
    return {"beta": beta, "se": standard_error, "p": p_value, "r_squared": r_squared}


def _logit(matrix: np.ndarray, target: np.ndarray) -> dict:
    # Very light L2 keeps near-MLE estimates while guaranteeing identifiability.
    model = LogisticRegression(C=1e6, max_iter=5000, solver="lbfgs")
    model.fit(matrix, target)
    beta = np.concatenate([model.intercept_, model.coef_[0]])
    probability = model.predict_proba(matrix)[:, 1]
    weights = probability * (1 - probability)
    covariance = np.linalg.pinv(matrix.T @ (matrix * weights[:, None]))
    return {"beta": beta, "se": np.sqrt(np.diag(covariance))}


def delivery_review_adjusted(orders: pd.DataFrame) -> dict:
    frame = _prepare(orders)
    if len(frame) < 300 or frame.late.nunique() < 2:
        return {"available": False, "reason": "Not enough valid delivered orders with reviews."}
    matrix, names = _design(frame)
    if np.linalg.matrix_rank(matrix) < matrix.shape[1]:
        return {"available": False, "reason": "The adjusted design is rank deficient; too few observations per level."}
    late_index = names.index("late")
    score = frame.review_score.to_numpy(dtype=float)
    fit = _ols_hc1(matrix, score)
    beta, se, p_value = fit["beta"], fit["se"], fit["p"]
    unadjusted = frame.groupby("late").review_score.mean()
    result = {"available": True, "n": int(len(frame)), "r_squared": round(fit["r_squared"], 4),
              "method": "OLS on numeric review score with HC1 robust standard errors; controls for order value, freight, category, state and month.",
              "late_coefficient": float(beta[late_index]), "ci95_low": float(beta[late_index] - 1.96 * se[late_index]),
              "ci95_high": float(beta[late_index] + 1.96 * se[late_index]), "p_value": float(p_value[late_index]),
              "unadjusted_late_mean": float(unadjusted.get(1.0, np.nan)), "unadjusted_ontime_mean": float(unadjusted.get(0.0, np.nan)),
              "unadjusted_difference": float(unadjusted.get(1.0, np.nan) - unadjusted.get(0.0, np.nan)),
              "late_rate": float(frame.late.mean()),
              "limitations": ["This is an observational association. Late delivery is not randomly assigned, so the coefficient does not prove causation.",
                              "Order value, freight and geography are only partial proxies: seller identity, product fragility and handling are unmeasured.",
                              "Review scores are ordinal, but the model treats them as numeric for interpretability.",
                              "Rare category and state levels are collapsed into 'other' to keep the design stable."]}
    try:
        # The rare-event logistic check is estimated on a capped sample for interactive speed.
        sample = frame if len(frame) <= LOGIT_SAMPLE else frame.sample(LOGIT_SAMPLE, random_state=42)
        sample_matrix, sample_names = _design(sample, reference=list(_dummy_columns(frame).columns))
        # Drop the design's intercept column: sklearn fits its own, and two intercepts are collinear.
        fit = _logit(sample_matrix[:, 1:], sample.low_review.to_numpy(dtype=float))
        index = sample_names.index("late")
        result["logit_odds_ratio"] = float(np.exp(fit["beta"][index]))
        result["logit_odds_ci95"] = [float(np.exp(fit["beta"][index] - 1.96 * fit["se"][index])),
                                     float(np.exp(fit["beta"][index] + 1.96 * fit["se"][index]))]
        result["logit_p_value"] = float(2 * (1 - norm.cdf(abs(fit["beta"][index] / fit["se"][index]))))
        result["logit_n"] = int(len(sample))
        result["logit_note"] = "Odds of a review score of 1-2 for late versus on-time orders, holding the same controls; estimated on a capped random sample for speed."
    except Exception as exc:
        result["logit_note"] = f"Logistic check unavailable: {exc}"
    return result


def _shrink(table: pd.DataFrame, min_orders: int) -> pd.DataFrame:
    """Beta-binomial shrinkage: low-volume estimates move toward the marketplace rate."""
    table = table.loc[table.n.ge(min_orders)].copy()
    if table.empty:
        return table
    table["raw_rate"] = table.late / table.n
    global_rate = table.late.sum() / table.n.sum()
    observed_var = table.raw_rate.var(ddof=1)
    sampling_var = (table.raw_rate * (1 - table.raw_rate) / table.n).mean()
    prior_var = max(observed_var - sampling_var, 1e-6)
    concentration = global_rate * (1 - global_rate) / prior_var - 1
    alpha = max(global_rate * concentration, 1e-3)
    beta = max((1 - global_rate) * concentration, 1e-3)
    table["shrunk_rate"] = (table.late + alpha) / (table.n + alpha + beta)
    table["marketplace_rate"] = global_rate
    table["prior_strength"] = float(alpha + beta)
    return table.sort_values("shrunk_rate", ascending=False)


def shrunk_seller_late_rates(orders: pd.DataFrame, items: pd.DataFrame, min_orders: int = 20) -> pd.DataFrame:
    """Seller-level late rates with shrinkage, built from the order-item grain."""
    if items.empty or orders.empty:
        return pd.DataFrame()
    mapping = items[["seller_id", "order_id"]].drop_duplicates().merge(orders[["order_id", "is_late", "category"]], on="order_id", how="inner")
    mapping = mapping.dropna(subset=["is_late"])
    if mapping.empty:
        return pd.DataFrame()
    table = mapping.groupby("seller_id").agg(n=("is_late", "size"), late=("is_late", "sum"),
                                             dominant_category=("category", lambda values: values.mode().iloc[0] if not values.mode().empty else "unknown")).reset_index()
    return _shrink(table, min_orders)


def freight_by_region(orders: pd.DataFrame, iterations: int = 400) -> pd.DataFrame:
    """Mean item freight per state with a bootstrap interval; distance is a proxy only."""
    frame = orders.loc[orders.item_count.gt(0) & orders.is_sales_order & orders.freight_value.notna()].copy()
    if frame.empty:
        return pd.DataFrame()
    frame["freight_per_item"] = frame.freight_value / frame.item_count.clip(lower=1)
    rng = np.random.default_rng(42)
    rows = []
    for state, group in frame.groupby("customer_state"):
        values = group.freight_per_item.to_numpy()
        if len(values) < 30:
            continue
        draws = [rng.choice(values, len(values)).mean() for _ in range(iterations)]
        rows.append({"customer_state": state, "orders": len(values), "mean_freight_per_item": float(values.mean()),
                     "ci95_low": float(np.quantile(draws, .025)), "ci95_high": float(np.quantile(draws, .975))})
    return pd.DataFrame(rows).sort_values("mean_freight_per_item", ascending=False)
