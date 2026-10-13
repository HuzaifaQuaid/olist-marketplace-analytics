"""Adjusted investigation: does the late-delivery/review association survive controls?"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

MAX_CATEGORIES = 40
MAX_STATES = 27


def _prepare(orders: pd.DataFrame) -> pd.DataFrame:
    frame = orders.dropna(subset=["review_score", "is_late"]).copy()
    frame = frame.loc[frame.item_value.gt(0) & frame.order_purchase_timestamp.notna()].copy()
    if frame.empty:
        return frame
    frame["late"] = frame.is_late.astype(int)
    frame["low_review"] = (frame.review_score <= 2).astype(int)
    frame["log_value"] = np.log1p(frame.item_value)
    frame["log_freight"] = np.log1p(frame.freight_value)
    frame["month"] = frame.order_purchase_timestamp.dt.month.astype(int)
    # Collapse rare levels so the design matrix stays stable and interpretable.
    for column, cap in [("category", MAX_CATEGORIES), ("customer_state", MAX_STATES)]:
        keep = frame[column].value_counts().head(cap).index
        frame[column] = np.where(frame[column].isin(keep), frame[column], "other")
    return frame


def delivery_review_adjusted(orders: pd.DataFrame) -> dict:
    """OLS on the numeric score plus a logistic robustness check, both with HC1 robust errors."""
    frame = _prepare(orders)
    if len(frame) < 300 or frame.late.nunique() < 2:
        return {"available": False, "reason": "Not enough valid delivered orders with reviews."}
    formula = "review_score ~ late + log_value + log_freight + C(category) + C(customer_state) + C(month)"
    try:
        model = smf.ols(formula, data=frame).fit(cov_type="HC1")
    except Exception as exc:  # singular or unbalanced design
        return {"available": False, "reason": f"Adjusted review model could not be estimated: {exc}"}
    confidence = model.conf_int().loc["late"]
    unadjusted = frame.groupby("late").review_score.mean()
    result = {"available": True, "n": int(len(frame)), "r_squared": float(model.rsquared),
              "method": "OLS on numeric review score; HC1 robust standard errors; controls for order value, freight, category, state and month.",
              "late_coefficient": float(model.params["late"]), "ci95_low": float(confidence[0]), "ci95_high": float(confidence[1]),
              "p_value": float(model.pvalues["late"]),
              "unadjusted_late_mean": float(unadjusted.get(1, np.nan)), "unadjusted_ontime_mean": float(unadjusted.get(0, np.nan)),
              "unadjusted_difference": float(unadjusted.get(1, np.nan) - unadjusted.get(0, np.nan)),
              "late_rate": float(frame.late.mean()),
              "limitations": ["This is an observational association. Late delivery is not randomly assigned, so the coefficient does not prove causation.",
                              "Order value, freight and geography are only partial proxies: seller identity, product fragility and handling are unmeasured.",
                              "Review scores are ordinal, but the model treats them as numeric for interpretability.",
                              "Rare category and state levels are collapsed into 'other' to keep the design stable."]}
    try:
        # The rare-event logistic check is estimated on a capped sample for interactive speed.
        logistic_frame = frame if len(frame) <= 30000 else frame.sample(30000, random_state=42)
        logistic = smf.logit("low_review ~ late + log_value + log_freight + C(category) + C(customer_state) + C(month)", data=logistic_frame).fit(disp=False, maxiter=100)
        interval = np.exp(logistic.conf_int().loc["late"])
        result["logit_odds_ratio"] = float(np.exp(logistic.params["late"]))
        result["logit_odds_ci95"] = [float(interval[0]), float(interval[1])]
        result["logit_p_value"] = float(logistic.pvalues["late"])
        result["logit_n"] = int(len(logistic_frame))
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
