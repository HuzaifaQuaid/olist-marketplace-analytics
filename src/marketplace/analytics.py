"""Approved, read-only analytics and generated evidence reports."""
from pathlib import Path
import json

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


def cohorts(orders: pd.DataFrame) -> pd.DataFrame:
    sales = orders.loc[orders.is_sales_order & orders.item_count.gt(0)].copy()
    sales["month"] = sales.order_purchase_timestamp.dt.to_period("M")
    sales["cohort"] = sales.groupby("customer_unique_id").month.transform("min")
    sales["age"] = sales.month.astype("int64") - sales.cohort.astype("int64")
    counts = sales.groupby(["cohort", "age"]).customer_unique_id.nunique().unstack()
    if counts.empty:
        return counts
    max_age = sales.month.max().ordinal - sales.cohort.min().ordinal
    counts = counts.reindex(columns=range(max_age + 1))
    result = counts.div(counts[0], axis=0)
    last = sales.month.max().ordinal
    for cohort in result.index:
        for age in result.columns:
            if cohort.ordinal + age <= last:
                result.loc[cohort, age] = result.loc[cohort, age] if pd.notna(result.loc[cohort, age]) else 0
    result.index = result.index.astype(str)
    return result


def delivery_association(orders: pd.DataFrame) -> dict:
    valid = orders.dropna(subset=["is_late", "review_score"])
    late = valid.loc[valid.is_late.eq(1), "review_score"]
    ontime = valid.loc[valid.is_late.eq(0), "review_score"]
    if min(len(late), len(ontime)) < 2:
        return {"available": False}
    rng = np.random.default_rng(42)
    # Bootstrap independently within delivery groups; ordinal scores treated as numeric.
    diffs = [rng.choice(late.to_numpy(), len(late)).mean() - rng.choice(ontime.to_numpy(), len(ontime)).mean() for _ in range(500)]
    test = ttest_ind(late, ontime, equal_var=False)
    return {"available": True, "late_n": len(late), "ontime_n": len(ontime),
            "late_mean": float(late.mean()), "ontime_mean": float(ontime.mean()),
            "difference": float(late.mean() - ontime.mean()), "ci95": np.quantile(diffs, [.025, .975]).tolist(),
            "welch_p": float(test.pvalue)}


def _artifact_sections(artifact_dir: Path) -> list[str]:
    """Later-stage evidence, included only when the artifacts exist."""
    sections = []
    forecast_path = artifact_dir / "forecast_report.json"
    if forecast_path.exists():
        report = json.loads(forecast_path.read_text())
        metrics = report["metrics"]
        best = min(metrics, key=lambda name: metrics[name]["rmse"])
        sections.append(
            f"### Forecasting ({report['next_month']})\n\n"
            f"Across {metrics['naive_last_value']['folds']} rolling one-month backtests, the best method by RMSE was **{best}** "
            f"(MAPE {metrics[best]['mape']:.1%}). The calendar/lag model forecast is "
            f"**R${report['forecast']['calendar_lag_model']['point']:,.0f}** (80% interval "
            f"R${report['forecast']['calendar_lag_model']['low_80']:,.0f} to R${report['forecast']['calendar_lag_model']['high_80']:,.0f}). "
            f"The naive last-value forecast is R${report['forecast']['naive_last_value']['point']:,.0f}. "
            "With under two years of complete months, seasonality is weakly identified and the simple baseline is competitive.\n")
    recommender_path = artifact_dir / "recommender_report.json"
    if recommender_path.exists():
        report = json.loads(recommender_path.read_text())
        if report.get("available"):
            metrics = report["metrics"]
            pop = metrics["popularity"]["map_at_k"]
            content = metrics["content_based"]["map_at_k"]
            winner = "popularity" if pop >= content else "content-based"
            sections.append(
                f"### Recommendations\n\n"
                f"On a temporal holdout with **{metrics['content_based']['evaluable_customers']}** evaluable repeat customers, "
                f"content-based recall@10 was **{metrics['content_based']['recall_at_k']:.1%}** versus "
                f"**{metrics['popularity']['recall_at_k']:.1%}** for popularity, with MAP@10 "
                f"**{metrics['content_based']['map_at_k']:.3f}** versus **{metrics['popularity']['map_at_k']:.3f}**. "
                f"The better measured ranking in this holdout was **{winner}**. The evaluation uses only products and popularity known in the training period; "
                "absolute scores remain low because Olist has little repeat purchasing and no browsing signal.\n")
    repurchase_path = artifact_dir / "repurchase_report.json"
    if repurchase_path.exists():
        report = json.loads(repurchase_path.read_text())
        metrics = report["metrics"]["logistic_regression"]
        sections.append(
            f"### Repurchase model\n\n"
            f"Target: a second observed purchase within **{report['horizon_days']} days** of the first, over "
            f"**{report['matured_customers']:,}** matured customers with a base rate of {report['overall_repeat_rate']:.1%}. "
            f"Held-out ROC-AUC **{metrics['roc_auc']:.2f}**, average precision **{metrics['average_precision']:.3f}**, "
            f"and **{metrics['lift_at_top_decile']:.1f}x** lift in the top decile. Discrimination is modest, so scores should rank customers, not promise outcomes.\n")
    return sections


def generate_report(db_path: Path, destination: Path) -> str:
    from marketplace import stats  # imported here to keep module import light
    with duckdb.connect(str(db_path), read_only=True) as con:
        orders = con.execute("SELECT * FROM fact_orders").df()
        source = con.execute("SELECT source FROM metadata").fetchone()[0]
        categories = con.execute("SELECT category, sum(price) AS sales FROM fact_items WHERE order_status NOT IN ('canceled','unavailable') GROUP BY 1 ORDER BY 2 DESC").df()
        quality = con.execute("SELECT * FROM data_quality WHERE count > 0").df()
    sales = orders.loc[orders.is_sales_order & orders.item_count.gt(0)]
    labeled = orders.dropna(subset=["is_late"])
    customer_counts = sales.groupby("customer_unique_id").order_id.nunique()
    assoc = delivery_association(orders)
    adjusted = stats.delivery_review_adjusted(orders)
    top = categories.iloc[0] if not categories.empty else None
    findings = [
        f"Eligible sales total **R${sales.item_value.sum():,.2f}** across **{len(sales):,} orders**; AOV is **R${sales.item_value.mean():,.2f}**, excluding freight.",
        f"The leading category is **{top.category}**, with **R${top.sales:,.2f}** in item value ({top.sales / sales.item_value.sum():.1%} of sales)." if top is not None and sales.item_value.sum() else "No category sales are available.",
        f"**{labeled.is_late.mean():.1%}** of **{len(labeled):,}** valid delivered orders arrived after their promised calendar date; unresolved orders are excluded.",
        f"Late orders average **{assoc['late_mean']:.2f}/5** versus **{assoc['ontime_mean']:.2f}/5** on time. Difference: **{assoc['difference']:.2f}**, bootstrap 95% CI **[{assoc['ci95'][0]:.2f}, {assoc['ci95'][1]:.2f}]**. This is association, not causation." if assoc["available"] else "Insufficient review evidence to compare late and on-time orders.",
        f"**{customer_counts.gt(1).mean():.1%}** of **{len(customer_counts):,}** observed purchasing customers have multiple orders. This is historical repetition, not a prediction of future repurchasing.",
    ]
    text = f"# Marketplace evidence report\n\nSource: **{source.upper()}**. " + ("Synthetic demonstration only; these are not Olist business findings.\n\n" if source == "synthetic" else "Historical Olist data; not current marketplace performance.\n\n")
    text += "## Five evidence-backed findings\n\n" + "\n\n".join(f"{i}. {finding}" for i, finding in enumerate(findings, 1))
    if adjusted.get("available"):
        text += (f"\n\n## Investigation: does the delivery effect survive controls?\n\n"
                 f"After controlling for order value, freight, category, state and month, late delivery is associated with a review-score change of "
                 f"**{adjusted['late_coefficient']:.2f} points** (95% CI {adjusted['ci95_low']:.2f} to {adjusted['ci95_high']:.2f}; n={adjusted['n']:,}), "
                 f"close to the unadjusted gap of {adjusted['unadjusted_difference']:.2f}. "
                 + (f"The odds of a 1-2 star review are **{adjusted.get('logit_odds_ratio', float('nan')):.1f}x** higher for late orders. " if adjusted.get("logit_odds_ratio") else "")
                 + "The association is robust but still observational: delivery timing is not randomly assigned.\n")
    sections = _artifact_sections(destination.parent)
    if sections:
        text += "\n## Later stages\n\n" + "\n".join(sections)
    text += "\n\n## Definitions and limits\n\nItem value excludes freight. Canceled and unavailable orders are excluded from sales; other statuses are included. Late means delivery calendar date is after the promised calendar date. Repeat purchases use customer_unique_id. Partial boundary months should not be used as full-month growth comparisons. No costs, inventory, browsing funnel, or causal effects can be inferred.\n\n"
    text += "## Nonzero quality checks\n\n" + (quality.to_string(index=False) if len(quality) else "No nonzero checks.") + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text)
    return text
