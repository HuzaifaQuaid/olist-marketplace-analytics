"""Approved, read-only analytics and generated evidence reports."""
from pathlib import Path
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


def generate_report(db_path: Path, destination: Path) -> str:
    with duckdb.connect(str(db_path), read_only=True) as con:
        orders = con.execute("SELECT * FROM fact_orders").df()
        source = con.execute("SELECT source FROM metadata").fetchone()[0]
        categories = con.execute("SELECT category, sum(price) AS sales FROM fact_items WHERE order_status NOT IN ('canceled','unavailable') GROUP BY 1 ORDER BY 2 DESC").df()
        quality = con.execute("SELECT * FROM data_quality WHERE count > 0").df()
    sales = orders.loc[orders.is_sales_order & orders.item_count.gt(0)]
    labeled = orders.dropna(subset=["is_late"])
    customer_counts = sales.groupby("customer_unique_id").order_id.nunique()
    assoc = delivery_association(orders)
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
    text += "\n\n## Definitions and limits\n\nItem value excludes freight. Canceled and unavailable orders are excluded from sales; other statuses are included. Late means delivery calendar date is after the promised calendar date. Repeat purchases use customer_unique_id. Partial boundary months should not be used as full-month growth comparisons. No costs, inventory, browsing funnel, or causal effects can be inferred.\n\n"
    text += "## Nonzero quality checks\n\n" + (quality.to_string(index=False) if len(quality) else "No nonzero checks.") + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text)
    return text
