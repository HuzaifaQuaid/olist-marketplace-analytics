"""Run with: streamlit run app.py"""
from pathlib import Path
import json
import sys
from html import escape

# Support Streamlit's script runner even when launched without an editable install.
sys.path.insert(0, str(Path(__file__).parent / "src"))

import duckdb
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import binomtest

from marketplace.analytics import cohorts, delivery_association
from marketplace.model import explain_order, features

ROOT = Path(__file__).parent
st.set_page_config(page_title="Olist | Marketplace Intelligence", page_icon="◈", layout="wide")
st.markdown(f"<style>{(ROOT / 'assets/dashboard.css').read_text()}</style>", unsafe_allow_html=True)


@st.cache_data
def load_data(path: str, modified: float):
    with duckdb.connect(path, read_only=True) as con:
        return {name: con.execute(f"SELECT * FROM {name}").df() for name in ["fact_orders", "fact_items", "data_quality", "metadata"]}


def money(value):
    return f"R${value:,.0f}"


def chart(fig):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=16, r=16, t=55, b=20), font=dict(color="#BCC9E0", size=12),
                      title_font=dict(size=17, color="#F0EDFF"),
                      colorway=["#A78BFA", "#67E8F9", "#FCD34D", "#F9A8D4"],
                      hoverlabel=dict(bgcolor="#252A44", font_color="#F4F3FF"),
                      transition=dict(duration=450), legend=dict(orientation="h", y=-.2))
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.07)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.07)")
    for index, trace in enumerate(fig.data):
        color = ["#A78BFA", "#67E8F9", "#FCD34D", "#F9A8D4"][index % 4]
        if trace.type == "scatter":
            trace.update(line=dict(color=color, width=3), marker=dict(color=color, size=7))
            if len(fig.data) == 1:
                trace.update(fill="tozeroy", fillcolor="rgba(167,139,250,0.08)")
        elif trace.type == "bar":
            trace.update(marker_color=color, marker_line_width=0)
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "scrollZoom": False})


st.sidebar.markdown('<div class="brand"><div class="brand-mark">◈</div><div><div class="brand-name">OLIST</div><div class="brand-sub">Intelligence studio</div></div></div>', unsafe_allow_html=True)
with st.sidebar.expander("Data connection", expanded=False):
    default_db = ROOT / "data/olist.duckdb"
    if not default_db.exists():
        default_db = ROOT / "deployment/olist.duckdb"
    db_path = Path(st.text_input("DuckDB file", str(default_db)))
if not db_path.exists():
    st.title("Marketplace intelligence")
    st.info("Create a database to get started. Use the synthetic demo, or ingest the real Olist CSVs.")
    st.code("olist demo\n# Or: olist ingest --raw-dir data/raw\n# Then: olist train && olist report", language="bash")
    st.stop()
try:
    data = load_data(str(db_path), db_path.stat().st_mtime)
except Exception as exc:
    st.error(f"Could not read the analytical database: {exc}")
    st.stop()
orders, items = data["fact_orders"], data["fact_items"]
metadata = data["metadata"].iloc[0]
if metadata.source == "synthetic":
    st.warning("SYNTHETIC DEMO · All figures and model results are generated examples, not real Olist findings.")
source_label = "SYNTHETIC DEMO" if metadata.source == "synthetic" else "HISTORICAL OLIST DATA"
st.markdown(f'''<div class="hero">
<div class="hero-eyebrow"><span class="status-dot"></span>Marketplace intelligence / {source_label}</div>
<h1>A clearer view of your marketplace.</h1>
<p>From sales momentum to the last mile. Explore the patterns behind performance, with evidence at every step.</p>
<div class="hero-tags"><span>◈ {len(orders):,} historical orders</span><span>Brazil · BRL</span><span>Validated DuckDB marts</span><span>Time-aware ML</span></div>
</div>''', unsafe_allow_html=True)

valid_dates = orders.order_purchase_timestamp.dropna()
if valid_dates.empty:
    st.error("No valid purchase timestamps are available.")
    st.stop()
date_range = st.sidebar.date_input("Purchase window", value=(valid_dates.min().date(), valid_dates.max().date()),
                                   min_value=valid_dates.min().date(), max_value=valid_dates.max().date())
states = st.sidebar.multiselect("Customer states", sorted(orders.customer_state.dropna().unique()))
if len(date_range) != 2:
    st.info("Select a start and end date.")
    st.stop()
start, end = date_range
mask = orders.order_purchase_timestamp.ge(pd.Timestamp(start)) & orders.order_purchase_timestamp.lt(pd.Timestamp(end) + pd.Timedelta(days=1))
if states:
    mask &= orders.customer_state.isin(states)
filtered = orders.loc[mask].copy()
filtered_items = items.loc[items.order_id.isin(filtered.order_id)]
sales = filtered.loc[filtered.is_sales_order & filtered.item_count.gt(0)].copy()
sale_items = filtered_items.loc[filtered_items.order_status.isin(sales.order_status.unique()) & filtered_items.order_id.isin(sales.order_id)]
if filtered.empty:
    st.info("No orders match these filters.")
    st.stop()
page = st.sidebar.radio("Workspace", ["Business overview", "Delivery & reviews", "Customers", "Late-delivery model", "Data quality", "Evidence desk"])
st.sidebar.divider()
st.sidebar.caption("Sales = item value excluding freight. Canceled and unavailable orders excluded. Delivery KPIs use valid delivered outcomes only.")
st.sidebar.caption(f"Built {metadata.built_at[:19]} · {metadata.source.upper()}")
page_details = {
    "Business overview": ("01", "The big picture", "Sales momentum, category leaders and seller performance."),
    "Delivery & reviews": ("02", "The customer experience", "Delivery friction, satisfaction and regional differences."),
    "Customers": ("03", "Beyond the first purchase", "Observed repeat behavior and customer acquisition cohorts."),
    "Late-delivery model": ("04", "Signals before the delay", "A transparent, time-aware delivery-risk experiment."),
    "Data quality": ("05", "Evidence you can trust", "Data contracts, validation results and source provenance."),
    "Evidence desk": ("06", "Ask better questions", "Controlled evidence lookups grounded in your actual data."),
}
number, title, subtitle = page_details[page]
st.markdown(f'<div class="section-heading"><span class="section-number">{number}</span><div><h2>{title}</h2><p>{subtitle}</p></div></div>', unsafe_allow_html=True)

if page == "Business overview":
    labeled = filtered.dropna(subset=["is_late"])
    a, b, c, d = st.columns(4)
    a.metric("Item sales", money(sales.item_value.sum()))
    b.metric("Sales orders", f"{len(sales):,}")
    c.metric("Average order value", money(sales.item_value.mean()) if len(sales) else "—")
    d.metric("Late-delivery rate", f"{labeled.is_late.mean():.1%}" if len(labeled) else "—")
    st.caption(f"Delivery denominator: {len(labeled):,} observed, valid delivered orders. Sales count requires at least one item.")
    if not sale_items.empty and sales.item_value.sum() > 0:
        leaders = sale_items.groupby("category").price.sum().sort_values(ascending=False)
        leader = escape(str(leaders.index[0]).replace("_", " ").title())
        st.markdown(f'<div class="insight-strip"><span class="insight-icon">✧</span><div><strong>IN FOCUS · CATEGORY LEADERSHIP</strong><p>{leader} leads this selection with {money(leaders.iloc[0])} in item sales — {leaders.iloc[0] / sales.item_value.sum():.1%} of the selected sales total. Explore the category rankings below.</p></div></div>', unsafe_allow_html=True)
    monthly = sales.assign(month=sales.order_purchase_timestamp.dt.to_period("M").dt.to_timestamp()).groupby("month").agg(
        item_sales=("item_value", "sum"), orders=("order_id", "count"), aov=("item_value", "mean")).reset_index()
    metric = st.radio("Monthly trend", ["item_sales", "orders", "aov"], horizontal=True,
                      format_func=lambda value: {"item_sales": "◈ Item sales", "orders": "▦ Order volume", "aov": "◇ Average order value"}[value])
    chart(px.line(monthly, x="month", y=metric, markers=True, title="Marketplace performance over time"))
    st.caption("Boundary months may be partial. Do not interpret partial-month changes as full-month growth or decline.")
    left, right = st.columns(2)
    categories = sale_items.groupby("category", as_index=False).agg(item_sales=("price", "sum"), orders=("order_id", "nunique")).sort_values("item_sales", ascending=False).head(12)
    sellers = sale_items.groupby("seller_id", as_index=False).agg(item_sales=("price", "sum"), orders=("order_id", "nunique")).sort_values("item_sales", ascending=False).head(12)
    with left:
        chart(px.bar(categories.sort_values("item_sales"), x="item_sales", y="category", orientation="h", title="Leading product categories"))
    with right:
        chart(px.bar(sellers.sort_values("item_sales"), x="item_sales", y="seller_id", orientation="h", title="Leading sellers"))
    st.download_button("Download monthly KPIs", monthly.to_csv(index=False), "monthly_kpis.csv", "text/csv")

elif page == "Delivery & reviews":
    labeled = filtered.dropna(subset=["is_late"])
    a, b, c = st.columns(3)
    a.metric("Median delivery days", f"{labeled.delivery_days.median():.0f}" if len(labeled) else "—")
    b.metric("Late deliveries", f"{labeled.is_late.sum():,.0f} / {len(labeled):,}")
    c.metric("Mean review score", f"{filtered.review_score.mean():.2f} / 5" if filtered.review_score.notna().any() else "—")
    assoc = delivery_association(filtered)
    left, right = st.columns(2)
    with left:
        reviews = labeled.dropna(subset=["review_score"]).assign(delivery=lambda x: x.is_late.map({0: "On time", 1: "Late"}))
        chart(px.histogram(reviews, x="review_score", color="delivery", barmode="group", title="Review distribution by delivery outcome"))
    with right:
        region = sale_items.groupby("customer_state", as_index=False).agg(mean_freight_per_item=("freight_value", "mean"), items=("order_item_id", "count"))
        chart(px.bar(region, x="customer_state", y="mean_freight_per_item", title="Freight charge per item by customer state"))
    if assoc["available"]:
        st.info(f"Late vs. on-time mean review difference: {assoc['difference']:.2f} points; bootstrap 95% CI [{assoc['ci95'][0]:.2f}, {assoc['ci95'][1]:.2f}]. Samples: {assoc['late_n']:,} late / {assoc['ontime_n']:,} on time. Welch p-value: {assoc['welch_p']:.3g}.")
    st.caption("Observational association only. Review scores are ordinal; the mean comparison treats them as numeric. Category, region, seller mix and order value may confound the relationship. Bootstrap interval is approximate.")
    min_orders = st.slider("Minimum observed orders per seller", 5, 100, 20)
    seller_orders = filtered_items[["seller_id", "order_id"]].drop_duplicates().merge(labeled[["order_id", "is_late"]], on="order_id")
    table = seller_orders.groupby("seller_id", as_index=False).agg(observed_orders=("order_id", "count"), late_orders=("is_late", "sum"))
    table = table.loc[table.observed_orders.ge(min_orders)].copy()
    table["late_rate"] = table.late_orders / table.observed_orders
    intervals = [binomtest(int(row.late_orders), int(row.observed_orders)).proportion_ci(method="wilson") for row in table.itertuples()]
    table["ci95_low"] = [interval.low for interval in intervals]
    table["ci95_high"] = [interval.high for interval in intervals]
    st.subheader("Seller delivery watchlist")
    st.dataframe(table.sort_values("late_rate", ascending=False), hide_index=True, width="stretch")
    st.caption("Order-level outcome attributed to each participating seller once. Multi-seller orders do not identify which seller caused a delay. Rates are unadjusted, not proof of seller fault.")

elif page == "Customers":
    counts = sales.groupby("customer_unique_id").order_id.nunique()
    a, b, c = st.columns(3)
    a.metric("Purchasing customers", f"{len(counts):,}")
    b.metric("Repeat customers", f"{counts.gt(1).sum():,}")
    c.metric("Observed repeat rate", f"{counts.gt(1).mean():.1%}" if len(counts) else "—")
    st.caption("Repeat metrics apply to the selected window. Cohorts below use all historical purchases for the selected states, so acquisition dates are not reset by the date filter.")
    cohort_orders = orders.loc[orders.customer_state.isin(states)] if states else orders
    retention = cohorts(cohort_orders)
    if not retention.empty:
        chart(px.imshow(retention, labels=dict(x="Months since first observed purchase", y="Acquisition cohort", color="Purchase rate"),
                        aspect="auto", color_continuous_scale="Teal", zmin=0, zmax=1, title="Customer cohort retention"))
    st.caption("Future/unobserved cohort cells are blank, not zero. Month 0 is 100% by definition. Recent cohorts have less follow-up. Historical repeat rate is not a repurchase-risk model.")

elif page == "Late-delivery model":
    st.subheader("Post-checkout delivery-risk experiment")
    report_path = ROOT / "artifacts/model_report.json"
    model_path = ROOT / "artifacts/late_delivery.joblib"
    if not report_path.exists() or not model_path.exists():
        st.info("Train the models first: olist train")
        st.stop()
    report = json.loads(report_path.read_text())
    if report["source"] != metadata.source or report.get("database_built_at") != metadata.built_at:
        st.error("Model artifacts do not match this database build. Retrain with this database before scoring.")
        st.stop()
    st.caption(f"Chronological split: {report['split_timestamp'][:10]} · Train: {report['train_orders']:,} · Held-out test: {report['test_orders']:,} · Test late rate: {report['test_late_rate']:.1%}")
    st.dataframe(pd.DataFrame(report["metrics"]).T, width="stretch")
    st.caption("Average precision measures rare-class ranking; Brier score measures probability error (lower is better). Fixed 0.5 threshold can have low recall on imbalanced data. No threshold tuning or model selection uses this test set.")
    importance = pd.DataFrame(report["importance"])
    chart(px.bar(importance.head(8).sort_values("average_precision_drop"), x="average_precision_drop", y="feature", orientation="h", title="Global permutation importance · held-out average-precision drop"))
    eligible = filtered.loc[filtered.item_count.gt(0) & filtered.promised_days.ge(0)]
    selected = st.selectbox("Inspect an order (historical feature replay)", eligible.order_id)
    if selected:
        row = eligible.loc[eligible.order_id.eq(selected)]
        # Only load locally generated artifacts; never accept arbitrary uploaded pickle files.
        pipeline = joblib.load(model_path)
        probability = pipeline.predict_proba(features(row))[:, 1][0]
        st.metric("Estimated late probability", f"{probability:.1%}")
        st.dataframe(explain_order(pipeline, row), hide_index=True, width="stretch")
        st.caption("Exact logistic feature contributions in log-odds, relative to the model intercept. Positive terms increase the score; negative terms decrease it. They explain the model, not the causes of delay. Inspection orders may be in training; held-out performance is only the table above.")
    with st.expander("Model limitations", expanded=True):
        for limitation in report["limitations"]:
            st.markdown(f"- {limitation}")

elif page == "Data quality":
    st.subheader("Trust starts with the data")
    st.dataframe(data["data_quality"], hide_index=True, width="stretch")
    st.markdown("""**Grain contracts**
- `fact_orders`: exactly one row per order; item and payment totals aggregated separately before joining.
- `fact_items`: one row per `(order_id, order_item_id)`; monetary category/seller attribution uses item price.
- Reviews: latest valid review per order, deterministically ordered by answer timestamp and review ID.
- Locations: median coordinate per ZIP prefix; never directly joined as a many-to-many table.

**Hard failures:** conflicting primary keys, missing required columns, orphan foreign keys, invalid monetary values, or broken revenue conservation. Date anomalies and missing attributes remain visible in quality checks; invalid delivery timelines are excluded from the late label.
""")
    st.json(metadata.to_dict())

else:
    st.subheader("Evidence desk")
    st.caption("Controlled evidence lookup — not an LLM. Only approved questions are available; no generated SQL or unsupported claims.")
    question = st.selectbox("Business question", ["What counts as revenue?", "Are late deliveries associated with poor reviews?", "Which categories lead sales?", "Can we calculate profit or optimize inventory?", "Can we predict who buys again?", "Where is the findings report?"])
    if question == "What counts as revenue?":
        st.write(f"The selected window has {money(sales.item_value.sum())} in item sales across {len(sales):,} orders. Freight is excluded. Canceled/unavailable orders are excluded. Payment receipts are stored separately and are not interchangeable with item sales.")
        st.caption("Evidence: fact_orders.item_value, is_sales_order, item_count; selected purchase/state filters.")
    elif question == "Are late deliveries associated with poor reviews?":
        st.json(delivery_association(filtered))
        st.write("An association is not a causal claim. Unresolved orders and missing reviews are excluded; seller/category/region differences can confound the comparison.")
    elif question == "Which categories lead sales?":
        table = sale_items.groupby("category", as_index=False).price.sum().sort_values("price", ascending=False).head(10)
        st.dataframe(table.rename(columns={"price": "item_sales_brl"}), hide_index=True)
        st.caption("Evidence: fact_items, summed item price; no duplicated order-level payments.")
    elif question == "Can we calculate profit or optimize inventory?":
        st.write("No. Olist does not provide reliable product costs or inventory levels. Sales and freight are not profit.")
    elif question == "Can we predict who buys again?":
        st.write("Not in this version. The Customers page measures observed repeat purchasing, not future purchase propensity. That requires its own time-aware prediction target and validation.")
    else:
        report_path = ROOT / "artifacts/findings.md"
        if report_path.exists():
            st.markdown(report_path.read_text())
            st.download_button("Download report", report_path.read_text(), "findings.md", "text/markdown")
        else:
            st.info("Generate the report with: olist report")
