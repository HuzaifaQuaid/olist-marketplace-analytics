"""Run with: streamlit run app.py"""
from pathlib import Path
import json
import sys
from html import escape

# Support Streamlit's script runner even when launched without an installed package.
sys.path.insert(0, str(Path(__file__).parent / "src"))

import duckdb
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import binomtest

from marketplace import forecasting, stats
from marketplace.analyst import Analyst
from marketplace.analytics import cohorts, delivery_association
from marketplace.model import explain_order, features
from marketplace.recommender import Recommender, build_catalog

ROOT = Path(__file__).parent
st.set_page_config(page_title="Olist | Marketplace Intelligence", page_icon="◈", layout="wide")
st.markdown(f"<style>{(ROOT / 'assets/dashboard.css').read_text()}</style>", unsafe_allow_html=True)


@st.cache_resource
def load_data(path: str, modified: float):
    with duckdb.connect(path, read_only=True) as con:
        return {name: con.execute(f"SELECT * FROM {name}").df() for name in ["fact_orders", "fact_items", "data_quality", "metadata"]}


@st.cache_data(show_spinner="Running adjusted regression…")
def adjusted_analysis(path: str, modified: float, start: str, end: str, states: tuple):
    with duckdb.connect(path, read_only=True) as con:
        orders = con.execute("SELECT * FROM fact_orders").df()
        items = con.execute("SELECT * FROM fact_items").df()
    mask = orders.order_purchase_timestamp.ge(pd.Timestamp(start)) & orders.order_purchase_timestamp.lt(pd.Timestamp(end) + pd.Timedelta(days=1))
    if states:
        mask &= orders.customer_state.isin(states)
    filtered = orders.loc[mask]
    return {"adjusted": stats.delivery_review_adjusted(filtered),
            "sellers": stats.shrunk_seller_late_rates(filtered, items.loc[items.order_id.isin(filtered.order_id)], 20),
            "freight": stats.freight_by_region(filtered)}


@st.cache_resource(show_spinner="Loading the analyst…")
def load_analyst(path: str, modified: float, artifacts: str, _data):
    return Analyst(Path(path), Path(artifacts), shared_data=_data)


@st.cache_resource(show_spinner="Building the product catalogue…")
def load_recommender(path: str, modified: float, _items):
    return Recommender(build_catalog(_items))


def money(value):
    return f"R${value:,.0f}"


def present_label(value, column: str | None = None):
    """Convert machine-style labels to readable display text without mutating data."""
    if not isinstance(value, str):
        return value
    if column in {"seller_id", "product_id", "customer_unique_id", "order_id"}:
        prefix = {"seller_id": "Seller", "product_id": "Product", "customer_unique_id": "Customer", "order_id": "Order"}[column]
        # IDs are not descriptive names; show a short, stable reference instead of a UUID.
        return f"{prefix} · {value[:8].upper()}"
    text = value.replace("_", " ").strip()
    if text.isupper() and len(text) <= 3:
        return text
    return text.title()


def display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a presentation-only copy with human-readable headers and labels."""
    shown = frame.copy()
    shown.columns = [present_label(str(column)) for column in shown.columns]
    for column in frame.columns:
        if column in {"category", "dominant_category", "feature", "series", "bucket", "method"}:
            shown[present_label(str(column))] = frame[column].map(lambda value: present_label(value, column))
        elif column in {"seller_id", "product_id", "customer_unique_id", "order_id"}:
            shown[present_label(str(column))] = frame[column].map(lambda value: present_label(value, column))
    return shown


def display_metric_table(frame: pd.DataFrame, index_name: str) -> pd.DataFrame:
    """Make model/evaluation row labels explicit and readable."""
    shown = frame.copy().reset_index(names=index_name)
    shown[index_name] = shown[index_name].map(lambda value: present_label(value))
    return display_frame(shown)


QUALITY_DESCRIPTIONS = {
    "exact duplicates removed": "Exact duplicate rows removed during import",
    "unparseable dates": "Non-empty date values that could not be read",
    "missing purchase timestamps": "Orders missing the purchase date",
    "missing promised dates": "Orders missing the estimated delivery date",
    "delivery before purchase": "Delivery dates earlier than purchase dates",
    "promise before purchase": "Estimated delivery earlier than purchase",
    "delivered status without delivery timestamp": "Marked delivered but missing a delivery date",
    "invalid scores excluded": "Invalid review scores excluded",
    "missing categories": "Products without a category label",
    "extra rows collapsed to latest per order": "Extra reviews reduced to the latest review per order",
    "payments differ from item plus freight by > r$0.05": "Payments differing from items plus freight by more than R$0.05",
    "no items": "Orders without item records",
    "no payments": "Orders without payment records",
}


def present_quality_check(value):
    raw = str(value)
    lower = raw.lower()
    for suffix, description in QUALITY_DESCRIPTIONS.items():
        if lower.endswith(suffix):
            subject = raw[: -len(suffix)].rstrip(": .")
            if "." in subject:
                table, field = subject.split(".", 1)
                subject = f"{present_label(table)} — {present_label(field)}"
            else:
                subject = present_label(subject)
            return f"{subject}: {description[0].lower() + description[1:]}"
    return present_label(raw)


def display_quality(frame: pd.DataFrame) -> pd.DataFrame:
    shown = frame.copy()
    if "check" in shown:
        shown["check"] = shown.check.map(present_quality_check)
    if "severity" in shown:
        shown["severity"] = shown.severity.map(lambda value: {"info": "Informational", "warning": "Review", "error": "Error"}.get(str(value), present_label(str(value))))
    shown.columns = ["Validation check" if str(column) == "check" else present_label(str(column)) for column in shown.columns]
    return shown


def display_metadata(metadata) -> pd.DataFrame:
    fields = [
        ("source", "Dataset", lambda value: "Historical Olist data" if value == "olist" else "Synthetic demo"),
        ("built_at", "Database built", lambda value: pd.to_datetime(value).strftime("%B %d, %Y at %H:%M")),
        ("snapshot_date", "Latest source-data timestamp", lambda value: str(value)),
        ("revenue_definition", "Sales measure", lambda value: str(value)),
    ]
    rows = []
    for key, label, formatter in fields:
        if key in metadata:
            value = formatter(metadata[key])
            if key == "revenue_definition":
                value = "Listed item value, excluding freight; canceled and unavailable orders excluded"
            rows.append({"Detail": label, "Value": value})
    return pd.DataFrame(rows)


def chart(fig):
    fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.35)",
                      margin=dict(l=16, r=16, t=55, b=20), font=dict(color="#624954", size=12),
                      title_font=dict(size=17, color="#422631"),
                      colorway=["#9B2F4A", "#D45465", "#E99A70", "#B8798E"],
                      hoverlabel=dict(bgcolor="#fff9f5", bordercolor="#d8b6bd", font_color="#34212b"),
                      transition=dict(duration=450), legend=dict(orientation="h", y=-.2))
    fig.update_xaxes(gridcolor="rgba(111,65,79,0.10)", zerolinecolor="rgba(111,65,79,0.18)")
    fig.update_yaxes(gridcolor="rgba(111,65,79,0.10)", zerolinecolor="rgba(111,65,79,0.18)")
    # Plotly's dark template can retain pale labels on the light chart cards.
    # Explicit axis, legend, and hover colors keep every chart readable.
    fig.update_xaxes(tickfont=dict(color="#624954"), title_font=dict(color="#624954"), showline=True, linecolor="#ddc6ca")
    fig.update_yaxes(tickfont=dict(color="#624954"), title_font=dict(color="#624954"), showline=True, linecolor="#ddc6ca")
    fig.update_layout(legend=dict(font=dict(color="#49343e")), hoverlabel=dict(font_color="#34212b"))
    for index, trace in enumerate(fig.data):
        color = ["#9B2F4A", "#D45465", "#E99A70", "#B8798E"][index % 4]
        # Plotly category axes otherwise expose source keys such as construction_tools_garden.
        for attr in ("x", "y", "labels"):
            values = getattr(trace, attr, None)
            if values is not None and not isinstance(values, str):
                try:
                    formatted = [present_label(value) for value in values]
                    setattr(trace, attr, formatted)
                except (TypeError, ValueError):
                    pass
        if trace.type == "scatter":
            trace.update(line=dict(color=color, width=3), marker=dict(color=color, size=7))
        elif trace.type in {"bar", "histogram"}:
            trace.update(marker_color=color, marker_line_width=0)
    fig.update_layout(legend_title_font=dict(color="#49343e"))
    st.plotly_chart(fig, theme=None, width="stretch", config={"displaylogo": False, "scrollZoom": False})


def artifact(name: str) -> dict | None:
    path = ARTIFACTS / name
    return json.loads(path.read_text()) if path.exists() else None


def matches(report: dict | None, metadata) -> bool:
    return bool(report) and report.get("source") == metadata.source and report.get("database_built_at") == metadata.built_at


def caveats(items: list[str]):
    with st.expander("Limitations of this result", expanded=False):
        for item in items:
            st.markdown(f"- {item}")


def render_analyst_evidence(evidence: dict):
    """Show compact evidence visually, while retaining full detail on demand."""
    if not evidence:
        st.caption("No supporting data was returned for this answer.")
        return

    scalar_items = [(key, value) for key, value in evidence.items()
                    if isinstance(value, (str, int, float, bool)) and value is not None]
    if scalar_items:
        cols = st.columns(min(len(scalar_items), 4))
        for index, (key, value) in enumerate(scalar_items[:4]):
            label = key.replace("_", " ").title()
            if isinstance(value, float):
                display = f"{value:,.3f}" if abs(value) < 1 else f"{value:,.2f}"
            elif isinstance(value, bool):
                display = "Yes" if value else "No"
            else:
                display = str(value)
            cols[index % len(cols)].metric(label, display)

    for key, value in evidence.items():
        label = key.replace("_", " ").title()
        if isinstance(value, list) and value and all(isinstance(row, dict) for row in value):
            with st.expander(f"📊 {label} · {len(value)} rows", expanded=True):
                st.dataframe(display_frame(pd.DataFrame(value)), hide_index=True, width="stretch")
        elif isinstance(value, dict):
            # Turn metric-like nested dictionaries into readable metric tiles.
            pairs = [(name, item) for name, item in value.items()
                     if isinstance(item, (str, int, float, bool)) and item is not None]
            if pairs:
                with st.expander(f"✦ {label}", expanded=False):
                    metric_cols = st.columns(min(len(pairs), 4))
                    for index, (name, item) in enumerate(pairs):
                        shown = f"{item:,.3f}" if isinstance(item, float) and abs(item) < 1 else (
                            f"{item:,.2f}" if isinstance(item, float) else str(item))
                        metric_cols[index % len(metric_cols)].metric(name.replace("_", " ").title(), shown)
                    remaining = {name: item for name, item in value.items()
                                 if not isinstance(item, (str, int, float, bool))}
                    if remaining:
                        st.json(remaining, expanded=False)
            else:
                with st.expander(f"✦ {label}", expanded=False):
                    st.json(value, expanded=False)
        elif isinstance(value, list) and value:
            with st.expander(f"Details · {label}", expanded=False):
                st.write(value)
        elif isinstance(value, str) and len(value) > 100:
            with st.expander(f"Details · {label}", expanded=False):
                st.write(value)


st.sidebar.markdown('<div class="brand"><div class="brand-mark">◈</div><div><div class="brand-name">OLIST</div><div class="brand-sub">Intelligence studio</div></div></div>', unsafe_allow_html=True)
with st.sidebar.expander("Data connection", expanded=False):
    candidates = [ROOT / "data/olist.duckdb", ROOT / "data/demo.duckdb", ROOT / "deployment/olist.duckdb"]
    default_db = next((path for path in candidates if path.exists()), candidates[0])
    db_path = Path(st.text_input("DuckDB file", str(default_db)))
if not db_path.exists():
    st.title("Marketplace intelligence")
    st.info("Create a database to get started. Use the synthetic demo, or ingest the real Olist CSVs.")
    st.code("olist demo\n# Or: olist ingest --raw-dir data/raw\n# Then: olist build", language="bash")
    st.stop()
try:
    data = load_data(str(db_path), db_path.stat().st_mtime)
except Exception as exc:
    st.error(f"Could not read the analytical database: {exc}")
    st.stop()
orders, items = data["fact_orders"], data["fact_items"]
metadata = data["metadata"].iloc[0]
source_label = "SYNTHETIC DEMO" if metadata.source == "synthetic" else "HISTORICAL OLIST DATA"
st.markdown(f'''<div class="hero">
<div class="hero-eyebrow"><span class="status-dot"></span>Marketplace intelligence / {source_label}</div>
<h1>A clearer view of your marketplace.</h1>
<p>From sales momentum to the next month's forecast. Explore what happened, why it happened, and what comes next — with evidence at every step.</p>
<div class="hero-tags"><span>◈ {len(orders):,} historical orders</span><span>Brazil · BRL</span><span>Validated DuckDB marts</span><span>Forecasting · Recommendations · Grounded analyst</span></div>
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

ARTIFACT_DIRS = [ROOT / "artifacts", ROOT / "artifacts" / "demo"]


def choose_artifacts() -> Path:
    """Pick the artifact directory whose reports match this database build."""
    for directory in ARTIFACT_DIRS:
        candidate = directory / "model_report.json"
        if candidate.exists():
            report = json.loads(candidate.read_text())
            if report.get("source") == metadata.source and report.get("database_built_at") == metadata.built_at:
                return directory
    return ARTIFACT_DIRS[0]


ARTIFACTS = choose_artifacts()

page = st.sidebar.radio("Workspace", ["Business overview", "Delivery & reviews", "Investigation", "Customers", "Repurchase model",
                                      "Forecasting", "Recommendations", "Late-delivery model", "AI analyst", "Data quality"])
st.sidebar.divider()
st.sidebar.caption("Sales = item value excluding freight. Canceled and unavailable orders excluded. Delivery KPIs use valid delivered outcomes only.")
st.sidebar.caption(f"Built {metadata.built_at[:19]} · {metadata.source.upper()}")
st.sidebar.markdown("[Dataset: Olist / Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) · [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)")
page_details = {
    "Business overview": ("01", "The big picture", "Sales momentum, category leaders and seller performance."),
    "Delivery & reviews": ("02", "The customer experience", "Delivery friction, satisfaction and regional differences."),
    "Investigation": ("03", "Does the effect survive controls?", "Adjusted regression, shrunk seller rates and freight by region."),
    "Customers": ("04", "Beyond the first purchase", "Observed repeat behavior and customer acquisition cohorts."),
    "Repurchase model": ("05", "Who buys again?", "A censoring-aware model of repeat purchase within 90 days."),
    "Forecasting": ("06", "What comes next?", "Backtested monthly sales forecasts with honest uncertainty."),
    "Recommendations": ("07", "What should we suggest?", "Popularity and content-based recommendations, evaluated."),
    "Late-delivery model": ("08", "Signals before the delay", "A transparent, time-aware delivery-risk experiment."),
    "AI analyst": ("09", "Ask in plain English", "An LLM that may only choose approved tools, always showing evidence."),
    "Data quality": ("10", "Evidence you can trust", "Data contracts, validation results and source provenance."),
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
        leader = escape(present_label(str(leaders.index[0]), "category"))
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
        categories = categories.copy()
        categories["category"] = categories.category.map(lambda value: present_label(value, "category"))
        chart(px.bar(categories.sort_values("item_sales"), x="item_sales", y="category", orientation="h", title="Leading product categories"))
    with right:
        sellers = sellers.copy()
        sellers["seller_id"] = sellers.seller_id.map(lambda value: present_label(value, "seller_id"))
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
    st.caption("Observational association only. Review scores are ordinal; the mean comparison treats them as numeric. Category, region, seller mix and order value may confound the relationship. Bootstrap interval is approximate. The Investigation workspace tests whether the effect survives controls.")
    min_orders = st.slider("Minimum observed orders per seller", 5, 100, 20)
    seller_orders = filtered_items[["seller_id", "order_id"]].drop_duplicates().merge(labeled[["order_id", "is_late"]], on="order_id")
    table = seller_orders.groupby("seller_id", as_index=False).agg(observed_orders=("order_id", "count"), late_orders=("is_late", "sum"))
    table = table.loc[table.observed_orders.ge(min_orders)].copy()
    table["late_rate"] = table.late_orders / table.observed_orders
    intervals = [binomtest(int(row.late_orders), int(row.observed_orders)).proportion_ci(method="wilson") for row in table.itertuples()]
    table["ci95_low"] = [interval.low for interval in intervals]
    table["ci95_high"] = [interval.high for interval in intervals]
    st.subheader("Seller delivery watchlist (raw rates)")
    st.dataframe(display_frame(table.sort_values("late_rate", ascending=False)), hide_index=True, width="stretch")
    st.caption("Order-level outcome attributed to each participating seller once. Multi-seller orders do not identify which seller caused a delay. Rates are unadjusted, not proof of seller fault. The Investigation workspace adds shrinkage.")

elif page == "Investigation":
    st.markdown("**Question:** is the late-delivery penalty in reviews real, or explained by order value, freight, category and region?")
    result = adjusted_analysis(str(db_path), db_path.stat().st_mtime, str(start), str(end), tuple(states))
    adjusted = result["adjusted"]
    if not adjusted.get("available"):
        st.info(adjusted.get("reason", "Insufficient data for the adjusted model."))
    else:
        left, right, third = st.columns(3)
        left.metric("Adjusted late effect", f"{adjusted['late_coefficient']:.2f} pts", help="OLS coefficient on review score")
        right.metric("Unadjusted gap", f"{adjusted['unadjusted_difference']:.2f} pts")
        third.metric("n", f"{adjusted['n']:,}")
        st.markdown(f"The adjusted effect is **{adjusted['late_coefficient']:.2f} review points** (95% CI {adjusted['ci95_low']:.2f} to {adjusted['ci95_high']:.2f}), "
                    f"versus an unadjusted gap of **{adjusted['unadjusted_difference']:.2f}**. "
                    + (f"Late orders are **{adjusted.get('logit_odds_ratio', float('nan')):.1f}x** more likely to receive a 1-2 star review (95% CI "
                       f"{adjusted.get('logit_odds_ci95', [float('nan'), float('nan')])[0]:.1f} to {adjusted.get('logit_odds_ci95', [float('nan'), float('nan')])[1]:.1f}). " if adjusted.get("logit_odds_ratio") else "")
                    + f"Model R²: {adjusted['r_squared']:.2f}.")
        st.info("Because the adjusted gap barely shrinks, the association is not explained away by value, freight, category or region. That is still not proof of causation: delivery timing is not randomly assigned and sellers or handling are unmeasured.")
        if adjusted.get("logit_note"):
            st.caption(adjusted["logit_note"] + (f" Sample size: {adjusted['logit_n']:,}." if adjusted.get("logit_n") else ""))
        caveats(adjusted["limitations"])
    st.divider()
    st.markdown("**Shrunk seller late-delivery rates:** small sellers are pulled toward the marketplace rate so a handful of late orders does not create a false outlier.")
    sellers = result["sellers"]
    if sellers.empty:
        st.info("Not enough delivered orders per seller in this selection.")
    else:
        display = sellers.head(15)[["seller_id", "dominant_category", "n", "late", "raw_rate", "shrunk_rate", "marketplace_rate"]]
        chart(px.bar(display.assign(seller_id=display.seller_id.map(lambda value: present_label(value, "seller_id"))).sort_values("shrunk_rate"), x="shrunk_rate", y="seller_id", orientation="h",
                     title="Highest shrunk late-delivery rates", hover_data=["n", "raw_rate"]))
        st.dataframe(display_frame(display.round(4)), hide_index=True, width="stretch")
        st.caption(f"Marketplace rate {sellers.marketplace_rate.iloc[0]:.2%}; prior strength ≈ {sellers.prior_strength.iloc[0]:.0f} pseudo-orders. Lower-volume sellers move further toward the marketplace rate.")
    st.divider()
    st.markdown("**Freight by customer state:** bootstrap 95% intervals. Distance is only a proxy — exact coordinates are not joined here.")
    freight = result["freight"]
    if freight.empty:
        st.info("Not enough orders per state in this selection.")
    else:
        fig = go.Figure(go.Bar(x=freight.customer_state, y=freight.mean_freight_per_item,
                               error_y=dict(type="data", array=(freight.ci95_high - freight.mean_freight_per_item),
                                             arrayminus=(freight.mean_freight_per_item - freight.ci95_low), color="#624954")))
        chart(fig.update_layout(title="Mean item freight by customer state", xaxis_title="State", yaxis_title="BRL per item"))
        st.dataframe(display_frame(freight.round(2)), hide_index=True, width="stretch")

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
                        aspect="auto", color_continuous_scale=["#FFF4EE", "#E8A397", "#A93250", "#76243F"], zmin=0, zmax=1, title="Customer cohort retention"))
    st.caption("Future/unobserved cohort cells are blank, not zero. Month 0 is 100% by definition. Recent cohorts have less follow-up. See the Repurchase model workspace for a time-aware prediction of buying again.")

elif page == "Repurchase model":
    report = artifact("repurchase_report.json")
    if not matches(report, metadata):
        st.info("Build this model first: olist repurchase")
        st.stop()
    metrics = report["metrics"]["logistic_regression"]
    a, b, c, d = st.columns(4)
    a.metric("Base repeat rate", f"{report['overall_repeat_rate']:.1%}")
    b.metric("Held-out ROC-AUC", f"{metrics['roc_auc']:.2f}")
    c.metric("Average precision", f"{metrics['average_precision']:.3f}")
    d.metric("Top-decile lift", f"{metrics['lift_at_top_decile']:.1f}x")
    st.markdown(f"**Question:** which customers are likely to buy again? The label is a **second observed purchase within {report['horizon_days']} days** of the first, "
                f"over **{report['matured_customers']:,}** customers who had the full observation horizon (so recent cohorts are excluded rather than mislabelled).")
    st.dataframe(display_metric_table(pd.DataFrame(report["metrics"]).T, "model"), hide_index=True, width="stretch")
    st.caption("Base rate is the naive prevalence; average precision measures rare-event ranking. The top decile repeats at "
               f"{metrics['top_decile_rate']:.1%} versus the {report['test_repeat_rate']:.1%} test base rate.")
    calibration = pd.DataFrame(report["calibration"])
    chart(px.bar(calibration.melt(id_vars="bucket", value_vars=["predicted", "actual"], var_name="series", value_name="rate"),
                 x="bucket", y="rate", color="series", barmode="group", title="Calibration on the held-out window"))
    st.caption("Predicted and observed repeat rates per score bucket. The model is reasonably calibrated, so scores can rank customers — but the event is rare and discrimination is modest, so treat scores as priorities, not promises.")
    caveats(report["limitations"])

elif page == "Forecasting":
    report = artifact("forecast_report.json")
    if not matches(report, metadata):
        st.info("Build the forecast first: olist forecast")
        st.stop()
    metrics = report["metrics"]
    best = min(metrics, key=lambda name: metrics[name]["rmse"])
    chosen = report.get("selected_method", best)
    a, b, c = st.columns(3)
    a.metric(f"Forecast · {report['next_month']}", money(report["forecast"][chosen]["point"]))
    interval = report["forecast"][chosen]
    b.metric("80% interval", f"BRL {interval['low_80']:,.0f} to {interval['high_80']:,.0f}")
    c.metric("Selected by rolling RMSE", chosen.replace("_", " "))
    history = pd.DataFrame(report["recent_actuals"])
    forecast = report["forecast"][chosen]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history.month, y=history.item_sales, mode="lines+markers", name="Actual", fill="tozeroy", fillcolor="rgba(169,50,80,0.10)"))
    fig.add_trace(go.Scatter(x=[report["next_month"]], y=[forecast["point"]], mode="markers", name="Forecast", marker=dict(size=14, symbol="diamond")))
    fig.add_trace(go.Scatter(x=[report["next_month"], report["next_month"]], y=[forecast["low_95"], forecast["high_95"]], mode="lines", name="95% interval", line=dict(width=10, color="rgba(103,232,249,0.25)")))
    chart(fig.update_layout(title="Recent actuals and next-month forecast", yaxis_title="Item sales (BRL)"))
    st.markdown(f"**Selected method:** {chosen.replace('_', ' ')} by lowest rolling-origin RMSE. With only **{report['complete_months']} complete months**, forecast errors are still substantial: naive MAPE {metrics['naive_last_value']['mape']:.1%}; calendar/lag MAPE {metrics['calendar_lag_model']['mape']:.1%}. Long-term seasonality cannot be trusted from this short history.")
    st.dataframe(display_metric_table(pd.DataFrame(metrics).T, "method"), hide_index=True, width="stretch")
    st.caption("Rolling one-month-ahead backtests across every available origin (no random splits). Naive carries the last value forward; seasonal naive repeats the value from twelve months earlier.")
    st.download_button("Download backtest results", (ARTIFACTS / "forecast_backtest.csv").read_text(), "forecast_backtest.csv", "text/csv")
    caveats(report["limitations"])

elif page == "Recommendations":
    catalogue = load_recommender(str(db_path), db_path.stat().st_mtime, items)
    report = artifact("recommender_report.json")
    left, right = st.columns(2)
    with left:
        st.markdown("**Recommend similar products to a product**")
        options = catalogue.catalog.sort_values("orders", ascending=False).head(400).product_id.tolist()
        product = st.selectbox("Product", options)
        if product:
            similar = catalogue.similar_to_product(product, 8)
            st.dataframe(display_frame(similar[["product_id", "category", "mean_price", "orders", "score"]].round(3)), hide_index=True, width="stretch")
    with right:
        st.markdown("**Recommend for a customer**")
        with duckdb.connect(str(db_path), read_only=True) as con:
            repeaters = con.execute("""
                SELECT customer_unique_id, list(product_id) AS products
                FROM (SELECT DISTINCT customer_unique_id, product_id FROM fact_items WHERE customer_unique_id IS NOT NULL)
                GROUP BY 1 HAVING count(*) >= 2
                USING SAMPLE 300 ROWS""").df()
        if repeaters.empty:
            st.info("No customers with repeat purchases are available.")
        else:
            customer = st.selectbox("Customer with purchase history", repeaters.customer_unique_id.tolist())
            row = repeaters.loc[repeaters.customer_unique_id.eq(customer)].iloc[0]
            recommendations = catalogue.for_customer(row.products, 8)
            st.dataframe(display_frame(recommendations[["product_id", "category", "mean_price", "orders"]].round(2)), hide_index=True, width="stretch")
            st.caption(f"This customer already bought {len(row.products)} distinct product(s); those are excluded from the suggestions.")
    st.divider()
    if matches(report, metadata) and report.get("available"):
        metrics = report["metrics"]
        a, b, c = st.columns(3)
        a.metric("Content MAP@10", f"{metrics['content_based']['map_at_k']:.3f}")
        b.metric("Popularity MAP@10", f"{metrics['popularity']['map_at_k']:.3f}")
        c.metric("Evaluable customers", f"{report['customers_with_repeat_history']:,}")
        st.dataframe(display_metric_table(pd.DataFrame(metrics).T, "method"), hide_index=True, width="stretch")
        st.caption(f"Strict temporal warm-start holdout at {report['cutoff'][:10]}: catalog/popularity fitted on earlier purchases only; recommend unseen products known before the test; measure Recall@10 as hits divided by relevant later items. Evaluable customers: {report['customers_with_repeat_history']:,}. Training catalog: {report['train_catalog_size']:,}; new test-window products excluded from warm-start metrics: {report['test_products_excluded_as_cold_start']:,}.")
        caveats(report["limitations"])
    else:
        st.info("Build the evaluation metrics first: olist recommend")

elif page == "Late-delivery model":
    st.subheader("Post-checkout delivery-risk experiment")
    report = artifact("model_report.json")
    model_path = ARTIFACTS / "late_delivery.joblib"
    if not matches(report, metadata) or not model_path.exists():
        st.info("Train the models first: olist train")
        st.stop()
    st.caption(f"Chronological split: {report['split_timestamp'][:10]} · Train: {report['train_orders']:,} · Held-out test: {report['test_orders']:,} · Test late rate: {report['test_late_rate']:.1%}")
    st.dataframe(display_metric_table(pd.DataFrame(report["metrics"]).T, "model"), hide_index=True, width="stretch")
    logit_metrics = report["metrics"]["logistic_regression"]
    st.caption(f"Average precision measures rare-class ranking; Brier score measures probability error (lower is better). The logistic operating threshold ({report['threshold']:.3f}) was selected on an earlier chronological validation window to favor recall (F2), then evaluated once on this held-out test window. At this threshold test recall is {logit_metrics['recall_at_threshold']:.1%}, but precision is {logit_metrics['precision_at_threshold']:.1%}: expect false alarms. Thresholds trade false alarms against missed late orders.")
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
    caveats(report["limitations"])

elif page == "AI analyst":
    st.markdown("**Ask a question in plain English.** Get a clear takeaway, the numbers behind it, and the boundaries of what the data can tell us.")
    try:
        analyst = load_analyst(str(db_path), db_path.stat().st_mtime, str(ARTIFACTS), data)
    except Exception as exc:
        st.error(f"The analyst could not start: {exc}")
        st.stop()
    if "analyst_turns" not in st.session_state:
        st.session_state.analyst_turns = []
    question = st.text_input("Your question", placeholder="Which categories had the biggest sales decline?", key="analyst_question_input")
    suggestions = analyst.suggestions()
    st.caption("Suggested questions")
    columns = st.columns(3)
    for index, suggestion in enumerate(suggestions):
        if columns[index % 3].button(suggestion, key=f"suggest-{index}"):
            question = suggestion
    if question and st.session_state.get("analyst_submitted_question") != question:
        with st.spinner("Choosing an approved analysis…"):
            answer = analyst.ask(question)
        st.session_state.analyst_turns.append(answer)
        st.session_state.analyst_submitted_question = question
    for turn in reversed(st.session_state.analyst_turns):
        with st.container(border=True):
            st.markdown(f"<div class='analyst-question'><span>YOU ASKED</span><br><strong>{escape(turn['question'])}</strong></div>", unsafe_allow_html=True)
            status = "Grounded in project data" if turn.get("tool") else "Outside the approved analyses"
            st.markdown(f"<div class='analyst-answer'><div class='analyst-answer-top'><span class='analyst-sparkle'>✦</span><span class='analyst-label'>THE TAKEAWAY</span><span class='analyst-status'>{escape(status)}</span></div><div class='analyst-answer-copy'>{escape(turn['answer'])}</div></div>", unsafe_allow_html=True)
            if turn.get("evaluation_note"):
                st.info(turn["evaluation_note"], icon="📌")
            evidence = turn.get("evidence", {})
            if evidence:
                st.markdown("#### The evidence")
                render_analyst_evidence(evidence)
            if turn.get("caveats"):
                with st.expander("⚠️ How to interpret this", expanded=False):
                    for item in turn["caveats"]:
                        st.markdown(f"- {item}")
            with st.expander("About this answer", expanded=False):
                st.caption(f"Approved analysis: `{turn.get('tool') or 'none'}` · Answer planner: `{turn.get('planner', 'unknown')}`. The analyst selects a fixed project tool; it does not generate SQL.")

else:
    st.subheader("Data checks and source details")
    quality = data["data_quality"].copy()
    quality = quality.loc[quality["count"].gt(0)]
    informational = quality.severity.eq("info") if "severity" in quality else pd.Series(False, index=quality.index)
    st.markdown("**Items to be aware of**")
    st.caption("These checks show records that were cleaned, excluded, or need context. A non-zero count is not automatically an error.")
    if quality.empty:
        st.success("No non-zero validation counts were recorded for this build.")
    else:
        review = quality.loc[~informational]
        info = quality.loc[informational]
        if not review.empty:
            st.dataframe(display_quality(review), hide_index=True, width="stretch")
        if not info.empty:
            with st.expander(f"Informational clean-up steps · {len(info)} checks", expanded=False):
                st.dataframe(display_quality(info), hide_index=True, width="stretch")
    st.markdown("**How the tables are structured**")
    st.markdown("""
- **Orders:** one row per order. Item totals and payment totals are calculated separately before combining, so sales are not accidentally multiplied.
- **Order items:** one row per product line in an order. Category and seller sales use the item's listed price.
- **Reviews:** where an order has multiple valid reviews, the latest one is selected using its recorded response time.
- **Locations:** multiple coordinate records for a postal-code prefix are summarized before use.

**Checks that stop the build:** conflicting IDs, missing required columns, orphan records, invalid monetary values, or sales totals changing unexpectedly during a join. Other issues—such as missing category labels—are reported and handled where possible.
""")
    st.markdown("**Dataset and refresh details**")
    st.dataframe(display_metadata(metadata.to_dict()), hide_index=True, width="stretch")
