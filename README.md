# Olist Marketplace Intelligence

**Hosted dashboard:** https://huzaifa-olist-intelligence.streamlit.app/

## Cloud deployment

Streamlit Community Cloud runs `app.py` from the `main` branch using Python 3.14
and `requirements.txt`. When a local `data/olist.duckdb` is absent, the app uses
the smaller `deployment/olist.duckdb` snapshot. Model artifacts are version-pinned
to their training runtime. Raw CSVs and credentials are not committed.
See `deployment/README.md` for dataset attribution and noncommercial share-alike
license requirements. Hosting makes the application publicly accessible;
the underlying Olist data remains historical, not real-time.

A runnable first release: validated multi-table ingestion, a business dashboard, customer cohorts, delivery/review investigation, three late-delivery model comparisons, per-order model explanations, and a generated five-finding evidence report.

**Synthetic demo results are not real Olist findings.** The app prominently labels its data source. No Kaggle credentials or LLM API keys are needed to try the demo.

## Quick start

From this project directory, using Python 3.11 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
olist demo
streamlit run app.py
```

Open the local URL printed by Streamlit, normally http://localhost:8501.

## Use the real dataset

Download the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) from Kaggle and extract its CSV files into `data/raw/`. Follow Kaggle's license/attribution requirements. The project does not download files automatically or bundle Olist records.

Expected files:

```text
olist_orders_dataset.csv
olist_order_items_dataset.csv
olist_customers_dataset.csv
olist_products_dataset.csv
olist_sellers_dataset.csv
olist_order_payments_dataset.csv
olist_order_reviews_dataset.csv
olist_geolocation_dataset.csv
product_category_name_translation.csv  # optional; original categories retained if absent
```

```bash
olist ingest --raw-dir data/raw
olist train
olist report
streamlit run app.py
```

Re-run **train and report whenever you re-ingest**; artifacts refer to the previous dataset until rebuilt. Ingestion replaces analytical/raw tables transactionally. Failed validation preserves an existing database. DuckDB requires the app to be stopped while you rebuild the database.

## Dashboard workspaces

| Workspace | What it answers |
| --- | --- |
| Business overview | Monthly sales, order count, AOV, leading categories and sellers |
| Delivery & reviews | Delivery duration, late rates, freight by state, review association, seller rates with Wilson intervals |
| Customers | Observed repeat purchasing and acquisition-cohort retention |
| Late-delivery model | Baseline/logistic/forest comparison, historical risk scoring and exact logistic contributions |
| Data quality | Validation counts, source metadata and table grains |
| Evidence desk | Approved evidence lookups and findings report; deliberately not an LLM |

Date and customer-state filters apply to business metrics. Cohorts use full history for selected states, and model evaluation uses its saved full-dataset test window; these exceptions are labeled in the UI.

## Data contracts

`fact_orders` has one row per order. Items and payments are independently aggregated **before** joining to orders: a two-item/two-payment order cannot quadruple revenue. `fact_items` retains one row per order-item key; category/seller revenue uses item price, not allocated order totals. Reviews are reduced to the latest valid review per order. Geolocation is reduced to median coordinates per ZIP and is not joined directly to orders; location predictors currently use states rather than unreliable ZIP distances.

Conflicting keys, orphan foreign keys, negative/missing money and revenue-conservation errors stop ingestion. Exact duplicate rows are removed and counted. Missing/invalid timestamps are flagged. Late labels exclude invalid delivery timelines. Missing categories are `unknown`; missing numeric model features are imputed using training data only. There is no universal imputation of business facts.

### KPI definitions

- **Item sales:** sum of item price for non-canceled/non-unavailable orders, excluding freight. This is gross merchandise item value, not net accounting revenue or profit.
- **Sales orders / AOV:** eligible orders with at least one item; AOV = item sales / sales orders.
- **Payment value:** separately aggregated receipt amount; not interchangeable with item sales. Reconciliation differences are reported, not forced to match.
- **Late delivery:** delivered calendar date later than estimated calendar date, for delivered-status orders with valid purchase, promise and delivery timestamps. Missing/unresolved outcomes are excluded.
- **Repeat customers:** distinct `customer_unique_id` with more than one eligible order in the selected window.
- **Cohorts:** first observed purchase month; percentage buying in each subsequent month, not cumulative retention. Unobserved future cells remain blank.

Boundary months may be partial. Category/seller order counts are not additive because an order can contain multiple categories/sellers.

## ML design

Prediction point: **immediately after checkout**, assuming promised delivery and basket details are available. Inputs are order value, freight, item/seller counts, product weight, category, seller/customer state, interstate share, purchase calendar features and promised interval. Actual delivery date, carrier handoff, review score and post-purchase payment events are **not inputs**.

An 80/20 purchase-time split is used, not a random split. Training outcomes must already be observed before the test period. Purchases within 45 days of the final purchase timestamp are excluded to reduce right-censoring. Three experiments share preprocessing learned on training data: prevalence baseline, logistic regression and random forest. Logistic regression is the predeclared scoring candidate, not chosen from test results. Average precision, ROC-AUC, Brier score, precision and recall at a fixed 0.5 threshold are saved. No performance guarantee is made.

Global permutation importance measures held-out average-precision changes. Per-order logistic contributions are additive log-odds terms plus the model intercept, **not causal explanations**. Only locally generated trusted `joblib` artifacts should be loaded.

Artifacts:

```text
data/olist.duckdb
artifacts/findings.md
artifacts/model_report.json
artifacts/late_delivery.joblib
artifacts/test_predictions.csv
```

## Tests

```bash
olist demo
python -m unittest discover -s tests -v
```

Tests cover grain and monetary conservation (including intentionally multi-payment demo orders), conflicting keys, orphans, date semantics, leakage feature exclusion, model/report generation, cohorts and every dashboard workspace.

## Staged roadmap — not yet implemented

1. **Deeper investigation:** adjusted category/value/region regressions and stratified sensitivity checks. Current comparisons are unadjusted and do not establish causation.
2. **Forecasting:** naïve and seasonal baselines first, rolling-origin backtests, calendar/lag model, and prediction intervals. Short history and partial months must be handled explicitly.
3. **Recommendations:** popularity baseline and attribute-based ranking; evaluate temporal holdouts and cold-start behavior. Only attempt collaborative filtering if observed overlap supports it. Product IDs have no readable product names in this dataset.
4. **Repurchase prediction:** define a fixed follow-up horizon and censoring-aware labels before training a model.
5. **Grounded AI analyst:** optional LLM API using a whitelist of parameterized query tools and versioned model outputs, with citations, refusal boundaries and evaluation. The evidence desk provides a deterministic starting point without claiming to be an AI assistant.

## Limitations

Olist is historical, geographically specific and observational. It does not reliably provide product costs, inventory or browsing activity. This project cannot calculate profit, optimize stock, infer browsing funnels, or prove why revenue/delivery outcomes changed. Delivered-only ML labels introduce selection bias; a maturity buffer is not a complete censoring solution. Seller watchlists attribute shared order outcomes, not seller responsibility. No personal details or credentials are required or sent to an external service.
