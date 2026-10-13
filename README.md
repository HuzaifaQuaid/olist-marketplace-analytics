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

A complete analytics stack on real, messy multi-table data: validated ingestion and grain-safe marts, a business dashboard, customer cohorts, an adjusted delivery/review investigation, a time-aware late-delivery model with explanations, backtested sales forecasting, evaluated product recommendations, a censoring-aware repurchase model, and a grounded analyst that may only use approved tools.

**Synthetic demo results are not real Olist findings.** The app prominently labels its data source. No Kaggle credentials or LLM API keys are needed to try the demo.

## Quick start

From this project directory, using Python 3.11 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
olist demo          # builds every stage on synthetic data into data/demo.duckdb
streamlit run app.py
```

Open the local URL printed by Streamlit, normally http://localhost:8501. The synthetic demo writes to its own `data/demo.duckdb` and `artifacts/demo/`, so it never overwrites real ingested data or the real reports.

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
olist build
streamlit run app.py
```

`olist build` re-runs the late-delivery model, the forecast, the recommender evaluation, the repurchase model and the report against the current database. Re-run it whenever you re-ingest; artifacts refer to the previous dataset until rebuilt. Ingestion replaces analytical/raw tables transactionally. Failed validation preserves an existing database. DuckDB requires the app to be stopped while you rebuild the database.

## Dashboard workspaces

| Workspace | What it answers |
| --- | --- |
| Business overview | Monthly sales, order count, AOV, leading categories and sellers |
| Delivery & reviews | Delivery duration, late rates, freight by state, review association, seller rates with Wilson intervals |
| Investigation | Adjusted regression (does the delivery effect survive controls?), shrunk seller rates, freight by state with bootstrap intervals |
| Customers | Observed repeat purchasing and acquisition-cohort retention |
| Repurchase model | Censoring-aware prediction of a repeat purchase within 90 days, with calibration |
| Forecasting | Naive, seasonal and calendar/lag forecasts, rolling-origin backtests and uncertainty intervals |
| Recommendations | Popularity and content-based recommendations with a temporal-holdout evaluation |
| Late-delivery model | Baseline/logistic/forest comparison, historical risk scoring and exact logistic contributions |
| AI analyst | An LLM (when a key is set) that may only choose approved tools, and always shows its evidence |
| Data quality | Validation counts, source metadata and table grains |

Date and customer-state filters apply to business metrics and the Investigation workspace. Cohorts use full history for selected states, and model evaluation uses its saved full-dataset test window; these exceptions are labeled in the UI.

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

An 80/20 purchase-time split is used, not a random split. Training outcomes must already be observed before the test period. Purchases within 45 days of the final purchase timestamp are excluded to reduce right-censoring. An earlier chronological validation slice selects an F2 threshold (favoring recall); the held-out test remains untouched until final evaluation. The selected high-recall operating point catches more late orders but creates many false alarms, so this is a review queue rather than an automatic intervention. Three experiments share preprocessing learned on the fit window: prevalence baseline, logistic regression and random forest. Logistic regression is the predeclared scoring candidate, not chosen from test results. Average precision, ROC-AUC, Brier score, precision and recall at validation-selected thresholds are saved. No performance guarantee is made.

Global permutation importance measures held-out average-precision changes. Per-order logistic contributions are additive log-odds terms plus the model intercept, **not causal explanations**. Only locally generated trusted `joblib` artifacts should be loaded.

## Investigation design

`stats.delivery_review_adjusted` fits an OLS model of review score on a late indicator with controls for log order value, log freight, category, customer state and month, using HC1 robust standard errors. Rare category/state levels are collapsed. A logistic model on a 1-2 star outcome is reported as a robustness check (estimated on a capped sample for interactive speed). Because late delivery is not randomly assigned, the coefficient is an adjusted association, not a causal effect. Seller late rates use beta-binomial shrinkage so low-volume sellers move toward the marketplace rate.

## Forecasting design

Complete calendar months only; leading and trailing partial months are dropped. Forecasts are evaluated with rolling-origin, one-month-ahead backtests (never random splits). Three methods share the target: a naive last-value baseline, a seasonal-naive baseline, and a calendar/lag Ridge model. Intervals are empirical residual quantiles pooled from the backtests. With under two years of complete months, seasonality is weakly identified and the naive baseline is often competitive; the UI states this explicitly rather than hiding it.

## Recommendation design

A popularity baseline and a content-based model (category plus scaled physical attributes, cosine similarity computed on demand so no product-by-product matrix is stored). Evaluation is a temporal holdout: fit catalog and popularity only on earlier purchases; recommend only products known before the holdout; exclude items the customer already bought; then compare later purchases using standard Recall@K (hits / relevant items), Precision@K and MAP@K. New products are counted as cold-start and excluded from warm-start ranking metrics; unknown customers fall back to popularity. Olist's sparse repeat purchasing means few customers are evaluable and collaborative filtering is not attempted.

## Repurchase design

Target: a second observed purchase within 90 days of the first. Only customers observed for the full horizon are labelled, so recent cohorts are excluded rather than mislabelled. Features come from the first order and its calendar context. A prevalence baseline and an (unweighted, therefore calibrated) logistic model are compared on a temporal split with ROC-AUC, average precision, Brier score and top-decile lift, plus a calibration curve.

## Grounded analyst design

`Analyst` exposes a whitelist of parameterized tools that read the validated marts and the versioned model/forecast/report artifacts. Routing is deterministic by default. If `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` is set, an LLM may choose a tool and parameters as JSON; anything outside the whitelist is rejected, and the model never writes SQL or free-form answers. Every answer returns structured evidence and caveats, and unsupported topics (profit, inventory, browsing, marketing spend, demographics) are refused with a reason. Set `OPENAI_MODEL`/`ANTHROPIC_MODEL` to override the default model.

Artifacts:

```text
data/olist.duckdb
artifacts/findings.md
artifacts/model_report.json
artifacts/late_delivery.joblib
artifacts/test_predictions.csv
artifacts/forecast_report.json
artifacts/forecast_backtest.csv
artifacts/recommender_report.json
artifacts/repurchase_report.json
artifacts/repurchase.joblib
artifacts/repurchase_predictions.csv
```

## Tests

```bash
olist demo
python -m unittest discover -s tests -v
```

Tests cover grain and monetary conservation (including intentionally multi-payment demo orders), conflicting keys, orphans, date semantics, leakage feature exclusion, model/report generation, cohorts, forecast backtesting and uncertainty, recommender exclusion/cold-start, repurchase horizon and censoring, investigation shrinkage and bounds, and grounded-analyst routing/refusals, plus every dashboard workspace.

## Current results (real Olist data)

- **Sales foundation:** R$13.49M item sales across 98,199 orders; AOV R$137.42; 6.8% of valid delivered orders late.
- **Investigation:** the adjusted late-delivery review penalty (−1.96 points) barely differs from the unadjusted gap (−2.02), so value, freight, category and region do not explain it away. Still observational.
- **Forecasting:** with 19 complete months, the naive baseline (MAPE 7.0%) beats the calendar/lag model (13.1%). A short history cannot support confident seasonality.
- **Recommendations:** strict temporal warm-start evaluation on 228 repeat customers; popularity currently outperforms content-based ranking (MAP@10 0.014 vs 0.007). Both absolute scores are low on sparse data.
- **Late-delivery model:** the untouched temporal test gives random forest ROC-AUC 0.744 / average precision 0.101, versus 0.035 prevalence AP. A validation-chosen logistic high-recall cutoff reaches 79% recall but only 5.9% precision: use as a manual review queue, not an automatic decision.
- **Repurchase:** 90-day repeat base rate 2.3%; logistic ROC-AUC 0.60 with 1.8x top-decile lift and reasonable calibration — usable for ranking, not for promising outcomes.

## Future work

1. **Threshold operations:** the chosen logistic high-recall cutoff finds ~79% of late deliveries but only ~5.9% precision on the held-out test. A human triage policy or cheaper second-stage check is needed before rollout; false-positive costs matter. Random forest ranks better, but the threshold must be reselected on validation before deployment.
2. **Richer forecasting** when more history exists: promotion/holiday regressors and model-based predictive intervals. The selected naive method now matches the minimum rolling-origin RMSE, rather than always presenting the calendar model.
3. **Recommendation improvement:** current temporal warm-start evaluation shows popularity wins; richer catalog text, basket data, inventory and online experiments would be needed to justify personalised recommendations.
4. **Causal designs** (matching or difference-in-differences around seller or logistics changes) to move beyond adjusted association.
5. **Analyst evaluation harness** with a labelled question set, so grounding and refusal behaviour are measured rather than assumed.

## Limitations

Olist is historical, geographically specific and observational. It does not reliably provide product costs, inventory or browsing activity. This project cannot calculate profit, optimize stock, infer browsing funnels, or prove why revenue/delivery outcomes changed. Delivered-only ML labels introduce selection bias; a maturity buffer is not a complete censoring solution. Seller watchlists attribute shared order outcomes, not seller responsibility. Forecasts cover item sales only and are not demand forecasts. Recommendations and repurchase predictions rank, they do not promise. No personal details or credentials are required, and without an API key the analyst runs entirely deterministically.
