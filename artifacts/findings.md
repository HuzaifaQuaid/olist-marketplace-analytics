# Marketplace evidence report

Source: **OLIST**. Historical Olist data; not current marketplace performance.

## Five evidence-backed findings

1. Eligible sales total **R$13,494,400.74** across **98,199 orders**; AOV is **R$137.42**, excluding freight.

2. The leading category is **health_beauty**, with **R$1,255,695.13** in item value (9.3% of sales).

3. **6.8%** of **96,470** valid delivered orders arrived after their promised calendar date; unresolved orders are excluded.

4. Late orders average **2.27/5** versus **4.29/5** on time. Difference: **-2.02**, bootstrap 95% CI **[-2.06, -1.98]**. This is association, not causation.

5. **3.0%** of **94,983** observed purchasing customers have multiple orders. This is historical repetition, not a prediction of future repurchasing.

## Investigation: does the delivery effect survive controls?

After controlling for order value, freight, category, state and month, late delivery is associated with a review-score change of **-1.96 points** (95% CI -1.99 to -1.92; n=95,824), close to the unadjusted gap of -2.02. The odds of a 1-2 star review are **15.9x** higher for late orders. The association is robust but still observational: delivery timing is not randomly assigned.

## Later stages

### Forecasting (2018-09-01)

Across 7 rolling one-month backtests, the best method by RMSE was **naive_last_value** (MAPE 7.0%). The calendar/lag model forecast is **R$952,936** (80% interval R$799,050 to R$1,021,999). The naive last-value forecast is R$848,860. With under two years of complete months, seasonality is weakly identified and the simple baseline is competitive.

### Recommendations

On a temporal holdout with **417** evaluable repeat customers, content-based recall@10 was **2.2%** versus **1.7%** for popularity, with MAP@10 **0.011** versus **0.007**. Absolute values are low because Olist has little repeat purchasing and no browsing or basket signal.

### Repurchase model

Target: a second observed purchase within **90 days** of the first, over **77,560** matured customers with a base rate of 2.3%. Held-out ROC-AUC **0.60**, average precision **0.036**, and **1.8x** lift in the top decile. Discrimination is modest, so scores should rank customers, not promise outcomes.


## Definitions and limits

Item value excludes freight. Canceled and unavailable orders are excluded from sales; other statuses are included. Late means delivery calendar date is after the promised calendar date. Repeat purchases use customer_unique_id. Partial boundary months should not be used as full-month growth comparisons. No costs, inventory, browsing funnel, or causal effects can be inferred.

## Nonzero quality checks

                                                     check  count severity
                     geolocation: exact duplicates removed 261831     info
       orders: delivered status without delivery timestamp      8  warning
                              products: missing categories    610  warning
         reviews: extra rows collapsed to latest per order    551     info
orders: payments differ from item plus freight by > R$0.05   1032  warning
                                          orders: no items    775  warning
                                       orders: no payments      1  warning
