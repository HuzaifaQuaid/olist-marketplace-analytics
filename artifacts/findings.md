# Marketplace evidence report

Source: **OLIST**. Historical Olist data; not current marketplace performance.

## Five evidence-backed findings

1. Eligible sales total **R$13,494,400.74** across **98,199 orders**; AOV is **R$137.42**, excluding freight.

2. The leading category is **health_beauty**, with **R$1,255,695.13** in item value (9.3% of sales).

3. **6.8%** of **96,470** valid delivered orders arrived after their promised calendar date; unresolved orders are excluded.

4. Late orders average **2.27/5** versus **4.29/5** on time. Difference: **-2.02**, bootstrap 95% CI **[-2.06, -1.98]**. This is association, not causation.

5. **3.0%** of **94,983** observed purchasing customers have multiple orders. This is historical repetition, not a prediction of future repurchasing.

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
