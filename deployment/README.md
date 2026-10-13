# Historical data snapshot

`olist.duckdb` contains only the validated dashboard marts (`fact_orders`,
`fact_items`), quality results, and metadata. Raw source tables, review text,
customer ZIP codes, and exact geolocation coordinates are not bundled here.
Identifiers remain pseudonymous source identifiers, not names or contact details.

Source: [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).
Credit: Olist and the dataset contributors. Dataset license: CC BY-NC-SA 4.0.
This snapshot is provided under the same license for noncommercial educational use.
License: https://creativecommons.org/licenses/by-nc-sa/4.0/

The dashboard shows historical observations, not live marketplace transactions.

The committed `artifacts/` reports (late-delivery model, forecast, recommender
evaluation, repurchase model) were built from this snapshot, so the hosted app
opens with every workspace populated. Raw analysis code is in `src/marketplace/`.
