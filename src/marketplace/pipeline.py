"""Validated ingestion and grain-safe analytical marts."""
from pathlib import Path
import json

import duckdb
import pandas as pd

TABLES = {
    "orders": (["order_id"], ["order_id", "customer_id", "order_status", "order_purchase_timestamp", "order_estimated_delivery_date", "order_delivered_customer_date"]),
    "customers": (["customer_id"], ["customer_id", "customer_unique_id", "customer_state", "customer_zip_code_prefix"]),
    "order_items": (["order_id", "order_item_id"], ["order_id", "order_item_id", "product_id", "seller_id", "price", "freight_value"]),
    "order_payments": (["order_id", "payment_sequential"], ["order_id", "payment_sequential", "payment_value"]),
    "order_reviews": (None, ["order_id", "review_id", "review_score", "review_answer_timestamp"]),
    "products": (["product_id"], ["product_id", "product_category_name", "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]),
    "sellers": (["seller_id"], ["seller_id", "seller_state", "seller_zip_code_prefix"]),
    "geolocation": (None, ["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng"]),
}


def build_database(raw_dir: Path, db_path: Path, source: str = "olist") -> dict:
    if source not in {"olist", "synthetic"}:
        raise ValueError("Source must be olist or synthetic")
    frames, checks = {}, []

    def check(name, count, severity="warning"):
        checks.append({"check": name, "count": int(count), "severity": severity})

    for name, (keys, required) in TABLES.items():
        path = raw_dir / f"olist_{name}_dataset.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Download and unzip the Olist dataset first.")
        frame = pd.read_csv(path)
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f"{name}: missing columns {sorted(missing)}")
        check(f"{name}: exact duplicates removed", frame.duplicated().sum(), "info")
        frame = frame.drop_duplicates().copy()
        if keys:
            if frame[keys].isna().any(axis=None) or frame.duplicated(keys).any():
                raise ValueError(f"{name}: null or conflicting primary keys {keys}; fix the source instead of silently dropping records")
        for col in frame.columns:
            if col.endswith("_timestamp") or col.endswith("_date") or col == "order_approved_at":
                converted = pd.to_datetime(frame[col], errors="coerce")
                check(f"{name}.{col}: unparseable dates", (frame[col].notna() & converted.isna()).sum())
                frame[col] = converted
        numeric = [c for c in frame if c in {"price", "freight_value", "payment_value", "review_score", "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm", "geolocation_lat", "geolocation_lng"}]
        for col in numeric:
            converted = pd.to_numeric(frame[col], errors="coerce")
            if name in {"order_items", "order_payments"} and (converted.isna().any() or converted.lt(0).any()):
                raise ValueError(f"{name}.{col}: missing, invalid, or negative monetary values")
            frame[col] = converted
        for col in [c for c in frame if c.endswith("_state") or c.endswith("_category_name")]:
            frame[col] = frame[col].astype("string").str.strip()
            frame[col] = frame[col].str.upper() if col.endswith("_state") else frame[col].str.lower()
        frames[name] = frame
    translation_path = raw_dir / "product_category_name_translation.csv"
    translation = pd.read_csv(translation_path) if translation_path.exists() else pd.DataFrame(columns=["product_category_name", "product_category_name_english"])
    if translation.product_category_name.duplicated().any():
        raise ValueError("Category translation has duplicate keys")
    frames["category_translation"] = translation
    relationships = [("orders", "customer_id", "customers", "customer_id"),
                     ("order_items", "order_id", "orders", "order_id"),
                     ("order_items", "product_id", "products", "product_id"),
                     ("order_items", "seller_id", "sellers", "seller_id"),
                     ("order_payments", "order_id", "orders", "order_id"),
                     ("order_reviews", "order_id", "orders", "order_id")]
    for child, key, parent, pk in relationships:
        orphan = ~frames[child][key].isin(frames[parent][pk])
        if orphan.any():
            raise ValueError(f"{child}.{key}: {orphan.sum()} orphan records")
    orders = frames["orders"]
    check("orders: missing purchase timestamps", orders.order_purchase_timestamp.isna().sum())
    check("orders: missing promised dates", orders.order_estimated_delivery_date.isna().sum())
    check("orders: delivery before purchase", (orders.order_delivered_customer_date < orders.order_purchase_timestamp).sum())
    check("orders: promise before purchase", (orders.order_estimated_delivery_date < orders.order_purchase_timestamp).sum())
    check("orders: delivered status without delivery timestamp", (orders.order_status.eq("delivered") & orders.order_delivered_customer_date.isna()).sum())
    reviews = frames["order_reviews"]
    invalid_review = ~reviews.review_score.between(1, 5)
    check("reviews: invalid scores excluded", invalid_review.sum())
    frames["order_reviews"] = reviews.loc[~invalid_review].copy()
    check("products: missing categories", frames["products"].product_category_name.isna().sum())
    check("reviews: extra rows collapsed to latest per order", frames["order_reviews"].duplicated("order_id").sum(), "info")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        con.execute("BEGIN TRANSACTION")
        for name, frame in frames.items():
            con.register("incoming", frame)
            con.execute(f"CREATE OR REPLACE TABLE raw_{name} AS SELECT * FROM incoming")
            con.unregister("incoming")
        con.execute(MART_SQL)
        revenue = con.execute("SELECT sum(price) FROM raw_order_items").fetchone()[0] or 0
        mart_revenue = con.execute("SELECT sum(item_value) FROM fact_orders").fetchone()[0] or 0
        if abs(revenue - mart_revenue) > .01:
            raise ValueError("Revenue conservation failed: order joins have lost or multiplied items")
        if con.execute("SELECT count(*)-count(DISTINCT order_id) FROM fact_orders").fetchone()[0]:
            raise ValueError("Order mart contains duplicate order IDs")
        mismatch = con.execute("SELECT count(*) FROM fact_orders WHERE abs(payment_value-item_value-freight_value)>0.05 AND payment_count>0").fetchone()[0]
        check("orders: payments differ from item plus freight by > R$0.05", mismatch)
        check("orders: no items", con.execute("SELECT count(*) FROM fact_orders WHERE item_count=0").fetchone()[0])
        check("orders: no payments", con.execute("SELECT count(*) FROM fact_orders WHERE payment_count=0").fetchone()[0])
        quality = pd.DataFrame(checks)
        con.register("quality_incoming", quality)
        con.execute("CREATE OR REPLACE TABLE data_quality AS SELECT * FROM quality_incoming")
        meta = pd.DataFrame([{"source": source, "built_at": pd.Timestamp.now().isoformat(), "raw_directory": str(raw_dir.resolve()),
                              "snapshot_date": str(orders.order_purchase_timestamp.max()), "revenue_definition": "Item value, excluding freight; non-canceled/non-unavailable orders"}])
        con.register("meta_incoming", meta)
        con.execute("CREATE OR REPLACE TABLE metadata AS SELECT * FROM meta_incoming")
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.close()
    return {"source": source, "orders": len(orders), "item_value": round(float(revenue), 2), "checks": checks}


MART_SQL = """
CREATE OR REPLACE TABLE dim_locations AS
SELECT geolocation_zip_code_prefix AS zip_prefix,
       median(geolocation_lat) AS latitude, median(geolocation_lng) AS longitude
FROM raw_geolocation GROUP BY 1;

CREATE OR REPLACE TABLE fact_items AS
SELECT i.*, coalesce(t.product_category_name_english, p.product_category_name, 'unknown') AS category,
       p.product_weight_g, p.product_length_cm, p.product_height_cm, p.product_width_cm,
       s.seller_state, o.order_purchase_timestamp, o.order_status, c.customer_state,
       c.customer_unique_id
FROM raw_order_items i
LEFT JOIN raw_products p USING(product_id)
LEFT JOIN raw_category_translation t USING(product_category_name)
LEFT JOIN raw_sellers s USING(seller_id)
LEFT JOIN raw_orders o USING(order_id)
LEFT JOIN raw_customers c USING(customer_id);

CREATE OR REPLACE TABLE fact_orders AS
WITH items AS (
 SELECT order_id, sum(price) AS item_value, sum(freight_value) AS freight_value,
        count(*) AS item_count, count(DISTINCT seller_id) AS seller_count,
        avg(product_weight_g) AS mean_weight_g,
        avg(CASE WHEN seller_state = customer_state THEN 0.0 ELSE 1.0 END) AS interstate_share
 FROM fact_items GROUP BY order_id
), dominant AS (
 SELECT order_id, category, seller_state,
        row_number() OVER(PARTITION BY order_id ORDER BY price DESC, order_item_id) AS rn
 FROM fact_items
), payments AS (
 SELECT order_id, sum(payment_value) AS payment_value, count(*) AS payment_count
 FROM raw_order_payments GROUP BY order_id
), reviews AS (
 SELECT order_id, review_score,
        row_number() OVER(PARTITION BY order_id ORDER BY review_answer_timestamp DESC NULLS LAST, review_id DESC, review_score DESC) AS rn
 FROM raw_order_reviews
)
SELECT o.*, c.customer_unique_id, c.customer_state,
       coalesce(i.item_value, 0) AS item_value, coalesce(i.freight_value, 0) AS freight_value,
       coalesce(i.item_count, 0) AS item_count, coalesce(i.seller_count, 0) AS seller_count,
       i.mean_weight_g, i.interstate_share, coalesce(d.category,'unknown') AS category, d.seller_state,
       coalesce(p.payment_value,0) AS payment_value, coalesce(p.payment_count,0) AS payment_count,
       r.review_score,
       date_diff('day', o.order_purchase_timestamp, o.order_delivered_customer_date) AS delivery_days,
       date_diff('day', o.order_purchase_timestamp, o.order_estimated_delivery_date) AS promised_days,
       CASE WHEN o.order_status='delivered'
                 AND o.order_delivered_customer_date >= o.order_purchase_timestamp
                 AND o.order_estimated_delivery_date >= o.order_purchase_timestamp
            THEN CAST(CAST(o.order_delivered_customer_date AS DATE) > CAST(o.order_estimated_delivery_date AS DATE) AS INTEGER)
            ELSE NULL END AS is_late,
       o.order_status NOT IN ('canceled','unavailable') AS is_sales_order
FROM raw_orders o
LEFT JOIN raw_customers c USING(customer_id)
LEFT JOIN items i USING(order_id)
LEFT JOIN dominant d ON o.order_id=d.order_id AND d.rn=1
LEFT JOIN payments p USING(order_id)
LEFT JOIN reviews r ON o.order_id=r.order_id AND r.rn=1;
"""
