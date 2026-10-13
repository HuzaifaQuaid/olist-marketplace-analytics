"""Deterministic synthetic fixture. Never presented as real Olist evidence."""
from pathlib import Path

import numpy as np
import pandas as pd


def generate_demo(directory: Path, n_orders: int = 2400, seed: int = 42) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    states = np.array(["SP", "RJ", "MG", "BA", "PR", "RS"])
    cities = {"SP": "sao paulo", "RJ": "rio de janeiro", "MG": "belo horizonte",
              "BA": "salvador", "PR": "curitiba", "RS": "porto alegre"}
    categories = ["beleza_saude", "cama_mesa_banho", "esporte_lazer", "informatica_acessorios", "moveis_decoracao", "relogios_presentes"]
    english = ["health_beauty", "bed_bath_table", "sports_leisure", "computers_accessories", "furniture_decor", "watches_gifts"]
    products = pd.DataFrame({
        "product_id": [f"p{i:04}" for i in range(150)],
        "product_category_name": rng.choice(categories, 150),
        "product_name_lenght": rng.integers(15, 65, 150),
        "product_description_lenght": rng.integers(80, 1800, 150),
        "product_photos_qty": rng.integers(1, 5, 150),
        "product_weight_g": rng.integers(100, 7000, 150),
        "product_length_cm": rng.integers(10, 65, 150),
        "product_height_cm": rng.integers(5, 40, 150),
        "product_width_cm": rng.integers(8, 50, 150),
    })
    sellers = pd.DataFrame({"seller_id": [f"s{i:03}" for i in range(28)],
                            "seller_state": rng.choice(states, 28),
                            "seller_zip_code_prefix": np.arange(10000, 10028)})
    sellers["seller_city"] = sellers.seller_state.map(cities)
    rows, customers, items, payments, reviews = [], [], [], [], []
    for i in range(n_orders):
        oid, cid = f"o{i:06}", f"c{i:06}"
        state = rng.choice(states, p=[.40, .18, .15, .10, .09, .08])
        date = pd.Timestamp("2017-01-01") + pd.Timedelta(days=int(rng.integers(0, 608)), hours=int(rng.integers(0, 24)))
        promise = date.normalize() + pd.Timedelta(days=int(rng.integers(14, 32)))
        seller = sellers.iloc[int(rng.integers(0, len(sellers)))]
        late = rng.random() < (.07 + .13 * (state != seller.seller_state) + .13 * (seller.seller_id in ["s003", "s011"]))
        delivered = promise + pd.Timedelta(days=int(rng.integers(1, 13)) if late else -int(rng.integers(1, 12)))
        canceled = rng.random() < .025
        rows.append({"order_id": oid, "customer_id": cid, "order_status": "canceled" if canceled else "delivered",
                     "order_purchase_timestamp": date, "order_approved_at": date + pd.Timedelta(hours=2),
                     "order_delivered_carrier_date": date + pd.Timedelta(days=3) if not canceled else pd.NaT,
                     "order_delivered_customer_date": delivered if not canceled else pd.NaT,
                     "order_estimated_delivery_date": promise})
        customers.append({"customer_id": cid, "customer_unique_id": f"u{int(rng.integers(0, n_orders * .8)):06}",
                          "customer_zip_code_prefix": 20000 + int(np.where(states == state)[0][0]),
                          "customer_city": cities[state], "customer_state": state})
        total = 0.0
        for j in range(1, int(rng.choice([1, 2, 3], p=[.74, .20, .06])) + 1):
            product = products.iloc[int(rng.integers(0, len(products)))]
            price = round(float(rng.lognormal(4.4, .65)), 2)
            freight = round(float(rng.uniform(9, 24) + 10 * (state != seller.seller_state)), 2)
            items.append({"order_id": oid, "order_item_id": j, "product_id": product.product_id,
                          "seller_id": seller.seller_id, "shipping_limit_date": date + pd.Timedelta(days=5),
                          "price": price, "freight_value": freight})
            total += price + freight
        splits = 2 if rng.random() < .15 else 1
        for k in range(1, splits + 1):
            amount = round(total / splits, 2) if k == 1 else round(total - round(total / splits, 2), 2)
            payments.append({"order_id": oid, "payment_sequential": k, "payment_type": "credit_card",
                             "payment_installments": int(rng.integers(1, 7)), "payment_value": amount})
        if not canceled:
            score = int(rng.choice([1, 2, 3, 4, 5], p=[.35, .25, .20, .12, .08] if late else [.04, .03, .09, .22, .62]))
            reviews.append({"review_id": f"r{i:06}", "order_id": oid, "review_score": score,
                            "review_comment_title": None, "review_comment_message": None,
                            "review_creation_date": delivered + pd.Timedelta(days=1),
                            "review_answer_timestamp": delivered + pd.Timedelta(days=2)})
    locations = pd.DataFrame({"geolocation_zip_code_prefix": list(range(10000, 10028)) + list(range(20000, 20006)),
                              "geolocation_lat": rng.uniform(-30, -12, 34), "geolocation_lng": rng.uniform(-55, -38, 34),
                              "geolocation_city": "synthetic", "geolocation_state": "SP"})
    tables = {"orders": pd.DataFrame(rows), "customers": pd.DataFrame(customers), "order_items": pd.DataFrame(items),
              "order_payments": pd.DataFrame(payments), "order_reviews": pd.DataFrame(reviews),
              "products": products, "sellers": sellers, "geolocation": locations}
    for name, frame in tables.items():
        frame.to_csv(directory / f"olist_{name}_dataset.csv", index=False)
    pd.DataFrame({"product_category_name": categories, "product_category_name_english": english}).to_csv(
        directory / "product_category_name_translation.csv", index=False)
