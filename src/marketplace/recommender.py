"""Popularity and content-based recommendations, evaluated on a temporal holdout."""
from pathlib import Path
import json

import duckdb
import numpy as np
import pandas as pd

CATEGORY_WEIGHT = 2.5
MIN_TRAIN_PURCHASES = 1
DEFAULT_K = 10
EVAL_CUSTOMERS = 1200


def build_catalog(items: pd.DataFrame) -> pd.DataFrame:
    sales = items.loc[~items.order_status.isin(["canceled", "unavailable"])].copy()
    catalog = sales.groupby("product_id").agg(
        category=("category", "first"),
        weight_g=("product_weight_g", "mean"),
        length_cm=("product_length_cm", "mean"),
        height_cm=("product_height_cm", "mean"),
        width_cm=("product_width_cm", "mean"),
        mean_price=("price", "mean"),
        orders=("order_id", "nunique"),
        item_sales=("price", "sum")).reset_index()
    catalog["category"] = catalog.category.fillna("unknown")
    return catalog


class Recommender:
    """Cosine similarity on category plus scaled physical attributes; no full product-by-product matrix is stored."""

    def __init__(self, catalog: pd.DataFrame):
        self.catalog = catalog.reset_index(drop=True)
        self.index = {product: row for row, product in enumerate(self.catalog.product_id)}
        categories = pd.get_dummies(self.catalog.category, dtype=float)
        numeric = np.column_stack([
            np.log1p(self.catalog.weight_g.fillna(self.catalog.weight_g.median()).to_numpy(dtype=float)),
            np.log1p((self.catalog.length_cm.fillna(20) * self.catalog.height_cm.fillna(15) * self.catalog.width_cm.fillna(20)).to_numpy(dtype=float)),
            np.log1p(self.catalog.mean_price.fillna(self.catalog.mean_price.median()).to_numpy(dtype=float)),
        ])
        numeric = (numeric - numeric.mean(axis=0)) / np.where(numeric.std(axis=0) == 0, 1, numeric.std(axis=0))
        matrix = np.column_stack([categories.to_numpy() * CATEGORY_WEIGHT, numeric])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self.matrix = matrix / np.where(norms == 0, 1, norms)
        self.popularity = self.catalog.sort_values(["orders", "item_sales"], ascending=False).product_id.tolist()

    def similar_to_product(self, product_id: str, k: int = DEFAULT_K) -> pd.DataFrame:
        if product_id not in self.index:
            return self._popular(k, reason="unknown product; showing overall popularity")
        scores = self.matrix @ self.matrix[self.index[product_id]]
        order = np.argsort(-scores)
        picked = [row for row in order if row != self.index[product_id]][:k]
        result = self.catalog.iloc[picked].copy()
        result["score"] = scores[picked]
        result["reason"] = "similar attributes to " + product_id
        return result

    def for_customer(self, purchased: list[str], k: int = DEFAULT_K) -> pd.DataFrame:
        seeds = [product for product in purchased if product in self.index]
        if not seeds:
            return self._popular(k, reason="no recognizable purchase history; showing overall popularity")
        seed_vectors = self.matrix[[self.index[product] for product in seeds]]
        scores = self.matrix @ seed_vectors.mean(axis=0)
        scores[[self.index[product] for product in seeds]] = -np.inf
        order = np.argsort(-scores)[:k]
        result = self.catalog.iloc[order].copy()
        result["score"] = scores[order]
        result["reason"] = "content similar to products this customer bought"
        return result

    def _popular(self, k: int, reason: str) -> pd.DataFrame:
        rows = [self.index[product] for product in self.popularity[:k]]
        result = self.catalog.iloc[rows].copy()
        result["score"] = np.nan
        result["reason"] = reason
        return result


def _rank_metrics(recommended: list[str], relevant: set[str], k: int) -> tuple[float, float, float]:
    hits = [position for position, product in enumerate(recommended[:k]) if product in relevant]
    recall = 1.0 if hits else 0.0
    precision = len(hits) / k
    average_precision = sum(1.0 / (position + 1) for position in hits) / min(len(relevant), k)
    return recall, precision, average_precision


def evaluate(recommender: Recommender, items: pd.DataFrame, k: int = DEFAULT_K) -> dict:
    """Train on earlier purchases, test whether later purchases appear in the top-K. Cold-start ignores unknown products."""
    sales = items.loc[~items.order_status.isin(["canceled", "unavailable"]) & items.customer_unique_id.notna()
                      & items.order_purchase_timestamp.notna()].copy()
    if sales.empty:
        return {"available": False, "reason": "No dated item purchases are available."}
    cutoff = sales.order_purchase_timestamp.quantile(.8)
    train = sales.loc[sales.order_purchase_timestamp < cutoff]
    test = sales.loc[sales.order_purchase_timestamp >= cutoff]
    train_history = train.groupby("customer_unique_id").product_id.apply(list)
    test_history = test.groupby("customer_unique_id").product_id.apply(set)
    evaluable = [customer for customer in test_history.index
                 if customer in train_history.index and test_history[customer] - set(train_history[customer])]
    popularity_rank = recommender.popularity
    if len(evaluable) > EVAL_CUSTOMERS:
        evaluable = list(pd.Series(evaluable).sample(EVAL_CUSTOMERS, random_state=42))
    rows = {"popularity": [], "content_based": []}
    for customer in evaluable:
        relevant = test_history[customer] - set(train_history[customer])
        purchased = train_history[customer]
        rows["popularity"].append(_rank_metrics(popularity_rank, relevant, k))
        rows["content_based"].append(_rank_metrics(recommender.for_customer(purchased, k).product_id.tolist(), relevant, k))
    metrics = {}
    for name, values in rows.items():
        array = np.array(values, dtype=float)
        metrics[name] = {"evaluable_customers": len(values), "recall_at_k": float(array[:, 0].mean()),
                         "precision_at_k": float(array[:, 1].mean()), "map_at_k": float(array[:, 2].mean())}
    return {"available": True, "k": k, "cutoff": str(cutoff), "customers_with_repeat_history": int(len(evaluable)),
            "train_purchases": int(len(train)), "test_purchases": int(len(test)),
            "metrics": metrics, "catalog_size": int(len(recommender.catalog)),
            "cold_start_products": int(recommender.catalog.category.eq("unknown").sum()),
            "limitations": ["Olist has very little repeat purchasing, so only a small share of customers can be evaluated and results are unstable.",
                            "Both baselines ignore price, availability and business constraints; a real system needs ranking objectives and online tests.",
                            "Similarity uses category and physical attributes only: there is no browsing, click or basket-overlap signal.",
                            "Cold-start products with no category or attributes fall back to popularity."]}


def train_recommender(db_path: Path, artifact_dir: Path) -> dict:
    with duckdb.connect(str(db_path), read_only=True) as con:
        items = con.execute("SELECT * FROM fact_items").df()
        source, built_at = con.execute("SELECT source, built_at FROM metadata").fetchone()
    recommender = Recommender(build_catalog(items))
    report = evaluate(recommender, items)
    report.update({"source": source, "database_built_at": built_at})
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "recommender_report.json").write_text(json.dumps(report, indent=2))
    return report


def load_recommender_report(artifact_dir: Path) -> dict | None:
    path = artifact_dir / "recommender_report.json"
    return json.loads(path.read_text()) if path.exists() else None
