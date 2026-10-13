import json
import sys
import tempfile
import unittest
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from marketplace.analyst import Analyst  # noqa: E402
from marketplace.analytics import cohorts, generate_report  # noqa: E402
from marketplace.demo import generate_demo  # noqa: E402
from marketplace.forecasting import load_forecast, train_forecast  # noqa: E402
from marketplace.model import FEATURES, train_model  # noqa: E402
from marketplace.pipeline import build_database  # noqa: E402
from marketplace.recommender import Recommender, build_catalog, train_recommender  # noqa: E402
from marketplace.repurchase import first_purchase_table, train_repurchase  # noqa: E402
from marketplace.stats import delivery_review_adjusted, freight_by_region, shrunk_seller_late_rates  # noqa: E402


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw"
        self.db = self.root / "test.duckdb"
        generate_demo(self.raw, n_orders=600)

    def tearDown(self):
        self.temp.cleanup()

    def build(self):
        return build_database(self.raw, self.db, "synthetic")

    def test_grain_and_money_conservation(self):
        self.build()
        with duckdb.connect(str(self.db)) as con:
            self.assertEqual(con.execute("SELECT count(*) FROM fact_orders").fetchone()[0], 600)
            item_total = con.execute("SELECT sum(price) FROM raw_order_items").fetchone()[0]
            order_total = con.execute("SELECT sum(item_value) FROM fact_orders").fetchone()[0]
            payment_total = con.execute("SELECT sum(payment_value) FROM raw_order_payments").fetchone()[0]
            mart_payment = con.execute("SELECT sum(payment_value) FROM fact_orders").fetchone()[0]
            self.assertAlmostEqual(item_total, order_total, places=6)
            self.assertAlmostEqual(payment_total, mart_payment, places=6)
            # The demo intentionally has multi-item and multi-payment orders.
            multiplied = con.execute("SELECT sum(i.price) FROM raw_order_items i JOIN raw_order_payments p USING(order_id)").fetchone()[0]
            self.assertGreater(multiplied, item_total)

    def test_conflicting_key_rejected(self):
        path = self.raw / "olist_orders_dataset.csv"
        frame = pd.read_csv(path)
        duplicate = frame.iloc[[0]].copy()
        duplicate["order_status"] = "unavailable"
        pd.concat([frame, duplicate]).to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "conflicting primary keys"):
            self.build()

    def test_orphans_rejected(self):
        path = self.raw / "olist_order_items_dataset.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "product_id"] = "missing"
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "orphan"):
            self.build()

    def test_delivery_calendar_day_and_invalid_timeline(self):
        path = self.raw / "olist_orders_dataset.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "order_status"] = "delivered"
        frame.loc[0, "order_purchase_timestamp"] = "2017-01-01 12:00:00"
        frame.loc[0, "order_estimated_delivery_date"] = "2017-01-20 00:00:00"
        frame.loc[0, "order_delivered_customer_date"] = "2017-01-20 18:00:00"
        frame.loc[1, "order_delivered_customer_date"] = "2010-01-01 00:00:00"
        frame.to_csv(path, index=False)
        self.build()
        with duckdb.connect(str(self.db)) as con:
            self.assertEqual(con.execute("SELECT is_late FROM fact_orders WHERE order_id='o000000'").fetchone()[0], 0)
            self.assertIsNone(con.execute("SELECT is_late FROM fact_orders WHERE order_id='o000001'").fetchone()[0])

    def test_training_and_report(self):
        self.build()
        report = train_model(self.db, self.root / "artifacts")
        self.assertEqual(set(report["metrics"]), {"prevalence_baseline", "logistic_regression", "random_forest"})
        self.assertNotIn("order_delivered_customer_date", FEATURES)
        self.assertNotIn("review_score", FEATURES)
        self.assertGreater(report["test_orders"], 30)
        self.assertEqual(report["source"], "synthetic")
        self.assertIn("threshold_selected_on_validation", report["metrics"]["logistic_regression"])
        self.assertGreaterEqual(report["threshold"], 0)
        self.assertLessEqual(report["threshold"], 1)
        text = generate_report(self.db, self.root / "findings.md")
        self.assertIn("Synthetic demonstration only", text)

    def test_cohort_future_blank_observed_zero(self):
        frame = pd.DataFrame({"order_id": ["1", "2", "3"], "customer_unique_id": ["a", "b", "c"],
                              "order_purchase_timestamp": pd.to_datetime(["2017-01-01", "2017-02-01", "2017-03-01"]),
                              "is_sales_order": True, "item_count": 1})
        result = cohorts(frame)
        self.assertEqual(result.loc["2017-01", 0], 1)
        self.assertEqual(result.loc["2017-01", 1], 0)
        self.assertTrue(pd.isna(result.loc["2017-03", 1]))


class StageTests(unittest.TestCase):
    """Forecasting, recommendations, repurchase, investigation and the analyst layer."""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.raw = cls.root / "raw"
        cls.db = cls.root / "stage.duckdb"
        cls.artifacts = cls.root / "artifacts"
        generate_demo(cls.raw, n_orders=2000)
        build_database(cls.raw, cls.db, "synthetic")
        cls.forecast = train_forecast(cls.db, cls.artifacts)
        cls.recommender = train_recommender(cls.db, cls.artifacts)
        cls.repurchase = train_repurchase(cls.db, cls.artifacts)
        with duckdb.connect(str(cls.db), read_only=True) as con:
            cls.orders = con.execute("SELECT * FROM fact_orders").df()
            cls.items = con.execute("SELECT * FROM fact_items").df()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_forecast_is_temporal_and_reports_uncertainty(self):
        self.assertGreaterEqual(self.forecast["complete_months"], 12)
        self.assertEqual(set(self.forecast["metrics"]), {"naive_last_value", "seasonal_naive_12m", "calendar_lag_model"})
        for metrics in self.forecast["metrics"].values():
            self.assertGreater(metrics["folds"], 0)
            self.assertIsNotNone(metrics["mape"])
        forecast = self.forecast["forecast"]["calendar_lag_model"]
        self.assertLess(forecast["low_95"], forecast["point"])
        self.assertLess(forecast["point"], forecast["high_95"])
        self.assertGreater(pd.Timestamp(self.forecast["next_month"]), pd.Timestamp(self.forecast["series_end"]))
        self.assertEqual(load_forecast(self.artifacts)["next_month"], self.forecast["next_month"])

    def test_recommender_excludes_seen_items_and_cold_starts(self):
        catalog = build_catalog(self.items)
        model = Recommender(catalog)
        seed = model.popularity[0]
        similar = model.similar_to_product(seed, 5)
        self.assertEqual(len(similar), 5)
        self.assertNotIn(seed, similar.product_id.tolist())
        purchased = model.popularity[:3]
        personalised = model.for_customer(purchased, 5)
        self.assertFalse(set(personalised.product_id) & set(purchased))
        unknown = model.for_customer(["not-a-product"], 4)
        self.assertEqual(len(unknown), 4)
        self.assertTrue(model.similar_to_product("missing", 3).reason.str.contains("popularity").all())
        self.assertIn("metrics", self.recommender)
        self.assertIn("train_catalog_size", self.recommender)
        self.assertIn("test_products_excluded_as_cold_start", self.recommender)
        self.assertTrue(any("training window" in limitation for limitation in self.recommender["limitations"]))

    def test_ranking_recall_is_fraction_not_hit_indicator(self):
        from marketplace.recommender import _rank_metrics
        recall, precision, average_precision = _rank_metrics(["a", "b", "c", "d"], {"b", "d", "x", "y"}, 4)
        self.assertAlmostEqual(recall, .5)
        self.assertAlmostEqual(precision, .5)
        self.assertGreater(average_precision, 0)

    def test_repurchase_label_respects_horizon_and_censoring(self):
        table = first_purchase_table(self.orders, 90)
        self.assertGreater(len(table), 500)
        # No labelled customer may be censored: first purchase plus horizon must be observable.
        snapshot = self.orders.order_purchase_timestamp.max()
        self.assertTrue((table.first_purchase + pd.Timedelta(days=90) <= snapshot).all())
        metrics = self.repurchase["metrics"]["logistic_regression"]
        self.assertGreaterEqual(metrics["roc_auc"], 0.5)
        for row in self.repurchase["calibration"]:
            self.assertLess(abs(row["predicted"] - row["actual"]), 0.35)

    def test_investigation_shrinks_and_bounds(self):
        adjusted = delivery_review_adjusted(self.orders)
        self.assertTrue(adjusted["available"])
        self.assertLess(adjusted["ci95_low"], adjusted["late_coefficient"])
        self.assertLess(adjusted["late_coefficient"], adjusted["ci95_high"])
        self.assertLess(adjusted["late_coefficient"], 0)
        sellers = shrunk_seller_late_rates(self.orders, self.items, min_orders=10)
        self.assertFalse(sellers.empty)
        self.assertTrue(sellers.shrunk_rate.between(0, 1).all())
        # Shrinkage must be monotone toward the marketplace rate.
        self.assertTrue((abs(sellers.shrunk_rate - sellers.marketplace_rate) <= abs(sellers.raw_rate - sellers.marketplace_rate) + 1e-9).all())
        freight = freight_by_region(self.orders, iterations=50)
        self.assertTrue((freight.ci95_low <= freight.mean_freight_per_item + 1e-9).all())
        self.assertTrue((freight.mean_freight_per_item <= freight.ci95_high + 1e-9).all())

    def test_analyst_is_grounded(self):
        analyst = Analyst(self.db, self.artifacts)
        answer = analyst.ask("Which categories had the biggest sales decline?")
        self.assertEqual(answer["tool"], "category_movers")
        self.assertIn("evidence", answer)
        refusal = analyst.ask("Can we calculate profit for each product?")
        self.assertIsNone(refusal["tool"])
        self.assertIn("profit", refusal["answer"].lower())
        unknown = analyst.ask("tell me about zzzz nonsense")
        self.assertIsNone(unknown["tool"])
        every = analyst.ask("What sales can we expect next month?")
        self.assertIn(every["tool"], {None, "forecast"})
        # The planner can only ever return an approved tool name.
        self.assertTrue(set(analyst.tools) >= {"forecast", "recommend", "repurchase", "seller_late_rates"})


class DashboardTests(unittest.TestCase):
    def test_all_pages(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=90).run()
        self.assertFalse(app.exception)
        if not app.sidebar.radio:
            self.skipTest("Run olist demo before dashboard integration tests")
        pages = ["Business overview", "Delivery & reviews", "Investigation", "Customers", "Repurchase model",
                 "Forecasting", "Recommendations", "Late-delivery model", "AI analyst", "Data quality"]
        for page in pages:
            app.sidebar.radio[0].set_value(page).run()
            self.assertFalse(app.exception, f"Dashboard failed on {page}")


if __name__ == "__main__":
    unittest.main()
