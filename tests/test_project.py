from pathlib import Path
import tempfile
import unittest

import duckdb
import pandas as pd
from streamlit.testing.v1 import AppTest

from marketplace.analytics import cohorts, generate_report
from marketplace.demo import generate_demo
from marketplace.model import FEATURES, train_model
from marketplace.pipeline import build_database


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
        # Every cohort has no return; observed missing ages still need zero columns.
        self.assertEqual(result.loc["2017-03", 0], 1)


class DashboardTests(unittest.TestCase):
    def test_all_pages(self):
        app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=40).run()
        self.assertFalse(app.exception)
        if not app.sidebar.radio:
            self.skipTest("Run olist demo before dashboard integration tests")
        for page in ["Business overview", "Delivery & reviews", "Customers", "Late-delivery model", "Data quality", "Evidence desk"]:
            app.sidebar.radio[0].set_value(page).run()
            self.assertFalse(app.exception, f"Dashboard failed on {page}")


if __name__ == "__main__":
    unittest.main()
