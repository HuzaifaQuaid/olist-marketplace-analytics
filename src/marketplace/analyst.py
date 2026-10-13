"""Grounded analyst: an LLM may only choose from approved tools, and every answer shows its evidence.

Without an API key the analyst still answers using a deterministic keyword router, so the
dashboard never depends on a paid service and never generates free-form SQL.
"""
from pathlib import Path
import json
import os

import duckdb
import numpy as np
import pandas as pd
import requests

from marketplace import forecasting, stats
from marketplace.analytics import delivery_association
from marketplace.recommender import Recommender, build_catalog, load_recommender_report
from marketplace.repurchase import load_repurchase_report

MAX_EVIDENCE_ROWS = 8
REFUSALS = {
    "profit": "This dataset has no reliable product costs, so profit, margin and profitability cannot be calculated. Sales and freight are not profit.",
    "inventory": "Inventory levels and stock-outs are not recorded, so stock optimisation and reorder points cannot be computed.",
    "browsing": "There is no browsing, click or session data, so funnels, cart abandonment and page-level behaviour cannot be analysed.",
    "marketing": "Ad spend, campaign or channel attribution data is absent, so return on ad spend and channel performance cannot be measured.",
    "demographics": "Customer age, gender, income and other demographics are not present, so audience segmentation by those traits is impossible.",
}


class Analyst:
    def __init__(self, db_path: Path, artifact_dir: Path | None = None):
        with duckdb.connect(str(db_path), read_only=True) as con:
            self.orders = con.execute("SELECT * FROM fact_orders").df()
            self.items = con.execute("SELECT * FROM fact_items").df()
            self.quality = con.execute("SELECT * FROM data_quality").df()
            self.metadata = con.execute("SELECT * FROM metadata").df().iloc[0].to_dict()
        self.artifact_dir = artifact_dir
        self.catalog = build_catalog(self.items)
        self.recommender = Recommender(self.catalog)
        self.tools = {
            "sales_overview": {"keywords": ["revenue", "sales", "sell", "aov", "order value", "turnover", "gmv"],
                               "description": "Total item sales, order count, average order value and the leading category.", "run": self._sales_overview},
            "category_movers": {"keywords": ["decline", "declined", "growth", "grew", "drop", "down", "falling", "rising", "movers", "category", "categories"],
                                "description": "Categories with the biggest sales decline or growth, last complete month versus the prior three.", "run": self._category_movers},
            "seller_late_rates": {"keywords": ["seller", "delivery", "late", "delayed", "supplier", "watchlist"],
                                  "description": "Sellers with the highest shrunk late-delivery rates.", "run": self._seller_late_rates},
            "delivery_review_association": {"keywords": ["review", "rating", "satisfaction", "score", "association", "relationship"],
                                            "description": "Adjusted and unadjusted association between late delivery and review score.", "run": self._delivery_review},
            "forecast": {"keywords": ["forecast", "predict", "next month", "future", "expected sales", "projection", "outlook"],
                         "description": "Backtested next-month sales forecast with intervals and its limitations.", "run": self._forecast},
            "recommend": {"keywords": ["recommend", "suggest", "similar", "upsell", "cross-sell", "what should"],
                          "description": "Product recommendations with content similarity and their evaluation metrics.", "run": self._recommend},
            "repurchase": {"keywords": ["repurchase", "buy again", "repeat", "return", "loyal", "churn", "retention"],
                           "description": "How well the models rank customers who buy again, plus support for identifying them.", "run": self._repurchase},
            "data_quality": {"keywords": ["quality", "missing", "duplicate", "trust", "clean", "validity"],
                             "description": "Validation checks and their nonzero counts.", "run": self._data_quality},
        }

    # ---- evidence tools -------------------------------------------------
    def _sales(self) -> pd.DataFrame:
        return self.orders.loc[self.orders.is_sales_order & self.orders.item_count.gt(0)]

    def _months(self) -> list:
        return forecasting.monthly_series(self.orders).month.tolist()

    def _sales_overview(self, params: dict) -> dict:
        sales = self._sales()
        if sales.empty:
            return {"answer": "No eligible sales exist in this snapshot.", "evidence": {}, "caveats": []}
        items = self.items.loc[~self.items.order_status.isin(["canceled", "unavailable"])]
        leaders = items.groupby("category").price.sum().sort_values(ascending=False)
        total = float(sales.item_value.sum())
        top = leaders.index[0]
        return {"answer": f"Item sales in this snapshot are {_brl(total)} across {len(sales):,} orders, "
                          f"an average order value of {_brl(sales.item_value.mean())}. {top} is the leading category at {_brl(leaders.iloc[0])}.",
                "evidence": {"item_sales": total, "orders": int(len(sales)), "aov": float(sales.item_value.mean()),
                             "leading_categories": [{"category": name, "item_sales": float(value)} for name, value in leaders.head(5).items()]},
                "caveats": ["Item sales exclude freight and canceled/unavailable orders.", "This is historical Olist data, not a live feed."]}

    def _category_movers(self, params: dict) -> dict:
        months = self._months()
        if len(months) < 5:
            return {"answer": "Too few complete months to compare category momentum.", "evidence": {}, "caveats": []}
        items = self.items.loc[~self.items.order_status.isin(["canceled", "unavailable"]) & self.items.order_purchase_timestamp.notna()].copy()
        items["month"] = items.order_purchase_timestamp.dt.to_period("M").dt.to_timestamp()
        items = items.loc[items.month.isin(months)]
        pivot = items.pivot_table(index="category", columns="month", values="price", aggfunc="sum").fillna(0.0)
        ordered = [month for month in months if month in pivot.columns]
        if len(ordered) < 4:
            return {"answer": "Not enough complete months per category.", "evidence": {}, "caveats": []}
        latest = pivot[ordered[-1]]
        baseline = pivot[ordered[-4:-1]].mean(axis=1)
        change = latest - baseline
        pct = change / baseline.replace(0, np.nan)
        table = pd.DataFrame({"category": pivot.index, "last_month": latest.to_numpy(), "prior_three_avg": baseline.to_numpy(),
                              "change": change.to_numpy(), "change_pct": pct.to_numpy()})
        table = table.loc[table.prior_three_avg.gt(0)]
        declines = table.sort_values("change").head(MAX_EVIDENCE_ROWS)
        gains = table.sort_values("change", ascending=False).head(MAX_EVIDENCE_ROWS)
        month_label = ordered[-1].strftime("%Y-%m")
        worst = declines.iloc[0]
        direction = "fell by" if worst.change < 0 else "rose by"
        return {"answer": f"In {month_label}, {worst.category} had the largest sales move: it {direction} {_brl(abs(worst.change))} versus its prior three-month average of {_brl(worst.prior_three_avg)}.",
                "evidence": {"compared_month": month_label, "baseline": "mean of the previous three complete months",
                             "largest_declines": declines.round(2).to_dict("records"), "largest_gains": gains.round(2).to_dict("records")},
                "caveats": ["Only complete calendar months are compared; partial boundary months are excluded.",
                            "A single month is noisy, and a decline can reflect seasonality, stock, promotions or platform changes, not necessarily demand."]}

    def _seller_late_rates(self, params: dict) -> dict:
        minimum = int(params.get("min_orders", 20))
        table = stats.shrunk_seller_late_rates(self.orders, self.items, minimum)
        if table.empty:
            return {"answer": "Not enough delivered orders per seller to estimate late-delivery rates.", "evidence": {}, "caveats": []}
        top = table.head(MAX_EVIDENCE_ROWS)[["seller_id", "n", "raw_rate", "shrunk_rate", "marketplace_rate"]]
        leader = top.iloc[0]
        return {"answer": f"{leader.seller_id} has the highest shrunk late-delivery rate at {leader.shrunk_rate:.1%} across {int(leader.n)} orders, "
                          f"against a marketplace rate of {leader.marketplace_rate:.1%}.",
                "evidence": {"min_orders": minimum, "sellers": top.round(4).to_dict("records")},
                "caveats": ["Shrinkage pulls small sellers toward the marketplace rate, so the ranking needs enough orders to be meaningful.",
                            "Multi-seller orders do not identify which seller caused a delay; this is association, not fault."]}

    def _delivery_review(self, params: dict) -> dict:
        adjusted = stats.delivery_review_adjusted(self.orders)
        unadjusted = delivery_association(self.orders)
        if not adjusted.get("available"):
            return {"answer": adjusted.get("reason", "Insufficient evidence."), "evidence": {}, "caveats": []}
        return {"answer": f"Late orders score {abs(adjusted['late_coefficient']):.2f} points lower on average after controlling for order value, freight, "
                          f"category, state and month (95% CI {adjusted['ci95_low']:.2f} to {adjusted['ci95_high']:.2f}). The unadjusted gap was "
                          f"{abs(adjusted['unadjusted_difference']):.2f} points.",
                "evidence": {"adjusted": adjusted, "unadjusted": unadjusted},
                "caveats": adjusted["limitations"]}

    def _forecast(self, params: dict) -> dict:
        report = forecasting.load_forecast(self.artifact_dir) if self.artifact_dir else None
        if not report:
            return {"answer": "No forecast artifact is available. Run the forecasting step first.", "evidence": {}, "caveats": []}
        model = report["forecast"]["calendar_lag_model"]
        naive = report["forecast"]["naive_last_value"]
        metrics = report["metrics"]["calendar_lag_model"]
        return {"answer": f"For {report['next_month']} the model expects about {_brl(model['point'])} in item sales "
                          f"(80% interval {_brl(model['low_80'])} to {_brl(model['high_80'])}). The naive last-value forecast is {_brl(naive['point'])}.",
                "evidence": {"horizon_months": report["horizon_months"], "model_metrics": metrics, "naive_metrics": report["metrics"]["naive_last_value"],
                             "forecast": report["forecast"], "recent_actuals": report["recent_actuals"]},
                "caveats": report["limitations"]}

    def _recommend(self, params: dict) -> dict:
        product = params.get("product_id")
        category = params.get("category")
        if product and product in self.recommender.index:
            table = self.recommender.similar_to_product(product, int(params.get("k", 5)))
        elif category:
            subset = self.catalog.loc[self.catalog.category.eq(category)]
            if subset.empty:
                return {"answer": f"No products are recorded for category {category}.", "evidence": {}, "caveats": []}
            table = self.recommender.similar_to_product(subset.product_id.iloc[0], int(params.get("k", 5)))
        else:
            popular = self.recommender.popularity[:5]
            table = self.catalog.loc[self.catalog.product_id.isin(popular)]
        report = load_recommender_report(self.artifact_dir) if self.artifact_dir else None
        columns = ["product_id", "category", "mean_price", "orders"]
        return {"answer": f"Top recommendations: {', '.join(table.product_id.head(5))}.",
                "evidence": {"recommendations": table[[column for column in columns if column in table]].round(2).to_dict("records"),
                             "evaluation": report.get("metrics") if report else None},
                "caveats": report["limitations"] if report else ["No evaluation report is available yet."]}

    def _repurchase(self, params: dict) -> dict:
        report = load_repurchase_report(self.artifact_dir) if self.artifact_dir else None
        if not report:
            return {"answer": "No repurchase model artifact is available. Build it first.", "evidence": {}, "caveats": []}
        metrics = report["metrics"]["logistic_regression"]
        return {"answer": f"Repurchase within {report['horizon_days']} days has a base rate of {report['overall_repeat_rate']:.1%}. "
                          f"The logistic model reaches ROC-AUC {metrics['roc_auc']:.2f} and {metrics['lift_at_top_decile']:.1f}x lift in the top decile.",
                "evidence": {"metrics": report["metrics"], "calibration": report["calibration"],
                             "horizon_days": report["horizon_days"], "matured_customers": report["matured_customers"]},
                "caveats": report["limitations"]}

    def _data_quality(self, params: dict) -> dict:
        nonzero = self.quality.loc[self.quality["count"].gt(0)]
        return {"answer": f"{len(nonzero)} data-quality checks have nonzero counts; these stay visible rather than being hidden.",
                "evidence": {"checks": nonzero.to_dict("records"), "source": self.metadata.get("source")},
                "caveats": ["Some flags (like missing categories) are informational and do not block analysis."]}

    # ---- routing --------------------------------------------------------
    def _plan(self, question: str) -> dict:
        lowered = question.lower()
        for topic, message in REFUSALS.items():
            if topic in lowered:
                return {"planner": "policy", "tool": None, "params": {}, "refusal": message}
        llm = self._llm_plan(question)
        if llm:
            return llm
        return self._keyword_plan(question)

    def _keyword_plan(self, question: str) -> dict:
        lowered = question.lower()
        # Longer, more specific phrases outweigh short generic words so intent wins ties.
        scores = {name: sum(len(keyword) + (4 if " " in keyword else 0) for keyword in tool["keywords"] if keyword in lowered)
                  for name, tool in self.tools.items()}
        best = max(scores, key=scores.get)
        if scores[best] == 0:
            return {"planner": "keyword", "tool": None, "params": {},
                    "refusal": "I could not map that question to an approved analysis. Try sales, category decline, sellers and late deliveries, reviews, forecasts, recommendations, repurchase or data quality."}
        params = {}
        words = question.split()
        for word in words:
            token = word.strip("?,.'\"")
            if token in self.recommender.index:
                params["product_id"] = token
        return {"planner": "keyword", "tool": best, "params": params}

    def _llm_plan(self, question: str) -> dict | None:
        """Optional: an LLM chooses a tool and parameters; it never writes SQL or free-form answers."""
        catalogue = "\n".join(f"- {name}: {tool['description']}" for name, tool in self.tools.items())
        prompt = (f"Choose exactly one analysis tool for the question. Reply with JSON: "
                  f'{{"tool": "<name>", "params": {{}}}}.\nTools:\n{catalogue}\n\nQuestion: {question}')
        try:
            if os.environ.get("OPENAI_API_KEY"):
                response = requests.post("https://api.openai.com/v1/chat/completions",
                                         headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                                         json={"model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), "temperature": 0,
                                               "response_format": {"type": "json_object"},
                                               "messages": [{"role": "user", "content": prompt}]}, timeout=20)
                response.raise_for_status()
                choice = json.loads(response.json()["choices"][0]["message"]["content"])
            elif os.environ.get("ANTHROPIC_API_KEY"):
                response = requests.post("https://api.anthropic.com/v1/messages",
                                         headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"},
                                         json={"model": os.environ.get("ANTHROPIC_MODEL", "claude-3-5-haiku-latest"), "max_tokens": 200,
                                               "messages": [{"role": "user", "content": prompt}]}, timeout=20)
                response.raise_for_status()
                text = response.json()["content"][0]["text"]
                choice = json.loads(text[text.find("{"):text.rfind("}") + 1])
            else:
                return None
        except Exception:
            return None
        tool = choice.get("tool")
        if tool not in self.tools:  # reject anything outside the whitelist
            return {"planner": "llm", "tool": None, "params": {},
                    "refusal": "The model suggested an analysis that is not on the approved list, so I did not run it."}
        return {"planner": "llm", "tool": tool, "params": choice.get("params") or {}}

    def ask(self, question: str) -> dict:
        plan = self._plan(question)
        if plan.get("tool") is None:
            return {"question": question, "planner": plan["planner"], "tool": None, "answer": plan["refusal"],
                    "evidence": {}, "caveats": ["Only approved analyses can be run; no SQL is generated from the question."]}
        result = self.tools[plan["tool"]]["run"](plan["params"])
        return {"question": question, "planner": plan["planner"], "tool": plan["tool"], "params": plan["params"], **result}

    def suggestions(self) -> list[str]:
        return ["Which categories had the biggest sales decline?",
                "Which sellers have the highest late-delivery rates?",
                "Are late deliveries associated with worse reviews?",
                "Explain the forecast and its limitations.",
                "What sales can we expect next month?",
                "Can we calculate profit or optimize inventory?"]


def _brl(value: float) -> str:
    return f"R${value:,.0f}"
