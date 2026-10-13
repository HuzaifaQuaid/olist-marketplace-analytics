from pathlib import Path
import argparse
import json

from marketplace.analytics import generate_report
from marketplace.demo import generate_demo
from marketplace.forecasting import train_forecast
from marketplace.model import train_model
from marketplace.pipeline import build_database
from marketplace.recommender import train_recommender
from marketplace.repurchase import train_repurchase


def _json(payload):
    print(json.dumps(payload, indent=2, default=str))


def build_all(db: Path, artifacts: Path, raw_dir: Path | None, source: str) -> None:
    if raw_dir is not None:
        _json(build_database(raw_dir, db, source=source))
    _json(train_model(db, artifacts))
    _json(train_forecast(db, artifacts))
    _json(train_recommender(db, artifacts))
    _json(train_repurchase(db, artifacts))
    generate_report(db, artifacts / "findings.md")
    print("wrote findings.md")


def main():
    parser = argparse.ArgumentParser(description="Olist marketplace analytics")
    parser.add_argument("command", choices=["demo", "ingest", "train", "forecast", "recommend", "repurchase", "report", "build"])
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--artifacts", type=Path, default=None)
    args = parser.parse_args()
    # The synthetic demo writes to its own database and artifacts so it can never overwrite real ingested data.
    db = args.db or (Path("data/demo.duckdb") if args.command == "demo" else Path("data/olist.duckdb"))
    artifacts = args.artifacts or (Path("artifacts/demo") if args.command == "demo" else Path("artifacts"))
    if args.command == "demo":
        demo = Path("data/demo")
        generate_demo(demo)
        build_all(db, artifacts, demo, "synthetic")
        print(f"\nDemo database written to {db} with artifacts in {artifacts}.")
        print("Start the app with: streamlit run app.py")
    elif args.command == "ingest":
        _json(build_database(args.raw_dir, db))
    elif args.command == "train":
        _json(train_model(db, artifacts))
    elif args.command == "forecast":
        _json(train_forecast(db, artifacts))
    elif args.command == "recommend":
        _json(train_recommender(db, artifacts))
    elif args.command == "repurchase":
        _json(train_repurchase(db, artifacts))
    elif args.command == "build":
        build_all(db, artifacts, None, "existing")
    else:
        print(generate_report(db, artifacts / "findings.md"))


if __name__ == "__main__":
    main()
