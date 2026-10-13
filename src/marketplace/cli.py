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
    parser.add_argument("--db", type=Path, default=Path("data/olist.duckdb"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    if args.command == "demo":
        demo = Path("data/demo")
        generate_demo(demo)
        build_all(args.db, args.artifacts, demo, "synthetic")
    elif args.command == "ingest":
        _json(build_database(args.raw_dir, args.db))
    elif args.command == "train":
        _json(train_model(args.db, args.artifacts))
    elif args.command == "forecast":
        _json(train_forecast(args.db, args.artifacts))
    elif args.command == "recommend":
        _json(train_recommender(args.db, args.artifacts))
    elif args.command == "repurchase":
        _json(train_repurchase(args.db, args.artifacts))
    elif args.command == "build":
        build_all(args.db, args.artifacts, None, "existing")
    else:
        print(generate_report(args.db, args.artifacts / "findings.md"))


if __name__ == "__main__":
    main()
