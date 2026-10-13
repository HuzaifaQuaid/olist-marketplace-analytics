from pathlib import Path
import argparse
import json

from marketplace.analytics import generate_report
from marketplace.demo import generate_demo
from marketplace.model import train_model
from marketplace.pipeline import build_database


def main():
    parser = argparse.ArgumentParser(description="Olist marketplace analytics")
    parser.add_argument("command", choices=["demo", "ingest", "train", "report"])
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--db", type=Path, default=Path("data/olist.duckdb"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    if args.command == "demo":
        demo = Path("data/demo")
        generate_demo(demo)
        print(json.dumps(build_database(demo, args.db, source="synthetic"), indent=2))
        print(json.dumps(train_model(args.db, args.artifacts), indent=2))
        generate_report(args.db, args.artifacts / "findings.md")
    elif args.command == "ingest":
        print(json.dumps(build_database(args.raw_dir, args.db), indent=2))
    elif args.command == "train":
        print(json.dumps(train_model(args.db, args.artifacts), indent=2))
    else:
        print(generate_report(args.db, args.artifacts / "findings.md"))


if __name__ == "__main__":
    main()
