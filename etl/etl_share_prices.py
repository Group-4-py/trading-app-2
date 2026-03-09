# etl_share_prices.py

from pathlib import Path
import argparse
from etl_utils import run_etl_for_ticker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument(
        "--input",
        default="data/raw/us-shareprices-daily.csv"
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_path = output_dir / f"{args.ticker}.parquet"

    run_etl_for_ticker(
        ticker=args.ticker,
        share_prices_path=input_path,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
