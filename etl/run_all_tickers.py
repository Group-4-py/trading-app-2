# etl/run_all_tickers.py

from pathlib import Path
from etl_utils import run_etl_for_ticker

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]


def main():
    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / "data" / "raw" / "us-shareprices-daily.csv"
    output_dir = project_root / "data" / "processed"

    for ticker in TICKERS:
        output_path = output_dir / f"{ticker}.parquet"
        print(f"Running ETL for {ticker} ...")
        run_etl_for_ticker(
            ticker=ticker,
            share_prices_path=input_path,
            output_path=output_path,
        )
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
