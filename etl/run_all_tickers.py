from pathlib import Path

from etl_utils import TICKER_LIST, load_share_prices, run_etl_for_ticker_from_df


def main():
    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / "data" / "raw" / "us-shareprices-daily.csv"
    output_dir = project_root / "data" / "processed"
    prices = load_share_prices(input_path, TICKER_LIST)

    for ticker in TICKER_LIST:
        output_path = output_dir / f"{ticker}.parquet"
        print(f"Running ETL for {ticker} ...")
        run_etl_for_ticker_from_df(
            ticker=ticker,
            prices=prices,
            output_path=output_path,
        )
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
