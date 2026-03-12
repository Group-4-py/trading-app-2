# etl/etl_share_prices.py

from pathlib import Path
import argparse
from etl_utils import run_etl_for_ticker


def main():
    # Set up argument parser to handle command-line inputs
    parser = argparse.ArgumentParser()
    
    # Define required argument for the stock ticker symbol
    parser.add_argument("--ticker", required=True)
    
    # Define optional argument for the input CSV file path with a default value
    parser.add_argument(
        "--input",
        default="data/raw/us-shareprices-daily.csv"
    )
    
    # Define optional argument for the output directory with a default value
    parser.add_argument(
        "--output-dir",
        default="data/processed"
    )
    
    # Parse the provided command-line arguments
    args = parser.parse_args()

    # Convert string paths to Path objects for easier manipulation
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    
    # Construct the full output path for the resulting parquet file
    output_path = output_dir / f"{args.ticker}.parquet"

    # Execute the ETL process for the specified ticker
    run_etl_for_ticker(
        ticker=args.ticker,
        share_prices_path=input_path,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
