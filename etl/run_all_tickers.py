# etl/run_all_tickers.py

from pathlib import Path
from tqdm import tqdm
from etl_utils import load_share_prices, run_etl_from_dataframe

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"]


def main():
    # 1. Define standard paths using the project root directory
    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / "data" / "raw" / "us-shareprices-daily.csv"
    output_dir = project_root / "data" / "processed"
    
    # 2. Ensure output directory exists before saving files to it
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Load the massive SimFin dataset into memory ONCE to save time
    print("Loading full SimFin dataset into memory (this may take a moment)...")
    prices_df = load_share_prices(input_path)
    print("Data loaded successfully!")

    # 4. Loop through each ticker and process it using the pre-loaded DataFrame
    for ticker in tqdm(TICKERS, desc="Processing Tickers"):
        output_path = output_dir / f"{ticker}.parquet"
        
        # We pass prices_df directly, avoiding thousands of redundant disk reads
        run_etl_from_dataframe(
            ticker=ticker,
            prices_df=prices_df,
            output_path=output_path,
        )


if __name__ == "__main__":
    main()
