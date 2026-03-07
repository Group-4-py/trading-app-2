"""
PySimFin – Python API Wrapper for SimFin (Requirement 2.1)

Object-oriented wrapper that simplifies interaction with the SimFin Data API.
Handles authentication, rate-limiting, and data conversion to Pandas DataFrames.

Usage:
    client = PySimFin(api_key="your-api-key")
    df = client.get_share_prices("AAPL", start="2023-01-01", end="2024-01-01")
"""

import time
import logging
import pandas as pd
import requests

from utils.config import SIMFIN_BASE_URL, SIMFIN_RATE_LIMIT

logger = logging.getLogger(__name__)


class SimFinAPIError(Exception):
    """Custom exception for SimFin API errors."""
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"SimFin API Error {status_code}: {message}")


class SimFinRateLimitError(SimFinAPIError):
    """Raised when the API rate limit is exceeded."""
    pass


class SimFinAuthError(SimFinAPIError):
    """Raised when authentication fails."""
    pass


class PySimFin:
    """
    Python wrapper for the SimFin Data API.

    Attributes:
        api_key (str): SimFin API key for authentication.
        base_url (str): Base URL for the SimFin API.
    """

    def __init__(self, api_key: str):
        """
        Initialize the PySimFin client.

        Args:
            api_key: Your SimFin API key.

        Raises:
            ValueError: If api_key is empty or None.
        """
        if not api_key:
            raise ValueError("API key must be provided. Get one at https://www.simfin.com/")

        self.api_key = api_key
        self.base_url = SIMFIN_BASE_URL
        self._last_request_time = 0.0
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"api-key {self.api_key}",
            "Accept": "application/json",
        })
        logger.info("PySimFin client initialized.")

    # ── Private Helpers ───────────────────────────────────────────────

    def _rate_limit(self):
        """Enforce rate limiting: max 2 requests per second (free tier)."""
        elapsed = time.time() - self._last_request_time
        if elapsed < SIMFIN_RATE_LIMIT:
            sleep_time = SIMFIN_RATE_LIMIT - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self._last_request_time = time.time()

    def _request(self, endpoint: str, params: dict = None) -> dict:
        """
        Make a GET request to the SimFin API with rate limiting and error handling.

        Args:
            endpoint: API endpoint path (e.g., '/companies/prices/compact').
            params: Query parameters.

        Returns:
            Parsed JSON response as a dictionary.

        Raises:
            SimFinAuthError: If authentication fails (401).
            SimFinRateLimitError: If rate limit is exceeded (429).
            SimFinAPIError: For other API errors.
        """
        self._rate_limit()

        url = f"{self.base_url}{endpoint}"
        logger.debug(f"GET {url} params={params}")

        try:
            response = self._session.get(url, params=params, timeout=30)
        except requests.exceptions.ConnectionError as e:
            raise SimFinAPIError(0, f"Connection error: {e}")
        except requests.exceptions.Timeout:
            raise SimFinAPIError(0, "Request timed out after 30 seconds.")
        except requests.exceptions.RequestException as e:
            raise SimFinAPIError(0, f"Request failed: {e}")

        # Handle HTTP errors
        if response.status_code == 401:
            raise SimFinAuthError(401, "Invalid API key. Check your credentials.")
        elif response.status_code == 429:
            raise SimFinRateLimitError(429, "Rate limit exceeded. Slow down requests.")
        elif response.status_code != 200:
            raise SimFinAPIError(
                response.status_code,
                f"Unexpected response: {response.text[:500]}"
            )

        return response.json()

    def _compact_to_dataframe(self, data: dict) -> pd.DataFrame:
        """
        Convert SimFin compact-format response to a Pandas DataFrame.

        The compact format returns:
            {"columns": [...], "data": [[...], [...]]}

        Args:
            data: Parsed JSON response with 'columns' and 'data' keys.

        Returns:
            pd.DataFrame with proper column names.
        """
        if not data or "columns" not in data or "data" not in data:
            logger.warning("Empty or malformed response from API.")
            return pd.DataFrame()

        df = pd.DataFrame(data["data"], columns=data["columns"])
        return df

    # ── Public Methods ────────────────────────────────────────────────

    def get_share_prices(
        self,
        ticker: str,
        start: str = None,
        end: str = None,
    ) -> pd.DataFrame:
        """
        Retrieve daily share prices for a given ticker.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL').
            start: Start date in 'YYYY-MM-DD' format. Defaults to 1 year ago.
            end: End date in 'YYYY-MM-DD' format. Defaults to today.

        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Adj. Close,
            Volume, Dividend, Shares Outstanding.

        Raises:
            ValueError: If ticker is empty.
            SimFinAPIError: On API errors.

        Example:
            >>> client = PySimFin("my-api-key")
            >>> df = client.get_share_prices("AAPL", "2024-01-01", "2024-06-01")
            >>> df.head()
        """
        if not ticker:
            raise ValueError("Ticker symbol must be provided.")

        ticker = ticker.upper().strip()

        params = {"ticker": ticker}
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        logger.info(f"Fetching share prices for {ticker} ({start} to {end})")

        data = self._request("/companies/prices/compact", params=params)

        # Handle list response (SimFin returns a list of objects)
        if isinstance(data, list) and len(data) > 0:
            data = data[0]

        df = self._compact_to_dataframe(data)

        if not df.empty and "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").reset_index(drop=True)

        return df

    def get_financial_statement(
        self,
        ticker: str,
        statement: str = "pl",
        start: str = None,
        end: str = None,
        period: str = "quarterly",
    ) -> pd.DataFrame:
        """
        Retrieve financial statements for a given ticker.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL').
            statement: Statement type: 'pl' (income), 'bs' (balance sheet),
                       'cf' (cash flow). Defaults to 'pl'.
            start: Start date in 'YYYY-MM-DD' format.
            end: End date in 'YYYY-MM-DD' format.
            period: 'quarterly' or 'annual'. Defaults to 'quarterly'.

        Returns:
            DataFrame with financial statement data.

        Raises:
            ValueError: If ticker is empty or statement type is invalid.
            SimFinAPIError: On API errors.
        """
        if not ticker:
            raise ValueError("Ticker symbol must be provided.")

        valid_statements = {"pl", "bs", "cf"}
        if statement not in valid_statements:
            raise ValueError(
                f"Invalid statement type '{statement}'. Must be one of: {valid_statements}"
            )

        ticker = ticker.upper().strip()

        params = {
            "ticker": ticker,
            "period": period,
        }
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        endpoint = f"/companies/statements/compact?statement={statement}"
        logger.info(f"Fetching {statement} statement for {ticker}")

        data = self._request(endpoint, params=params)

        if isinstance(data, list) and len(data) > 0:
            data = data[0]

        df = self._compact_to_dataframe(data)
        return df

    def get_company_info(self, ticker: str) -> dict:
        """
        Retrieve company general information.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Dictionary with company info (name, industry, employees, etc.)
        """
        if not ticker:
            raise ValueError("Ticker symbol must be provided.")

        ticker = ticker.upper().strip()
        params = {"ticker": ticker}

        data = self._request("/companies/general/compact", params=params)

        if isinstance(data, list) and len(data) > 0:
            data = data[0]

        df = self._compact_to_dataframe(data)

        if not df.empty:
            return df.iloc[0].to_dict()
        return {}

    def close(self):
        """Close the underlying HTTP session."""
        self._session.close()
        logger.info("PySimFin session closed.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __repr__(self):
        return f"PySimFin(base_url='{self.base_url}')"
