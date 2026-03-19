"""Generate executive_summary.pdf from structured content using fpdf2."""

from fpdf import FPDF
from pathlib import Path

OUTPUT = Path(__file__).parent / "executive_summary.pdf"


class SummaryPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 8, "AutoTrader - Executive Summary", align="L")
            self.ln(12)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(30, 30, 80)
        self.ln(4)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(30, 30, 80)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def bullet(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        x = self.get_x()
        self.cell(6, 5.5, "- ")
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def bold_bullet(self, label, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        x = self.get_x()
        self.cell(6, 5.5, "- ")
        self.set_font("Helvetica", "B", 10)
        self.write(5.5, f"{label}: ")
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def table_row(self, cols, widths, bold=False):
        style = "B" if bold else ""
        self.set_font("Helvetica", style, 9)
        h = 6
        for i, col in enumerate(cols):
            self.cell(widths[i], h, col, border=1)
        self.ln(h)


def build():
    pdf = SummaryPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ── Title page header ────────────────────────────────────────────
    pdf.ln(15)
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(30, 30, 80)
    pdf.cell(0, 12, "AutoTrader", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, "AI-Powered Daily Trading System", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, "Executive Summary - March 2026", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, "Bojana Belincevic, David Carrillo, Sebastiao Clemente, Bassem El Halawani, Theo Henry, Ocke Moulijn", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)

    # ── 1. Introduction ──────────────────────────────────────────────
    pdf.section_title("1. Introduction")
    pdf.body_text(
        "AutoTrader is an automated daily trading system that predicts next-day stock price "
        "movements for five major US companies using machine learning. The system combines a "
        "data analytics module (ETL pipeline and ML model) with an interactive web application "
        "built in Streamlit, allowing users to view real-time predictions, analyze model "
        "performance, and backtest trading strategies."
    )
    pdf.body_text(
        "The five companies tracked are Apple (AAPL), Microsoft (MSFT), Alphabet (GOOG), "
        "Amazon (AMZN), and NVIDIA (NVDA), representing a cross-section of the US technology "
        "and consumer sectors."
    )

    # ── 2. Data Sources ──────────────────────────────────────────────
    pdf.section_title("2. Data Sources")
    pdf.body_text(
        "All financial data is sourced from SimFin (simfin.com), a platform providing free "
        "access to fundamental and market data for publicly traded companies."
    )
    pdf.bold_bullet("Share Prices (Mandatory)",
        "Daily OHLCV (Open, High, Low, Close, Volume) data covering approximately five years "
        "of trading history. This is the primary dataset used to define the ML model's features "
        "and target variable.")
    pdf.bold_bullet("Company Information (Mandatory)",
        "Metadata including ticker symbol, company name, industry sector, and other identifiers.")
    pdf.bold_bullet("Financial Statements (Optional)",
        "Quarterly income statements, balance sheets, and cash flow statements are accessible "
        "via the API wrapper but not used as model features in the current version.")
    pdf.body_text(
        "For the offline training phase, data was obtained via SimFin's bulk download. For the "
        "online web application, data is fetched in real time through a custom Python API wrapper."
    )

    # ── 3. ETL Pipeline ──────────────────────────────────────────────
    pdf.section_title("3. ETL Pipeline")
    pdf.body_text(
        "The ETL (Extract, Transform, Load) pipeline converts raw share-price data into a "
        "feature-rich dataset suitable for machine learning. It is implemented in Python using "
        "Pandas and follows a 10-step process:"
    )
    steps = [
        "Clean and validate raw price data (standardize columns, handle missing values, remove duplicates)",
        "Compute daily returns: 1-day, 5-day, and 10-day percentage changes",
        "Compute rolling volatility: 10-day and 20-day standard deviation of returns",
        "Compute moving averages: Simple (5, 10, 20, 50-day) and Exponential (12, 26-day), normalized",
        "Compute RSI-14: Relative Strength Index, a momentum oscillator",
        "Compute MACD: Moving Average Convergence Divergence with signal line",
        "Compute Bollinger Bands: upper/lower bands and bandwidth (20-period, 2 std deviations)",
        "Compute volume features: 10-day volume moving average and volume ratio",
        "Compute ATR-14: Average True Range, a volatility measure normalized by price",
        "Generate target: binary label (1 = next day UP, 0 = next day DOWN)",
    ]
    for i, step in enumerate(steps, 1):
        pdf.bullet(f"Step {i}: {step}")
    pdf.ln(2)
    pdf.body_text(
        "The same ETL pipeline is used in both the offline training script and the live web "
        "application, ensuring consistency between training and inference."
    )

    # ── 4. Machine Learning Model ────────────────────────────────────
    pdf.section_title("4. Machine Learning Model")
    pdf.body_text(
        "The prediction task is binary classification: given today's market data and technical "
        "indicators, predict whether the stock price will go UP (1) or DOWN (0) the next "
        "trading day."
    )
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Model Architecture:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.bullet("Algorithm: Logistic Regression with balanced class weights")
    pdf.bullet("Preprocessing: StandardScaler (feature normalization)")
    pdf.bullet("Features: 21 technical indicators + 4 one-hot encoded ticker dummies (25 total)")
    pdf.bullet("Regularization: C=10.0, selected from grid search over [0.01, 0.1, 1.0, 10.0]")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Training Methodology:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.bullet("All five tickers combined into a single dataset (5,950 samples after ETL)")
    pdf.bullet("80/20 temporal train-test split preserving chronological order (no data leakage)")
    pdf.bullet("Balanced class weights address the imbalance between UP and DOWN trading days")
    pdf.ln(2)
    pdf.body_text(
        "The model achieves approximately 52% test accuracy, which is expected for daily stock "
        "price prediction. The system is designed to demonstrate a complete ML pipeline rather "
        "than achieve high predictive accuracy."
    )

    # ── 5. Web Application ───────────────────────────────────────────
    pdf.section_title("5. Web Application Architecture")
    pdf.body_text(
        "The web application is built with Streamlit and structured as a multi-page app:"
    )
    widths = [35, 145]
    pdf.table_row(["Page", "Purpose"], widths, bold=True)
    pdf.table_row(["Home", "System overview, tracked companies, team profiles, tech stack"], widths)
    pdf.table_row(["Go Live", "Real-time data via API, ETL on fresh data, ML predictions and trading signals"], widths)
    pdf.table_row(["Model Insights", "Feature importance, confusion matrix, accuracy metrics, distributions"], widths)
    pdf.table_row(["Backtesting", "Strategy simulator: Buy-and-Hold, Buy-and-Sell, and Benchmark comparison"], widths)
    pdf.ln(4)
    pdf.body_text(
        "The PySimFin API Wrapper is a custom object-oriented Python class that handles all "
        "communication with the SimFin Data API. It includes rate limiting (max 2 requests per "
        "second), custom exception classes for authentication and rate-limit errors, and "
        "automatic response format conversion to Pandas DataFrames."
    )
    pdf.body_text(
        "When no API key is provided, the application generates realistic synthetic data using "
        "geometric Brownian motion, allowing full functionality in demo mode."
    )

    # ── 6. Trading Strategies ────────────────────────────────────────
    pdf.section_title("6. Trading Strategies (Bonus)")
    pdf.body_text("Two ML-driven trading strategies are implemented alongside a benchmark:")
    pdf.bold_bullet("Buy-and-Hold",
        "Scans the backtest for UP-prediction days, spreads capital across up to 12 "
        "bullish entry points, and never sells during the backtest.")
    pdf.bold_bullet("Buy-and-Sell",
        "Active trading: buys when model confidence for UP exceeds 50%, sells on bearish "
        "signals, after 3 days maximum hold, or at a 2% stop-loss.")
    pdf.bold_bullet("Benchmark",
        "Buys once on day one and holds the full period with no ML input, used as a baseline.")
    pdf.body_text(
        "Performance metrics include total return, annualized return, Sharpe ratio, maximum "
        "drawdown, win rate, and total trades."
    )

    # ── 7. Challenges & Conclusions ──────────────────────────────────
    pdf.section_title("7. Challenges and Conclusions")
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Challenges:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.bold_bullet("Class imbalance",
        "The initial model exhibited bias toward predicting a single class. Adding balanced "
        "class weights resolved this, producing meaningful predictions for both UP and DOWN.")
    pdf.bold_bullet("Feature stationarity",
        "Raw price levels are non-stationary, so all price-based features were normalized by "
        "current price to make them comparable across tickers and time periods.")
    pdf.bold_bullet("API rate limits",
        "SimFin's free tier allows only 2 requests per second, requiring careful rate limiting "
        "in the API wrapper.")
    pdf.bold_bullet("Team coordination",
        "Managing concurrent development across ETL, ML, API, and frontend components required "
        "clear interfaces and consistent use of Git branching.")
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Conclusions:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.body_text(
        "AutoTrader demonstrates a complete end-to-end machine learning trading system: from "
        "raw data ingestion through feature engineering, model training, real-time prediction, "
        "and interactive visualization. While daily stock prediction accuracy is inherently "
        "limited, the system showcases clean Python engineering, proper ML methodology (temporal "
        "splits, no data leakage), and a functional, user-friendly web interface. The modular "
        "architecture allows each component to be improved independently."
    )
    pdf.body_text(
        "The application is deployed on Streamlit Cloud and publicly accessible without local "
        "installation."
    )

    # ── Save ─────────────────────────────────────────────────────────
    pdf.output(str(OUTPUT))
    print(f"PDF saved to {OUTPUT} ({OUTPUT.stat().st_size:,} bytes, {pdf.page_no()} pages)")


if __name__ == "__main__":
    build()
