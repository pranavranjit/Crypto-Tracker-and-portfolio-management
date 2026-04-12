import yfinance as yf
import pandas as pd
from pathlib import Path
import sys

# List of mutual funds / ETFs (mix of popular ETFs and well-known mutual funds)
TICKERS = [
    "VOO", "VTI", "SPY", "QQQ", "IWM", "BND", "BNDX", "VEMAX", "VXUS",
    "VFIAX", "VTSAX", "FXAIX", "VIG", "VIGAX", "SCHZ"
]
START = "2010-01-01"

OUT_DIR = Path(__file__).resolve().parents[1] / "week5_funds"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "stage_1_fund_data.csv"

print(f"Saving consolidated data to: {OUT_FILE}")

rows = []
for t in TICKERS:
    try:
        print(f"Downloading {t} ...")
        tk = yf.Ticker(t)
        df = tk.history(start=START, auto_adjust=True, actions=False)
        if df is None or df.empty:
            print(f"  Warning: no data for {t}")
            continue
        df = df.reset_index()
        # normalize column names
        df = df.rename(columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Dividends": "dividends",
            "Stock Splits": "stock_splits",
            "Adj Close": "adj_close",
        })
        df["ticker"] = t
        rows.append(df)
    except Exception as e:
        print(f"  Error downloading {t}: {e}")

if not rows:
    print("No data downloaded. Exiting with code 2.")
    sys.exit(2)

out_df = pd.concat(rows, ignore_index=True, sort=False)
# ensure date column is present and sorted
out_df["date"] = pd.to_datetime(out_df["date"], errors="coerce")
out_df = out_df.dropna(subset=["date"]).sort_values(["ticker", "date"]).reset_index(drop=True)

# save
out_df.to_csv(OUT_FILE, index=False)
print(f"Downloaded data rows: {len(out_df)}")
print("Done.")
