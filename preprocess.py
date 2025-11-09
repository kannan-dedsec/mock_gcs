import pandas as pd
import numpy as np
from pathlib import Path

data_dir = Path("data")
files = list(data_dir.glob("*.csv"))

all_data = []

for f in files:
    stock_name = f.name.split("__")[0]
    print(f"📂 Processing {f.name} for stock {stock_name}...")

    # Load CSV safely
    df = pd.read_csv(f)

    if "timestamp" not in df.columns:
        print(f"⚠️  Skipping {f.name} — no 'timestamp' column found.")
        continue

    # Clean and parse timestamps robustly
    df["timestamp"] = (
        df["timestamp"]
        .astype(str)
        .str.replace(r"(\d{2}):(\d{2}):(\d{2}):(\d{2})", r"\1:\2:\3", regex=True)
        .str.replace(r"(\d{2}:\d{2}):(\d{2}):", r"\1:\2", regex=True)
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", format="mixed")

    bad_rows = df["timestamp"].isna().sum()
    if bad_rows > 0:
        print(f"⚠️  Dropping {bad_rows} bad timestamp rows in {f.name}")
        df = df.dropna(subset=["timestamp"])

    if df.empty:
        print(f"⚠️  Skipping {f.name} — all rows invalid after cleanup.")
        continue

    df = df.sort_values("timestamp")

    # ✅ Ensure timestamp uniqueness before resampling
    if df["timestamp"].duplicated().any():
        dup_count = df["timestamp"].duplicated().sum()
        print(f"⚠️  Found {dup_count} duplicate timestamps in {f.name}, aggregating...")
        df = df.groupby("timestamp", as_index=False).last()

    # Fill missing timestamps (1 min frequency)
    df = df.set_index("timestamp").resample("1min").nearest().reset_index()

    # Handle any missing close values
    df["close"] = df["close"].ffill().bfill()

    # Compute 10-minute moving average
    df["ma_10"] = df["close"].rolling(10, min_periods=1).mean()

    # Target: 1 if next 5-min close > current close
    df["future_close"] = df["close"].shift(-5)
    df["target"] = (df["future_close"] > df["close"]).astype(int)
    df.drop(columns=["future_close"], inplace=True)

    # Add stock name and id placeholder
    df["stock_symbol"] = stock_name
    all_data.append(df)

# Combine all stocks
if not all_data:
    raise ValueError("❌ No valid CSVs found in the data directory.")

combined = pd.concat(all_data, ignore_index=True)

# Assign numeric stock_id
stock_map = {s: i for i, s in enumerate(combined["stock_symbol"].unique())}
combined["stock_id"] = combined["stock_symbol"].map(stock_map)

# Save final output
Path("processed").mkdir(exist_ok=True)
combined.to_parquet("processed/stocks.parquet")

print(f"✅ Processed data saved to processed/stocks.parquet with {len(combined)} total rows")
