
import pandas as pd
from pathlib import Path

folder = Path("data/experiments")

files = {
    "Normal": "normal_baseline.csv",
    "High Latency": "high_latency.csv",
    "Connection Loss": "connection_loss.csv",
}

for label, filename in files.items():
    path = folder / filename
    df = pd.read_csv(path)

    print("\n" + "=" * 45)
    print("CLASS:", label)
    print("File:", filename)
    print("Samples:", len(df))
    print("Successful pings:", (df["ping_success"] == True).sum())
    print("Failed pings:", (df["ping_success"] == False).sum())
    print("Latency statistics:")
    print(df["latency_ms"].describe())
    print("Missing values:")
    print(df.isna().sum())
