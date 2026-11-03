
import pandas as pd
from pathlib import Path

folder = Path("data/experiments")

# Load the three original datasets
normal = pd.read_csv(folder / "normal_baseline.csv")
high = pd.read_csv(folder / "high_latency.csv")
loss = pd.read_csv(folder / "connection_loss.csv")

# Assign class labels
normal["label"] = "Normal"
high["label"] = "High Latency"
loss["label"] = "Connection Loss"

# Remove transitional/low-latency samples from high-latency class
high = high[high["latency_ms"] >= 200].copy()

# Combine datasets
data = pd.concat(
    [normal, high, loss],
    ignore_index=True
)

# Keep useful numerical model features
features = [
    "ping_success",
    "latency_ms",
    "cpu_percent",
    "memory_percent",
    "upload_bytes_per_sec",
    "download_bytes_per_sec"
]

# Convert ping success to numerical values
data["ping_success"] = (
    data["ping_success"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0
    })
)

# Save prepared dataset
output = folder / "tinyguard_dataset.csv"
data[features + ["label"]].to_csv(output, index=False)

print("Dataset prepared successfully!")
print("\nClass distribution:")
print(data["label"].value_counts())

print("\nDataset shape:", data[features + ["label"]].shape)
print("\nSaved to:", output)
