import json
import numpy as np
import pandas as pd
import tensorflow as tf

# Load dataset
df = pd.read_csv("data/experiments/tinyguard_dataset.csv")

features = [
    "ping_success",
    "latency_ms",
    "cpu_percent",
    "memory_percent",
    "upload_bytes_per_sec",
    "download_bytes_per_sec"
]

classes = ["Normal", "High Latency", "Connection Loss"]

# Load preprocessing values
prep = np.load("models/preprocessing.npz")

medians = prep["medians"]
mean = prep["scaler_mean"]
scale = prep["scaler_scale"]

# Convert data
X = df[features].copy()
X["ping_success"] = (
    X["ping_success"].astype(str).str.lower()
    .map({"true": 1, "false": 0})
    .fillna(X["ping_success"])
)

X = X.to_numpy(dtype=np.float32)
X = np.where(np.isnan(X), medians, X)
X = (X - mean) / scale

# Load INT8 model
interpreter = tf.lite.Interpreter(
    model_path="models/tinyguard_int8.tflite"
)
interpreter.allocate_tensors()

input_info = interpreter.get_input_details()[0]
output_info = interpreter.get_output_details()[0]

input_scale, input_zero = input_info["quantization"]
output_scale, output_zero = output_info["quantization"]

# Test a few examples from each class
print("\nTINYGUARD QUANTIZED PREDICTION TEST")
print("----------------------------------")

for label in range(3):
    indices = np.where(
        df["label"].to_numpy() == classes[label]
    )[0]

    print(f"\nActual class: {classes[label]}")

    for idx in indices[:3]:
        sample = X[idx:idx+1]

        # Quantize input
        q_input = np.round(
            sample / input_scale + input_zero
        )
        q_input = np.clip(q_input, -128, 127).astype(np.int8)

        interpreter.set_tensor(
            input_info["index"], q_input
        )
        interpreter.invoke()

        q_output = interpreter.get_tensor(
            output_info["index"]
        )

        # Dequantize output
        scores = (
            q_output.astype(np.float32) - output_zero
        ) * output_scale

        prediction = classes[int(np.argmax(scores))]

        print(
            f"Expected: {classes[label]:16s} "
            f"Predicted: {prediction:16s} "
            f"Scores: {np.round(scores[0], 3)}"
        )