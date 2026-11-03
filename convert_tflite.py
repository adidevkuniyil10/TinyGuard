
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / "data/experiments/tinyguard_dataset.csv"
MODEL = BASE / "models/tinyguard_model.keras"
PREP = BASE / "models/preprocessing.npz"
SAVED = BASE / "models/saved_model"
OUTPUT = BASE / "models/tinyguard_int8.tflite"

FEATURES = [
    "ping_success",
    "latency_ms",
    "cpu_percent",
    "memory_percent",
    "upload_bytes_per_sec",
    "download_bytes_per_sec"
]

CLASSES = ["Normal", "High Latency", "Connection Loss"]

# Load preprocessing parameters
p = np.load(PREP)
medians = p["medians"].astype(np.float32)
mean = p["scaler_mean"].astype(np.float32)
scale = p["scaler_scale"].astype(np.float32)

# Recreate training data for calibration
df = pd.read_csv(DATA)
parts = []

for name in CLASSES:
    group = df[df["label"] == name].copy()
    parts.append(group.iloc[:int(len(group) * 0.8)])

train = pd.concat(parts, ignore_index=True)

X = train[FEATURES].to_numpy(dtype=np.float32)
X = np.where(np.isnan(X), medians, X)
X = ((X - mean) / scale).astype(np.float32)

print("Calibration samples:", len(X))

# Load Keras model
model = tf.keras.models.load_model(MODEL)

# Export to SavedModel using Keras 3 API
if SAVED.exists():
    import shutil
    shutil.rmtree(SAVED)

model.export(str(SAVED))
print("SavedModel exported:", SAVED)

# Load SavedModel as a concrete function
loaded = tf.saved_model.load(str(SAVED))
infer = loaded.signatures["serve"]

# Convert SavedModel using a concrete function
converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [infer],
    loaded
)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_dataset():
    for sample in X:
        yield [sample.reshape(1, 6)]

converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite = converter.convert()

OUTPUT.write_bytes(tflite)
print("\nINT8 conversion successful!")
print("Model size:", len(tflite), "bytes")

# Inspect model input and output
interpreter = tf.lite.Interpreter(model_path=str(OUTPUT))
interpreter.allocate_tensors()

inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

print("Input dtype:", inp["dtype"])
print("Input shape:", inp["shape"])
print("Input quantization:", inp["quantization"])

print("Output dtype:", out["dtype"])
print("Output shape:", out["shape"])
print("Output quantization:", out["quantization"])

metadata = {
    "features": FEATURES,
    "classes": CLASSES,
    "medians": medians.tolist(),
    "scaler_mean": mean.tolist(),
    "scaler_scale": scale.tolist(),
    "input_scale": float(inp["quantization"][0]),
    "input_zero_point": int(inp["quantization"][1]),
    "output_scale": float(out["quantization"][0]),
    "output_zero_point": int(out["quantization"][1])
}

(BASE / "models/tinyguard_metadata.json").write_text(
    json.dumps(metadata, indent=4)
)

print("Metadata saved.")
