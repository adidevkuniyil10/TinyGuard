
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------------
# 1. Configuration
# -------------------------------

DATA_FILE = Path("data/experiments/tinyguard_dataset.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = [
    "ping_success",
    "latency_ms",
    "cpu_percent",
    "memory_percent",
    "upload_bytes_per_sec",
    "download_bytes_per_sec"
]

CLASSES = [
    "Normal",
    "High Latency",
    "Connection Loss"
]

np.random.seed(42)
tf.random.set_seed(42)

# -------------------------------
# 2. Load dataset
# -------------------------------

df = pd.read_csv(DATA_FILE)

# Convert ping_success into 0 or 1
df["ping_success"] = (
    df["ping_success"]
    .astype(str)
    .str.lower()
    .map({"true": 1, "false": 0, "1": 1, "0": 0})
)

df["label"] = pd.Categorical(
    df["label"],
    categories=CLASSES
)

df = df.dropna(subset=["label", "ping_success"])

# Convert feature columns to numeric
for col in FEATURES:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# -------------------------------
# 3. Chronological holdout
# Keep the last 20% of each class
# for preliminary testing
# -------------------------------

train_parts = []
test_parts = []

for label in CLASSES:
    group = df[df["label"] == label].copy()

    # Preserve collection order
    split_index = int(len(group) * 0.8)

    train_parts.append(group.iloc[:split_index])
    test_parts.append(group.iloc[split_index:])

train_df = pd.concat(train_parts)
test_df = pd.concat(test_parts)

print("Training samples:", len(train_df))
print("Testing samples:", len(test_df))

print("\nTraining class counts:")
print(train_df["label"].value_counts())

# -------------------------------
# 4. Prepare features
# -------------------------------

X_train = train_df[FEATURES].to_numpy(dtype=np.float32)
X_test = test_df[FEATURES].to_numpy(dtype=np.float32)

# Replace missing values using training medians only
medians = np.nanmedian(X_train, axis=0)
medians = np.where(np.isfinite(medians), medians, 0.0)

X_train = np.where(np.isnan(X_train), medians, X_train)
X_test = np.where(np.isnan(X_test), medians, X_test)

# Standardize using training data only
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

# Convert labels into integer IDs
class_to_id = {name: i for i, name in enumerate(CLASSES)}

y_train = train_df["label"].astype(str).map(class_to_id).to_numpy()
y_test = test_df["label"].astype(str).map(class_to_id).to_numpy()

# Shuffle training samples
rng = np.random.default_rng(42)
order = rng.permutation(len(X_train))
X_train = X_train[order]
y_train = y_train[order]

# -------------------------------
# 5. Balance classes
# -------------------------------

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(CLASSES)),
    y=y_train
)

class_weights = {
    i: float(weight)
    for i, weight in enumerate(weights)
}

print("\nClass weights:", class_weights)

# -------------------------------
# 6. Build small neural network
# -------------------------------

model = Sequential([
    Input(shape=(len(FEATURES),)),
    Dense(16, activation="relu"),
    Dense(8, activation="relu"),
    Dense(len(CLASSES), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------------
# 7. Train model
# -------------------------------

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=12,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)

# -------------------------------
# 8. Evaluate on held-out data
# -------------------------------

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

predictions = np.argmax(
    model.predict(X_test, verbose=0),
    axis=1
)

print("\nTest accuracy:", round(accuracy * 100, 2), "%")
print("Test loss:", round(float(loss), 4))

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=np.arange(len(CLASSES)),
    target_names=CLASSES,
    zero_division=0
))

print("Confusion matrix:")
print(confusion_matrix(
    y_test,
    predictions,
    labels=np.arange(len(CLASSES))
))

# -------------------------------
# 9. Save model and preprocessing
# -------------------------------

model.save(MODEL_DIR / "tinyguard_model.keras")

np.savez(
    MODEL_DIR / "preprocessing.npz",
    medians=medians,
    scaler_mean=scaler.mean_,
    scaler_scale=scaler.scale_
)

with open(MODEL_DIR / "classes.json", "w") as f:
    json.dump({
        "features": FEATURES,
        "classes": CLASSES
    }, f, indent=4)

print("\nModel and preprocessing saved in:", MODEL_DIR)
