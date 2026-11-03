import tensorflow as tf
from pathlib import Path

model_path = Path("models/tinyguard_int8.tflite")

if not model_path.exists():
    raise FileNotFoundError(f"Model not found: {model_path}")

interpreter = tf.lite.Interpreter(model_path=str(model_path))
interpreter.allocate_tensors()

print("\nTINYGUARD MODEL VERIFICATION")
print("----------------------------")

print("Model size:", model_path.stat().st_size, "bytes")

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\nInput:")
print("Shape:", input_details[0]["shape"])
print("Data type:", input_details[0]["dtype"])
print("Quantization:", input_details[0]["quantization"])

print("\nOutput:")
print("Shape:", output_details[0]["shape"])
print("Data type:", output_details[0]["dtype"])
print("Quantization:", output_details[0]["quantization"])

print("\nOperators:")
ops = interpreter._get_ops_details()
for op in ops:
    print(op["op_name"])

print("\nVerification complete.")