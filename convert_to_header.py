from pathlib import Path

model_path = Path("models/tinyguard_int8.tflite")
header_path = Path("models/tinyguard_model.h")

if not model_path.exists():
    raise FileNotFoundError(model_path)

model_bytes = model_path.read_bytes()

with header_path.open("w") as f:
    f.write("#pragma once\n")
    f.write("#include <cstdint>\n\n")
    f.write("alignas(16) const unsigned char tinyguard_model[] = {\n")

    for i, byte in enumerate(model_bytes):
        if i % 12 == 0:
            f.write("    ")

        f.write(f"0x{byte:02X}, ")

        if i % 12 == 11:
            f.write("\n")

    f.write("\n};\n")
    f.write(f"const unsigned int tinyguard_model_len = {len(model_bytes)};\n")

print("C header generated successfully!")
print("Header:", header_path)
print("Model bytes:", len(model_bytes))