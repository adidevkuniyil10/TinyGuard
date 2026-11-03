
import csv
import platform
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import psutil

TARGET = "1.1.1.1"
INTERVAL = 3
DATA_DIR = Path("data")
CSV_FILE = DATA_DIR / "connection_loss.csv"

FIELDS = [
    "timestamp",
    "target",
    "ping_success",
    "latency_ms",
    "cpu_percent",
    "memory_percent",
    "bytes_sent",
    "bytes_received",
    "upload_bytes_per_sec",
    "download_bytes_per_sec",
]


def ping_target(target):
    if platform.system() == "Windows":
        command = ["ping", "-n", "1", "-w", "1500", target]
    else:
        command = ["ping", "-c", "1", target]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=3
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return 0, None

    match = re.search(
        r"time\s*[=<]\s*([\d.]+)\s*ms",
        result.stdout,
        re.IGNORECASE
    )

    if result.returncode == 0 and match:
        latency = float(match.group(1))
        if "<" in match.group(0):
            latency = max(0.1, latency / 2)
        return 1, latency

    return 0, None


def collect_sample(previous_net, previous_time):
    cpu = psutil.cpu_percent(interval=0.2)
    memory = psutil.virtual_memory().percent

    network = psutil.net_io_counters()
    current_time = time.monotonic()

    if previous_net is None:
        upload_rate = 0
        download_rate = 0
    else:
        elapsed = max(current_time - previous_time, 0.001)

        upload_rate = max(
            0, network.bytes_sent - previous_net.bytes_sent
        ) / elapsed

        download_rate = max(
            0, network.bytes_recv - previous_net.bytes_recv
        ) / elapsed

    success, latency = ping_target(TARGET)

    sample = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "target": TARGET,
        "ping_success": success,
        "latency_ms": "" if latency is None else round(latency, 2),
        "cpu_percent": round(cpu, 2),
        "memory_percent": round(memory, 2),
        "bytes_sent": network.bytes_sent,
        "bytes_received": network.bytes_recv,
        "upload_bytes_per_sec": round(upload_rate, 2),
        "download_bytes_per_sec": round(download_rate, 2),
    }

    return sample, network, current_time


def main():
    DATA_DIR.mkdir(exist_ok=True)

    file_exists = CSV_FILE.exists() and CSV_FILE.stat().st_size > 0

    previous_net = None
    previous_time = None
    count = 0

    print("====================================")
    print("       TINYGUARD TELEMETRY")
    print("====================================")
    print("Target:", TARGET)
    print("Sampling interval:", INTERVAL, "seconds")
    print("Saving to:", CSV_FILE.resolve())
    print("Press CTRL+C to stop.\n")

    try:
        with CSV_FILE.open(
            "a", newline="", encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(file, fieldnames=FIELDS)

            if not file_exists:
                writer.writeheader()

            while True:
                start = time.monotonic()

                sample, previous_net, previous_time = collect_sample(
                    previous_net, previous_time
                )

                writer.writerow(sample)
                file.flush()

                count += 1

                if sample["ping_success"]:
                    status = f'{sample["latency_ms"]} ms'
                else:
                    status = "NO REPLY"

                print(
                    f'Sample {count:04d} | '
                    f'Ping: {status:>10} | '
                    f'CPU: {sample["cpu_percent"]:5.1f}% | '
                    f'RAM: {sample["memory_percent"]:5.1f}% | '
                    f'Upload: {sample["upload_bytes_per_sec"]:10.1f} B/s | '
                    f'Download: {sample["download_bytes_per_sec"]:10.1f} B/s'
                )

                elapsed = time.monotonic() - start
                time.sleep(max(0, INTERVAL - elapsed))

    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

    print("Samples collected:", count)
    print("CSV saved at:", CSV_FILE.resolve())


if __name__ == "__main__":
    main()
