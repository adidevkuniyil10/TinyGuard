
import json
import time

import paho.mqtt.client as mqtt

from collector import collect_sample, TARGET, INTERVAL

BROKER = "broker.hivemq.com"
PORT = 1883

INPUT_TOPIC = "tinyguard/test/24BCE2369"
OUTPUT_TOPIC = "tinyguard/test/24BCE2369/prediction"


def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print("Connected to HiveMQ!")
        client.subscribe(OUTPUT_TOPIC, qos=1)
        print("Listening for ESP32 predictions...")
    else:
        print("MQTT connection failed:", reason_code)


def on_message(client, userdata, message):
    try:
        result = json.loads(message.payload.decode())
        print("\n=== TINYGUARD ESP32 PREDICTION ===")
        print("Prediction:", result)
        print("=================================\n")
    except Exception:
        print("ESP32 response:", message.payload.decode(errors="replace"))


def main():
    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id="tinyguard_windows_bridge"
    )

    client.on_connect = on_connect
    client.on_message = on_message

    print("Connecting to HiveMQ...")
    client.connect(BROKER, PORT, 60)
    client.loop_start()

    previous_net = None
    previous_time = None

    print("\nTinyGuard live telemetry bridge started.")
    print("Target:", TARGET)
    print("Press CTRL+C to stop.\n")

    count = 0

    try:
        while True:
            start = time.monotonic()

            sample, previous_net, previous_time = collect_sample(
                previous_net, previous_time
            )

            features = [
                sample["ping_success"],
                sample["latency_ms"] if sample["latency_ms"] != "" else 0,
                sample["cpu_percent"],
                sample["memory_percent"],
                sample["upload_bytes_per_sec"],
                sample["download_bytes_per_sec"],
            ]

            payload = ",".join(str(value) for value in features)

            info = client.publish(
                INPUT_TOPIC,
                payload=payload,
                qos=1,
                retain=False
            )

            info.wait_for_publish(timeout=5)
            count += 1

            status = (
                f'{sample["latency_ms"]} ms'
                if sample["ping_success"]
                else "NO REPLY"
            )

            print(
                f"Sample {count:04d} | "
                f"Ping: {status} | "
                f"Published: {payload}"
            )

            elapsed = time.monotonic() - start
            time.sleep(max(0, INTERVAL - elapsed))

    except KeyboardInterrupt:
        print("\nStopping TinyGuard bridge...")

    finally:
        client.loop_stop()
        client.disconnect()
        print("Bridge stopped.")


if __name__ == "__main__":
    main()
