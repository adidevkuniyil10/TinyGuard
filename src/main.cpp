
#include <Arduino.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "tinyguard_model.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Wi-Fi configuration
const char* ssid = "Wokwi-GUEST";
const char* password = "";

// MQTT configuration
const char* mqtt_server = "broker.hivemq.com";
const char* input_topic = "tinyguard/test/24BCE2369";
const char* output_topic = "tinyguard/test/24BCE2369/prediction";

WiFiClient espClient;
PubSubClient mqtt(espClient);

// TensorFlow Lite Micro configuration
constexpr int kTensorArenaSize = 16 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

const char* class_names[] = {
    "Normal",
    "High Latency",
    "Connection Loss"
};

// Training preprocessing values
const float feature_mean[6] = {
    0.891089109f,
    115.805281f,
    8.92904291f,
    70.0561061f,
    4494.891f,
    54130.8258f
};

const float feature_scale[6] = {
    0.311527381f,
    169.060259f,
    8.78318682f,
    2.99443137f,
    16741.6377f,
    137581.183f
};

tflite::MicroMutableOpResolver<2> resolver;
tflite::MicroInterpreter* interpreter = nullptr;

TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

unsigned long lastMqttAttempt = 0;
unsigned long lastWiFiAttempt = 0;

// Run TinyGuard inference
bool runPrediction(const float raw[6]) {

    for (int i = 0; i < 6; i++) {

        float standardized =
            (raw[i] - feature_mean[i]) / feature_scale[i];

        int32_t quantized =
            lroundf(standardized / input->params.scale)
            + input->params.zero_point;

        if (quantized < -128) quantized = -128;
        if (quantized > 127) quantized = 127;

        input->data.int8[i] = (int8_t)quantized;
    }

    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("Inference failed!");
        return false;
    }

    int best_index = 0;
    float best_score = -1.0f;

    Serial.println("\nPrediction scores:");

    for (int i = 0; i < 3; i++) {

        float score =
            (output->data.int8[i] - output->params.zero_point)
            * output->params.scale;

        Serial.print(class_names[i]);
        Serial.print(": ");
        Serial.print(score * 100.0f, 2);
        Serial.println("%");

        if (score > best_score) {
            best_score = score;
            best_index = i;
        }
    }

    Serial.print("\nTinyGuard prediction: ");
    Serial.println(class_names[best_index]);

    // Publish prediction to Python
    char response[160];

    snprintf(
        response,
        sizeof(response),
        "{\"prediction\":\"%s\",\"score\":%.4f}",
        class_names[best_index],
        best_score
    );

    if (mqtt.connected()) {

        bool sent = mqtt.publish(output_topic, response);

        Serial.println(
            sent ? "Prediction published!"
                 : "Prediction publish failed!"
        );
    }

    return true;
}

// MQTT telemetry callback
void callback(char* topic, byte* payload, unsigned int length) {

    Serial.println("\n=== Telemetry received ===");

    if (length >= 160) {
        Serial.println("Message too long!");
        return;
    }

    char message[160];

    memcpy(message, payload, length);
    message[length] = '\0';

    Serial.print("Received: ");
    Serial.println(message);

    float features[6];

    int parsed = sscanf(
        message,
        "%f,%f,%f,%f,%f,%f",
        &features[0],
        &features[1],
        &features[2],
        &features[3],
        &features[4],
        &features[5]
    );

    if (parsed != 6) {
        Serial.println("Invalid telemetry format!");
        return;
    }

    Serial.println("Running TinyGuard inference...");

    runPrediction(features);
}

// Initialize model
bool initializeModel() {

    const tflite::Model* model =
        tflite::GetModel(tinyguard_model);

    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model schema mismatch!");
        return false;
    }

    if (resolver.AddFullyConnected() != kTfLiteOk ||
        resolver.AddSoftmax() != kTfLiteOk) {

        Serial.println("Operator registration failed!");
        return false;
    }

    static tflite::MicroInterpreter static_interpreter(
        model,
        resolver,
        tensor_arena,
        kTensorArenaSize
    );

    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("Tensor allocation failed!");
        return false;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    if (input == nullptr || output == nullptr ||
        input->type != kTfLiteInt8 ||
        output->type != kTfLiteInt8 ||
        input->dims->data[1] != 6 ||
        output->dims->data[1] != 3) {

        Serial.println("Unexpected model tensor format!");
        return false;
    }

    Serial.println("TinyGuard model initialized!");

    Serial.print("Tensor arena used: ");
    Serial.println(interpreter->arena_used_bytes());

    return true;
}

// Connect to Wi-Fi
bool connectWiFi() {

    Serial.println("\nConnecting to Wi-Fi...");

    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false);
    WiFi.begin(ssid, password, 6);

    int attempts = 0;

    while (WiFi.status() != WL_CONNECTED && attempts < 30) {
        delay(500);
        Serial.print(".");
        attempts++;
    }

    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("\nWi-Fi connection failed!");
        return false;
    }

    Serial.println("\nWi-Fi connected!");
    Serial.println(WiFi.localIP());

    return true;
}

// Connect to MQTT
bool connectMQTT() {

    Serial.println("Connecting to HiveMQ...");

    if (mqtt.connect("TinyGuard_ESP32_Inference")) {

        Serial.println("MQTT connected!");

        if (mqtt.subscribe(input_topic)) {
            Serial.println("Subscribed to telemetry topic!");
        } else {
            Serial.println("MQTT subscription failed!");
        }

        return true;
    }

    Serial.print("MQTT connection failed. State: ");
    Serial.println(mqtt.state());

    return false;
}

void setup() {

    Serial.begin(115200);
    delay(1000);

    Serial.println("\n=== TinyGuard ESP32 ===");

    // Initialize ML model
    if (!initializeModel()) {
        Serial.println("Model initialization failed!");
        return;
    }

    // Connect Wi-Fi
    if (connectWiFi()) {

        mqtt.setServer(mqtt_server, 1883);
        mqtt.setCallback(callback);
        mqtt.setBufferSize(256);

        connectMQTT();
    }

    lastWiFiAttempt = millis();
    lastMqttAttempt = millis();
}

void loop() {

    // Recover Wi-Fi connection
    if (WiFi.status() != WL_CONNECTED) {

        if (millis() - lastWiFiAttempt >= 10000) {

            lastWiFiAttempt = millis();

            Serial.println("Wi-Fi disconnected. Reconnecting...");

            WiFi.mode(WIFI_STA);
            WiFi.begin(ssid, password, 6);
        }

        delay(10);
        return;
    }

    // Recover MQTT connection
    if (!mqtt.connected()) {

        if (millis() - lastMqttAttempt >= 3000) {

            lastMqttAttempt = millis();

            Serial.println("Reconnecting to MQTT...");

            connectMQTT();
        }
    }

    mqtt.loop();

    delay(10);
}
