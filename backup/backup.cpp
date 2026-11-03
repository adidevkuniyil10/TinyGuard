
#include <Arduino.h>
#include <math.h>
#include "tinyguard_model.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

constexpr int kTensorArenaSize = 16 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

const char* class_names[] = {
    "Normal",
    "High Latency",
    "Connection Loss"
};

const float feature_mean[6] = {
    0.891089109f, 115.805281f, 8.92904291f,
    70.0561061f, 4494.891f, 54130.8258f
};

const float feature_scale[6] = {
    0.311527381f, 169.060259f, 8.78318682f,
    2.99443137f, 16741.6377f, 137581.183f
};

tflite::MicroMutableOpResolver<2> resolver;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

bool runPrediction(const char* test_name, const float raw[6]) {
    Serial.println();
    Serial.print("Testing: ");
    Serial.println(test_name);

    // Standardize and quantize the six input features.
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

    // Run TinyGuard inference.
    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("Inference failed!");
        return false;
    }

    // Read and dequantize the three output scores.
    int best_index = 0;
    float best_score = -1.0f;

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

    Serial.print("PREDICTION: ");
    Serial.println(class_names[best_index]);

    return true;
}

void setup() {
    Serial.begin(115200);
    delay(1000);

    Serial.println("\n=== TinyGuard ESP32 ===");

    const tflite::Model* model =
        tflite::GetModel(tinyguard_model);

    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Schema version mismatch!");
        return;
    }

    if (resolver.AddFullyConnected() != kTfLiteOk ||
        resolver.AddSoftmax() != kTfLiteOk) {
        Serial.println("Operator registration failed!");
        return;
    }

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize
    );

    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("Tensor allocation failed!");
        return;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    if (input == nullptr || output == nullptr ||
        input->type != kTfLiteInt8 ||
        output->type != kTfLiteInt8 ||
        input->dims->data[1] != 6 ||
        output->dims->data[1] != 3) {
        Serial.println("Unexpected model tensor format!");
        return;
    }

    Serial.println("TinyGuard model initialized!");
    Serial.print("Tensor arena used: ");
    Serial.println(interpreter->arena_used_bytes());

    // Representative test samples in original feature order:
    // ping_success, latency, CPU, memory, upload, download.

    const float normal[6] = {
        1, 43, 5.8f, 68.4f, 357.67f, 34539.21f
    };

    const float high_latency[6] = {
        1, 502, 5.8f, 68.4f, 357.67f, 34539.21f
    };

    const float connection_loss[6] = {
        0, 43, 5.8f, 68.4f, 357.67f, 34539.21f
    };

    Serial.println("\n=== TinyGuard Prediction Tests ===");

    runPrediction("Normal network", normal);
    runPrediction("High latency", high_latency);
    runPrediction("Connection loss", connection_loss);

    Serial.println("\n=== Tests completed ===");
}

void loop() {
    delay(10000);
}
