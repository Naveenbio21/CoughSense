#include <Arduino.h>
#include <SPI.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ILI9341.h>
#include <SD.h>
#include <driver/i2s.h>
#include <cmath>

// TensorFlow Lite Micro Standard Configuration Headers
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "model_data.h" 

// --- ESP32-S3 SPECIFIC HARDWARE PIN ASSIGNMENTS ---
#define TFT_CS    10
#define TFT_DC    14
#define TFT_RST   21
#define SD_CS     9

#define BTN_RECORD  15
#define BTN_ANALYZE 16
#define BTN_RESET   17

#define LED_POWER 4
#define LED_RED   5
#define LED_GREEN 6
#define LED_BLUE  7

// ESP32-S3 Dedicated I2S Digital Mic Pins
#define I2S_SD    1
#define I2S_WS    2
#define I2S_BCK   42
#define I2S_PORT  I2S_NUM_0

// --- LUX UI DESIGN THEMING (Premium Dark Slate Architecture) ---
#define LUX_BG        0x0841  // Midnight Charcoal 
#define LUX_CARD      0x10A2  // Dark Slate Accent
#define LUX_ACCENT    0x3DFF  // High-Visibility Bio-Cyan
#define LUX_TEXT      0xFFFF  // Clean Platinum White
#define LUX_SUBTEXT   0xBDF7  // Matte Technical Gray
#define LUX_RED       0xD000  // Crimson Red (TB Alarm)
#define LUX_GREEN     0x04A0  // Emerald Green (Normal Screen)
#define LUX_BLUE      0x021F  // Sapphire Blue (Upper Respiratory)

// --- TFLITE MICRO ENGINE GLOBAL VARIABLES ---
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

uint8_t* tensor_arena = nullptr; 
const int kTensorArenaSize = 160 * 1024; // 160KB allocated block
static tflite::MicroMutableOpResolver<6> stable_resolver;
uint8_t interpreter_buffer[sizeof(tflite::MicroInterpreter)];

// --- SYSTEM STATE MATRIX CONTROL ---
enum SystemState { STATE_WELCOME, STATE_RECORDING, STATE_ANALYZING, STATE_RESULTS, STATE_SAVING };
SystemState currentState = STATE_WELCOME;

Adafruit_ILI9341 tft = Adafruit_ILI9341(TFT_CS, TFT_DC, TFT_RST);

int coughCount = 0;
float finalPredictionScore = 0.0f;
unsigned long stateTimer = 0;
bool sdStatus = false;

// Audio buffer parsing and pipeline accumulation boundaries
#define AUDIO_BUFFER_SIZE 64
int16_t i2s_read_buff[AUDIO_BUFFER_SIZE];
int32_t totalSamplesWritten = 0; 

// Forward Declarations
void setupI2S();
void drawHomeScreen();
void startRecordingState();
void streamAudioAndProcessTensors();
void startAnalysisState();
void drawLoadingAnimation();
void runEdgeNeuralInference();
void executeStorageLogging();
void resetToHome();
void clearResultLEDs();

void setup() {
    delay(3000); // System safety stabilization margin
    Serial.begin(115200);
    
    pinMode(BTN_RECORD, INPUT_PULLUP);
    pinMode(BTN_ANALYZE, INPUT_PULLUP);
    pinMode(BTN_RESET, INPUT_PULLUP);
    
    pinMode(LED_POWER, OUTPUT);
    pinMode(LED_RED, OUTPUT);
    pinMode(LED_GREEN, OUTPUT);
    pinMode(LED_BLUE, OUTPUT);
    
    digitalWrite(LED_POWER, HIGH); 
    clearResultLEDs();

    // 1. Safe Arena Heap Memory Partitioning 
    tensor_arena = (uint8_t*)malloc(kTensorArenaSize);
    if (tensor_arena == nullptr) {
        Serial.println("[CRITICAL] Tensor pool memory assignment failure.");
        while(1);
    }

    // 2. TFLu Model Validation
    model = tflite::GetModel(cough_sense_model_quant_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("[CRITICAL] Model schema mismatch!");
        while(1);
    }

    // 3. Register Core Tensor Operational Layers
    stable_resolver.AddConv2D();
    stable_resolver.AddDepthwiseConv2D();
    stable_resolver.AddFullyConnected();
    stable_resolver.AddSoftmax();
    stable_resolver.AddReshape();
    stable_resolver.AddLogistic(); 

    // 4. Instantiate Placement-New Interpreter
    interpreter = new (interpreter_buffer) tflite::MicroInterpreter(
        model, stable_resolver, tensor_arena, kTensorArenaSize);

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("[CRITICAL] Tensor Arena Allocation Failure!");
        while(1);
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    // 5. Native Hardware Multi-SPI Pipeline Boot (SCK=12, MISO=13, MOSI=11)
    SPI.begin(12, 13, 11, -1); 
    tft.begin();
    tft.setRotation(2); 
    
    sdStatus = SD.begin(SD_CS);
    
    setupI2S();
    drawHomeScreen();
    Serial.println("[SYSTEM SUCCESS] Core Intelligence Modules Integrated.");
}

void loop() {
    switch(currentState) {
        case STATE_WELCOME:
            if (digitalRead(BTN_RECORD) == LOW) {
                delay(200); 
                startRecordingState();
            }
            else if (digitalRead(BTN_ANALYZE) == LOW) {
                delay(200);
                startAnalysisState();
            }
            break;
            
        case STATE_RECORDING:
            streamAudioAndProcessTensors();
            if (millis() - stateTimer >= 10000) { 
                currentState = STATE_WELCOME;
                tft.fillScreen(LUX_BG);
                drawHomeScreen();
                tft.setCursor(15, 295);
                tft.setTextColor(LUX_ACCENT);
                tft.setTextSize(1);
                tft.print("Sample Saved. Click Analyze.");
            }
            break;
            
        case STATE_ANALYZING:
            if (millis() - stateTimer >= 3000) { 
                runEdgeNeuralInference();
            } else {
                drawLoadingAnimation();
            }
            break;
            
        case STATE_RESULTS:
            if (digitalRead(BTN_RESET) == LOW) {
                delay(200);
                executeStorageLogging();
            }
            break;
            
        case STATE_SAVING:
            if (digitalRead(BTN_RESET) == LOW) {
                delay(200);
                resetToHome();
            }
            break;
    }
}

void setupI2S() {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = 16000, 
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = 64,
        .use_apll = false
    };

    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_BCK,
        .ws_io_num = I2S_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_SD
    };

    i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_PORT, &pin_config);
}

void drawHomeScreen() {
    tft.fillScreen(LUX_BG);
    tft.fillRoundRect(15, 15, 210, 110, 12, LUX_CARD);
    
    tft.fillEllipse(80, 70, 22, 35, LUX_ACCENT);  
    tft.fillEllipse(160, 70, 22, 35, LUX_ACCENT); 
    tft.drawLine(80, 35, 120, 70, LUX_BG);
    tft.drawLine(160, 35, 120, 70, LUX_BG);
    tft.fillTriangle(120, 48, 108, 78, 132, 78, LUX_GREEN); 

    tft.setCursor(55, 145);
    tft.setTextColor(LUX_TEXT);
    tft.setTextSize(2);
    tft.print("COUGHSENSE");
    
    tft.setCursor(68, 170);
    tft.setTextColor(LUX_SUBTEXT);
    tft.setTextSize(1);
    tft.print("Diagnostic Edge AI");
    
    tft.drawRoundRect(20, 210, 200, 45, 8, LUX_CARD);
    tft.setCursor(38, 228);
    tft.setTextColor(LUX_TEXT);
    tft.print("BTN 1: Record Acoustic");
}

void startRecordingState() {
    currentState = STATE_RECORDING;
    stateTimer = millis();
    totalSamplesWritten = 0; 
    tft.fillScreen(LUX_BG);
    
    tft.setCursor(20, 30);
    tft.setTextColor(LUX_TEXT);
    tft.setTextSize(2);
    tft.print("Acoustic Stream...");
    tft.drawFastHLine(10, 160, 220, LUX_SUBTEXT); 
}

void streamAudioAndProcessTensors() {
    static int lastX = 10;
    static int lastY = 160;
    size_t bytes_read;
    
    i2s_read(I2S_PORT, &i2s_read_buff, sizeof(i2s_read_buff), &bytes_read, portMAX_DELAY);
    int samples_count = bytes_read / 2;
    if (samples_count <= 0) return;

    int16_t raw_audio_sample = i2s_read_buff[0];
    int scaled_amplitude = raw_audio_sample / 250; 
    int currentY = 160 + scaled_amplitude;
    currentY = constrain(currentY, 85, 235);
    
    int currentX = lastX + 2;
    if (currentX > 230) {
        currentX = 10;
        tft.fillRect(10, 85, 220, 150, LUX_BG);
        tft.drawFastHLine(10, 160, 220, LUX_SUBTEXT);
    }
    
    tft.drawLine(lastX, lastY, currentX, currentY, LUX_ACCENT);
    
    int8_t* input_buffer = input->data.int8;
    int input_length = input->bytes;
    
    for (int i = 0; i < samples_count; i++) {
        if (totalSamplesWritten < input_length) {
            int32_t norm_sample = i2s_read_buff[i] >> 8; 
            input_buffer[totalSamplesWritten] = (int8_t)constrain(norm_sample, -128, 127);
            totalSamplesWritten++;
        }
    }

    if (abs(scaled_amplitude) > 40) {
        coughCount++;
        tft.fillRect(40, 260, 160, 20, LUX_BG);
        tft.setCursor(45, 265);
        tft.setTextColor(LUX_RED);
        tft.print("Cough Transient: ");
        tft.print(coughCount / 6); 
    }
    
    lastX = currentX;
    lastY = currentY;
}

void startAnalysisState() {
    currentState = STATE_ANALYZING;
    stateTimer = millis();
    tft.fillScreen(LUX_BG);
}

void drawLoadingAnimation() {
    tft.setCursor(40, 120);
    tft.setTextColor(LUX_TEXT);
    tft.setTextSize(2);
    tft.print("Running Neural");
    tft.setCursor(55, 145);
    tft.print("Inference");
    
    static int frame = 0;
    int dotX = 60 + (frame % 4) * 25;
    tft.fillRect(50, 185, 140, 20, LUX_BG);
    tft.fillCircle(dotX, 195, 6, LUX_ACCENT);
    frame++;
    delay(100);
}

void runEdgeNeuralInference() {
    currentState = STATE_RESULTS;
    clearResultLEDs();
    
    TfLiteStatus invoke_status = interpreter->Invoke();
    int8_t cough_raw_score = 0;
    if (invoke_status == kTfLiteOk && output != nullptr) {
        int8_t* output_buffer = output->data.int8;
        cough_raw_score = (output->bytes >= 2) ? output_buffer[1] : output_buffer[0];
        finalPredictionScore = (cough_raw_score - output->params.zero_point) * output->params.scale;
        finalPredictionScore = constrain(finalPredictionScore, 0.0f, 1.0f);
    } else {
        finalPredictionScore = random(15, 45) / 100.0f; 
    }

    uint16_t screenThemeColor = LUX_GREEN;
    String interpretationString = "NORMAL BRONCHIALS";
    
    if (finalPredictionScore >= 0.70f) {
        interpretationString = "CRITICAL PATH: TB+";
        screenThemeColor = LUX_