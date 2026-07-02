import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from utils import load_and_align_waveform, extract_acoustic_features

# Global Graph Boundaries (Matching 16kHz audio processed over a 10s frame)
INPUT_SHAPE = (128, 313, 1)  
NUM_CLASSES = 3  # [0: Green/Normal, 1: Yellow/Mild Abnormality, 2: Red/High TB Risk] 
BATCH_SIZE = 8
EPOCHS = 15

def build_compact_edge_cnn():
    """
    Compiles a highly compact convolutional neural network engineered 
    to minimize memory footprint constraints on the ESP32-S3 .
    """
    model = models.Sequential([
        layers.Input(shape=INPUT_SHAPE),
        
        # First Feature Extraction Blocks
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Second Feature Extraction Blocks
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Dropout regularization to prevent overfitting on limited clinical cohorts
        layers.Dropout(0.25),
        
        # Flatten and Dense Softmax Head for Triage Routing
        layers.Flatten(),
        layers.Dense(16, activation='relu'),
        layers.Dense(NUM_CLASSES, activation='softmax') 
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

if __name__ == "__main__":
    print("[INIT] Booting CoughSense Edge Training Pipeline... - main.py:45")
    
    # Generate mock pipeline data arrays to test workspace compilation
    # Replace this block with your actual clinical dataset loader arrays
    print("[DATA] Generating validation sample array schemas... - main.py:49")
    X_train_mock = np.random.randn(60, *INPUT_SHAPE).astype(np.float32)
    y_train_mock = np.random.randint(0, NUM_CLASSES, size=(60,))
    
    # Compile Network
    coughsense_net = build_compact_edge_cnn()
    coughsense_net.summary()
    
    # Train Model
    print(f"\n[TRAINING] Fitting network graph layers across {EPOCHS} epochs... - main.py:58")
    coughsense_net.fit(X_train_mock, y_train_mock, epochs=2, batch_size=BATCH_SIZE) # Mocked run duration
    
    # Strict INT8 Quantization Process for TinyML Edge Deployment 
    print("\n[QUANTIZATION] Initializing PostTraining INT8 Quantization Engine... - main.py:62")
    converter = tf.lite.TFLiteConverter.from_keras_model(coughsense_net)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Representative dataset generator required to calibrate floating-point weights to fixed integers
    def calibration_data_generator():
        for item in tf.data.Dataset.from_tensor_slices(X_train_mock).batch(1).take(10):
            yield [item]
            
    converter.representative_dataset = calibration_data_generator
    
    # Enforce full fixed-point operations to remove floating hardware dependencies 
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    print("[CONVERT] Encoding model to optimized .tflite flatbuffer structure... - main.py:78")
    quantized_tflite_model = converter.convert()
    
    # Write optimized flatbuffer to disk array
    output_filename = "cough_sense_model_quant.tflite"
    with open(output_filename, "wb") as f:
        f.write(quantized_tflite_model)
        
    print(f"[SUCCESS] Ultracompact model asset generated at: ./{output_filename} - main.py:86")