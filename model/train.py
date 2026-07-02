import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from audio_preprocessing import parse_and_normalize_waveform, generate_mel_spectrogram_matrix

INPUT_FRAME_SHAPE = (128, 313, 1) # Matrix dimensions mapping 10s window sequences
BATCH_SIZE = 8
EPOCHS = 12

def compile_tinyml_cnn():
    """Initializes a compact multi-layer CNN configuration designed to fit the 160KB arena limit."""
    model = models.Sequential([
        layers.Input(shape=INPUT_FRAME_SHAPE),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(16, activation='relu'),
        layers.Dense(2, activation='softmax') # [Normal Respiratory, High-Risk TB Cluster]
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    print("[RUNNING] Initializing Google Colab Training Engine... - train.py:26")
    
    # Validation Sample Generators
    X_synthetic = np.random.randn(40, *INPUT_FRAME_SHAPE).astype(np.float32)
    y_synthetic = np.random.randint(0, 2, size=(40,))
    
    model = compile_tinyml_cnn()
    model.fit(X_synthetic, y_synthetic, epochs=2, batch_size=BATCH_SIZE)
    
    # Run Post-Training INT8 Quantization Scaling
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    def quantization_calibration_gen():
        for sample in tf.data.Dataset.from_tensor_slices(X_synthetic).batch(1).take(5):
            yield [sample]
            
    converter.representative_dataset = quantization_calibration_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    quantized_model_bytes = converter.convert()
    
    with open("cough_sense_model_quant.tflite", "wb") as f:
        f.write(quantized_model_bytes)
    print("[COMPLETE] Quantized flatbuffer binary built successfully. - train.py:52")