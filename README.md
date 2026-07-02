# Coughsense

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/notebooks/gpu.ipynb)

Coughsense is a portable, edge AI-powered acoustic diagnostic scanner engineered for real-time respiratory health screening and early Tuberculosis (TB) triage in resource-constrained environments.

## 💡 Key Technical Features
* **Offline-First Processing Profile:** Runs localized inference using an 8-bit quantized TinyML network deployed straight onto an ESP32-S3 microcontroller—operating entirely independent of internet access or external cloud ecosystems.
* **Low-Noise Biomedical Audio Capture:** Collects acoustic biomarkers via a digital MEMS microphone sampled at 16 kHz, compressing runtime inputs into uniform structural matrix windows.
* **Instant Point-Of-Care Triage:** Segregates anomalous cough sequences (Normal Bronchials vs Upper Respiratory vs Active High-Risk TB Vectors) via visual screen themes and an integrated on-board LED routing network in less than 10 seconds.
* **Field Durability Tuning:** Uses a customized 3D-printed acoustic isolation enclosure to block heavy clinical background noise before routing signals to the model buffer.

## 🛠️ System Tech Stack
* **Hardware:** ESP32-S3 Dual-Core SoC, ILI9341 Portrait Display Panel, Digital MEMS Mic Array, Dedicated SD Card Logging Bridge.
* **Software Layer:** TensorFlow Lite Micro Framework, Arduino C++ Core, PlatformIO Build Pipeline, Librosa Processing Engine.
