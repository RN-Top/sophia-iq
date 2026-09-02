# 📡 SOPHIA-IQ // PLEROMA SPECTRAL COMMAND

**SOPHIA-IQ** is a Software-Defined Radio (SDR) and Signal Intelligence (SIGINT) dashboard designed for wideband spectrum monitoring, automated signal detection, and neural audio transcription powered by OpenAI's Whisper model.

---

## ✨ Features

* **Wideband Spectrum Visualizer:** Real-time Plotly spectral analyzer capable of tuning across 0.1 MHz to 6000 MHz (including 1420 MHz SETI/Hydrogen line frequencies).
* **Live Audio Intercept Engine:** Supports live FM, AM, and Raw I/Q baseband audio demodulation.
* **Neural Speech-to-Text (Whisper AI):** Automatically processes incoming voice signals, detects intelligibility, and renders real-time speech transcripts.
* **Capture & Recording Suite:** Download raw `.wav` audio captures and high-resolution PNG spectral snapshots directly from the UI.
* **Interactive Dynamic State UI:** Cyberpunk-inspired state engine that dynamically shifts glowing UI states based on active RF power and Signal-to-Noise Ratio (SNR).

---

## 📁 Repository Structure

```text
sophia-iq/
├── .streamlit/
│   └── config.toml      # Streamlit UI configuration
├── app.py               # Main application code (UI, DSP & Whisper AI)
├── requirements.txt     # Python dependencies
├── packages.txt         # Linux system package dependencies (FFmpeg)
└── README.md            # Documentation
