# 📡 SOPHIA-IQ // PLEROMA SPECTRAL COMMAND

**SOPHIA-IQ** is a Software-Defined Radio (SDR) and Signal Intelligence (SIGINT) dashboard designed for wideband spectrum monitoring, live microphone/signal capture, and neural audio transcription powered by OpenAI's Whisper model.

---

## ✨ Features

* **Wideband Spectrum Visualizer:** Real-time Plotly spectral analyzer capable of tuning across 0.1 MHz to 6000 MHz (including 1420 MHz SETI/Hydrogen line frequencies).
* **Live Audio Capture Engine:** Directly records live audio via browser microphone input or demodulates baseband streams.
* **Neural Speech-to-Text (Whisper AI):** Automatically transcribes captured voice and signals into real-time text output using OpenAI Whisper.
* **Capture & Recording Suite:** Download captured `.wav` audio files directly to your device and save high-resolution PNG image snapshots of the spectral graph.
* **Flicker-Free UI Architecture:** Built with isolated Streamlit fragments (`@st.fragment`) to ensure smooth visualizer updates without interrupting audio recording or downloads.

---

## 📁 Repository Structure

```text
sophia-iq/
├── .streamlit/
│   └── config.toml      # Streamlit UI configuration
├── app.py               # Core application code (UI, DSP, & Whisper AI)
├── requirements.txt     # Python dependencies
├── packages.txt         # System dependencies (FFmpeg)
└── README.md            # Documentation
