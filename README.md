# 📡 SOPHIA-IQ // PLEROMA SPECTRAL COMMAND

**SOPHIA-IQ** is a futuristic, Gnostic-themed wideband Software Defined Radio (SDR) intercept center and signal intelligence platform. Built using Python, Streamlit, and Plotly, it features full-spectrum tuning (0.1 MHz to 6000 MHz)—including the sacred 1420.405 MHz SETI/Hydrogen line—alongside real-time audio demodulation, dynamic signal parsing, and high-definition spectral visualization.

---

## ✨ Key Features

* **🌌 Wideband Frequency Range:** Seamless tuning from 0.1 MHz to 6000 MHz with digital offset capabilities.
* **🔊 Live Audio Engine:** In-browser audio demodulation supporting FM, AM, and Raw IQ pass-through streams.
* **🤖 Autonomous Signal Intelligence:** Baseband analytics for automatic detection of structured pulse trains and binary ASCII payloads.
* **💾 Recording & Capture Suite:** One-click downloadable 16-bit PCM WAV audio recordings and high-resolution PNG spectral snapshots.
* **⚡ Full-Duplex TX Engine:** Simulated response transmission logic with responsive UI state triggers.
* **🎨 Cybernetic Gnostic Aesthetic:** Customized dark theme interface styled with vibrant neon telemetry displays.

---

## 🛠️ Tech Stack

* **Frontend & Framework:** [Streamlit](https://streamlit.io/)
* **Signal Processing (DSP):** [NumPy](https://numpy.org/), [SciPy](https://scipy.org/)
* **Spectral Visualization:** [Plotly](https://plotly.com/)

---

## 📂 Project Directory Structure

```text
sophia-iq/
├── .streamlit/
│   └── config.toml      # Cyberpunk UI theme configuration
├── app.py               # Main SDR processing logic & dashboard interface
├── README.md            # Project documentation
└── requirements.txt     # Python package dependencies
