import streamlit as st
import numpy as np
import scipy.signal as signal
import plotly.graph_objects as go
import io
import wave
import re
import tempfile
import whisper

# ==============================================================================
# [SECTION 1] PAGE CONFIGURATION & STYLING
# ==============================================================================

st.set_page_config(
    page_title="SOPHIA-IQ // PLEROMA SPECTRAL COMMAND", 
    layout="wide", 
    page_icon="📡",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp { background-color: #04060A; }
    .status-panel {
        text-align: center;
        padding: 24px;
        border-radius: 16px;
        background: #080C16;
    }
    </style>
""", unsafe_allow_html=True)

# Cache AI Model Load so it doesn't reload continuously
@st.cache_resource
def load_speech_model():
    return whisper.load_model("tiny")

whisper_model = load_speech_model()

# Header
st.title("📡 SOPHIA-IQ // PLEROMA SPECTRAL COMMAND")
st.caption("Gnostic Wideband Transceiver • Neural Signal Intercept & Audio Intelligence")


# ==============================================================================
# [SECTION 2] SIDEBAR CONTROLS & FREQUENCY TUNING
# ==============================================================================

st.sidebar.header("🎛️ RF FRONTEND CONTROLS")

target_freq_mhz = st.sidebar.number_input(
    "Tuned Center Frequency (MHz)", 
    value=1420.405, 
    min_value=0.1, 
    max_value=6000.0, 
    step=1.0, 
    format="%.3f"
)
translated_offset = st.sidebar.slider("Digital Translator Offset (kHz)", -500, 500, 150)
demod_mode = st.sidebar.selectbox("Live Audio Demodulation", ["FM (Frequency Modulation)", "AM (Amplitude Envelope)", "Raw IQ Pass-through"])
tx_enabled = st.sidebar.checkbox("Enable Full-Duplex TX Engine")
enable_ai_transcription = st.sidebar.checkbox("Enable Neural Audio Transcription (Whisper AI)", value=True)


# ==============================================================================
# [SECTION 3] SIGNAL PROCESSING & AI TRANSCRIPTION ENGINE
# ==============================================================================

def generate_baseband_iq(sample_rate=2.4e6, num_samples=16384):
    """Generates wideband complex I/Q samples."""
    t = np.arange(num_samples) / sample_rate
    noise = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) * 0.08
    carrier = 0.6 * np.exp(1j * 2 * np.pi * (translated_offset * 1e3) * t)
    audio_mod = 0.5 * (1 + np.sin(2 * np.pi * 440 * t))
    return noise + (carrier * audio_mod)

def parse_baseband_payload(iq_samples):
    """Analyzes baseband phase transitions for binary/ASCII payloads."""
    phases = np.angle(iq_samples[:128])
    bits = "".join(["1" if p > 0 else "0" for p in phases])
    
    try:
        bytes_list = [bits[i:i+8] for i in range(0, len(bits), 8)]
        decoded_text = "".join([chr(int(b, 2)) for b in bytes_list if len(b) == 8])
        if decoded_text.isprintable() and len(decoded_text.strip()) > 0:
            return "MATCHED", f"Decoded ASCII Payload: '{decoded_text}'"
    except Exception:
        pass
    
    pulses = [len(g) for g in re.findall(r'1+', bits)]
    if len(pulses) >= 3:
        return "MATCHED", f"Structured Pulse Train Detected: {pulses[:6]}"
        
    return "SEARCHING", "No coherent artificial payload parsed on baseband."

def demodulate_to_pcm(iq_samples, mode='FM', target_sample_rate=44100):
    """Demodulates baseband IQ data into 16-bit PCM WAV audio."""
    if "FM" in mode:
        audio = np.angle(iq_samples[1:] * np.conj(iq_samples[:-1]))
    elif "AM" in mode:
        audio = np.abs(iq_samples) - np.mean(np.abs(iq_samples))
    else:
        audio = np.real(iq_samples)

    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    audio_pcm = (audio * 32767).astype(np.int16)

    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wav_file:
        wav_file.setnchannels(1)      
        wav_file.setsampwidth(2)      
        wav_file.setframerate(target_sample_rate)
        wav_file.writeframes(audio_pcm.tobytes())
    
    return wav_io.getvalue()

def transcribe_audio_payload(audio_bytes):
    """Transcribes demodulated audio using OpenAI Whisper AI."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name

        result = whisper_model.transcribe(tmp_path)
        transcript = result.get("text", "").strip()
        return transcript if transcript else "No intelligible voice detected."
    except Exception as e:
        return f"Transcription error: {str(e)}"


# Compute DSP & Telemetry
iq_data = generate_baseband_iq()
fft_vals = np.abs(np.fft.fftshift(np.fft.fft(iq_data))) ** 2
power_dbm = 10 * np.log10(np.mean(fft_vals) + 1e-12)
peak_dbm = 10 * np.log10(np.max(fft_vals) + 1e-12)
snr = peak_dbm - power_dbm
audio_bytes = demodulate_to_pcm(iq_data, mode=demod_mode)
status_type, comms_payload = parse_baseband_payload(iq_data)

if snr > 12 or status_type == "MATCHED":
    avatar_state = "COMMUNICATION DETECTED"
    color = "#00FFCC"
    glow = "rgba(0, 255, 204, 0.5)"
elif tx_enabled:
    avatar_state = "TRANSMITTING RESPONSE"
    color = "#FF3366"
    glow = "rgba(255, 51, 102, 0.5)"
else:
    avatar_state = "WIDEBAND SCANNING"
    color = "#0099FF"
    glow = "rgba(0, 153, 255, 0.3)"


# ==============================================================================
# [SECTION 4] DASHBOARD UI LAYOUT & TELEMETRY
# ==============================================================================

col1, col2 = st.columns([1.1, 1.9])

with col1:
    st.subheader("🤖 SOPHIA AI State Engine")

    st.markdown(f"""
        <div class="status-panel" style="border: 2px solid {color}; box-shadow: 0 0 30px {glow};">
            <h2 style="color: {color}; margin: 0; font-family: monospace; letter-spacing: 2px;">{avatar_state}</h2>
            <hr style="border-color: {color}33; margin: 15px 0;">
            <p style="color: #A0AAB5; margin: 4px 0;">Power: <b style="color: #FFF;">{power_dbm:.2f} dBm</b></p>
            <p style="color: #A0AAB5; margin: 4px 0;">SNR: <b style="color: {color};">{snr:.2f} dB</b></p>
        </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    
    m1, m2 = st.columns(2)
    m1.metric("Local Tuner", f"{target_freq_mhz:.3f} MHz")
    m2.metric("Baseband Offset", f"{(target_freq_mhz + (translated_offset/1000)):.3f} MHz")

    # Live Audio Component
    st.subheader("🔊 Live Intercept Audio Stream")
    st.audio(audio_bytes, format="audio/wav")

    # Audio Download Suite
    st.subheader("💾 Recording & Capture Suite")
    st.download_button(
        label="💾 Save Intercepted WAV Audio File",
        data=audio_bytes,
        file_name=f"sophia_intercept_{(target_freq_mhz + (translated_offset/1000)):.3f}MHz.wav",
        mime="audio/wav",
        use_container_width=True
    )

    # Audio Intelligence & Transcript Section
    st.subheader("📝 Live AI Audio Transcript")
    if enable_ai_transcription:
        transcript_text = transcribe_audio_payload(audio_bytes)
        st.text_area("Whisper AI Speech-to-Text Log", value=transcript_text, height=100)
    else:
        st.caption("AI Audio Transcription disabled in sidebar.")

    # Signal Intelligence Output
    st.subheader("🌐 Baseband Parser Output")
    if status_type == "MATCHED":
        st.success(f"**ALERT:** Coherent signal burst locked!\n\n{comms_payload}")
    else:
        st.info("Monitoring wideband spectrum for structured modulation...")


# ==============================================================================
# [SECTION 5] SPECTRAL VISUALIZER (STABLE / ZERO FLICKER)
# ==============================================================================

with col2:
    st.subheader("📊 High-Definition Spectral Visualizer")
    
    freqs = np.linspace(target_freq_mhz - 1.2, target_freq_mhz + 1.2, len(fft_vals))
    fft_db = 10 * np.log10(fft_vals + 1e-12)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=freqs, 
        y=fft_db, 
        mode='lines', 
        name='RF Power',
        line=dict(color=color, width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=freqs, 
        y=fft_db, 
        fill='tozeroy',
        name='Noise Floor',
        fillcolor=glow,
        line=dict(color='rgba(0,0,0,0)', width=0),
        showlegend=False
    ))
    
    fig.update_layout(
        xaxis_title="Frequency Spectrum (MHz)",
        yaxis_title="Power Density (dB)",
        template="plotly_dark",
        plot_bgcolor="#04060A",
        paper_bgcolor="#04060A",
        margin=dict(l=20, r=20, t=20, b=20),
        height=520,
        xaxis=dict(gridcolor="#121826", zerolinecolor="#121826"),
        yaxis=dict(gridcolor="#121826", zerolinecolor="#121826")
    )
    
    st.plotly_chart(fig, use_container_width=True, config={
        'toImageButtonOptions': {
            'format': 'png',
            'filename': f'sophia_spectrum_{target_freq_mhz:.3f}MHz',
            'height': 720,
            'width': 1280,
            'scale': 2
        }
    })
