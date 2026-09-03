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
st.caption("Gnostic Wideband Transceiver • Real-Time Signal Intercept & Audio Intelligence Engine")


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
audio_duration = st.sidebar.slider("Capture / Record Buffer Duration (Seconds)", 1, 30, 5)
demod_mode = st.sidebar.selectbox("Live Audio Demodulation", ["FM (Frequency Modulation)", "AM (Amplitude Envelope)", "Raw IQ Pass-through"])
tx_enabled = st.sidebar.checkbox("Enable Full-Duplex TX Engine State")
enable_ai_transcription = st.sidebar.checkbox("Enable Neural Audio Transcription (Whisper AI)", value=True)


# ==============================================================================
# [SECTION 3] EXTENDED AUDIO & SIGNAL PROCESSING ENGINE
# ==============================================================================

def generate_baseband_iq(duration_sec=5, sample_rate=44100):
    """Generates complex I/Q samples over a user-selected time duration (seconds)."""
    num_samples = int(sample_rate * duration_sec)
    t = np.arange(num_samples) / sample_rate
    
    # Base RF Noise Floor
    noise = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) * 0.05
    
    # Complex Carrier and Low-Frequency Tone Modulation (Simulated Voice Band)
    carrier = 0.6 * np.exp(1j * 2 * np.pi * (translated_offset * 10) * t)
    audio_mod = 0.5 * (1 + np.sin(2 * np.pi * 440 * t) + 0.25 * np.sin(2 * np.pi * 880 * t))
    
    return noise + (carrier * audio_mod)

def parse_baseband_payload(iq_samples):
    """Analyzes baseband phase transitions for binary or structured pulse trains."""
    phases = np.angle(iq_samples[:128])
    bits = "".join(["1" if p > 0 else "0" for p in phases])
    
    pulses = [len(g) for g in re.findall(r'1+', bits)]
    if len(pulses) >= 3:
        return "MATCHED", f"Structured Transmission Burst Detected: Pulse pattern {pulses[:6]}"
        
    return "SEARCHING", "Wideband spectrum operational. No structured digital payload locked."

def demodulate_to_extended_wav(iq_samples, mode='FM', sample_rate=44100):
    """Demodulates long IQ buffers into full-length 16-bit PCM WAV audio."""
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
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_pcm.tobytes())
    
    return wav_io.getvalue()

def transcribe_audio_payload(audio_bytes):
    """Transcribes extended audio buffer using Whisper AI."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name

        result = whisper_model.transcribe(tmp_path)
        transcript = result.get("text", "").strip()
        return transcript if transcript else "No intelligible speech detected in capture."
    except Exception as e:
        return f"Transcription error: {str(e)}"


# ==============================================================================
# [SECTION 4] DASHBOARD UI LAYOUT & AUDIO CAPTURE CONTROLS
# ==============================================================================

col1, col2 = st.columns([1.1, 1.9])

with col1:
    st.subheader("🎙️ Long-Buffer Signal Capture & Demodulation")
    
    if st.button(f"⚡ Capture & Demodulate {audio_duration}s Intercept Sample", use_container_width=True):
        with st.spinner(f"Processing {audio_duration} seconds of baseband I/Q data..."):
            long_iq = generate_baseband_iq(duration_sec=audio_duration)
            st.session_state.captured_wav = demodulate_to_extended_wav(long_iq, mode=demod_mode)
            st.session_state.status_type, st.session_state.comms_payload = parse_baseband_payload(long_iq)

    if "captured_wav" in st.session_state:
        st.subheader("🔊 Live Intercept Audio Player")
        st.caption(f"Demodulated audio buffer duration: **{audio_duration} seconds**")
        st.audio(st.session_state.captured_wav, format="audio/wav")

        st.subheader("💾 Recording & Export Suite")
        st.download_button(
            label=f"💾 Save {audio_duration}s Intercepted WAV File",
            data=st.session_state.captured_wav,
            file_name=f"sophia_intercept_{target_freq_mhz:.3f}MHz_{audio_duration}s.wav",
            mime="audio/wav",
            use_container_width=True
        )

        st.subheader("📝 Live AI Speech Transcript")
        if enable_ai_transcription:
            with st.spinner("Whisper AI processing captured audio..."):
                transcript_text = transcribe_audio_payload(st.session_state.captured_wav)
                st.text_area("Whisper AI Speech-to-Text Log", value=transcript_text, height=100)
        else:
            st.caption("AI Audio Transcription disabled in sidebar.")

        st.subheader("🌐 Baseband Signal Intelligence")
        st.info(st.session_state.get("comms_payload", "Monitoring baseband..."))
    else:
        st.info("Click the capture button above to generate and process a long-duration audio sample.")


# ==============================================================================
# [SECTION 5] REAL-TIME SPECTRAL VISUALIZER (SMOOTH FRAGMENT UPDATE)
# ==============================================================================

@st.fragment(run_every=1.0)
def render_live_spectrum():
    # Rapid short-sample generation for smooth spectral visual updates
    realtime_iq = generate_baseband_iq(duration_sec=0.1)
    fft_vals = np.abs(np.fft.fftshift(np.fft.fft(realtime_iq))) ** 2
    
    power_dbm = 10 * np.log10(np.mean(fft_vals) + 1e-12)
    peak_dbm = 10 * np.log10(np.max(fft_vals) + 1e-12)
    snr = peak_dbm - power_dbm

    if snr > 12:
        avatar_state = "COMMUNICATION DETECTED"
        color = "#00FFCC"
        glow = "rgba(0, 255, 204, 0.4)"
    elif tx_enabled:
        avatar_state = "TRANSMITTING RESPONSE"
        color = "#FF3366"
        glow = "rgba(255, 51, 102, 0.4)"
    else:
        avatar_state = "WIDEBAND SCANNING"
        color = "#0099FF"
        glow = "rgba(0, 153, 255, 0.2)"

    with col2:
        st.subheader("🤖 SOPHIA Receiver State Engine")
        st.markdown(f"""
            <div class="status-panel" style="border: 2px solid {color}; box-shadow: 0 0 20px {glow};">
                <h3 style="color: {color}; margin: 0; font-family: monospace; letter-spacing: 1px;">{avatar_state}</h3>
                <p style="color: #A0AAB5; margin: 8px 0 0 0;">Tuned: <b>{target_freq_mhz:.3f} MHz</b> | Offset: <b>{translated_offset} kHz</b> | SNR: <b>{snr:.1f} dB</b></p>
            </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        st.subheader("📊 High-Definition Real-Time Spectral Visualizer")

        freqs = np.linspace(target_freq_mhz - 1.2, target_freq_mhz + 1.2, len(fft_vals))
        fft_db = 10 * np.log10(fft_vals + 1e-12)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=freqs, y=fft_db, mode='lines', name='RF Power', line=dict(color=color, width=2)))
        fig.add_trace(go.Scatter(x=freqs, y=fft_db, fill='tozeroy', fillcolor=glow, line=dict(color='rgba(0,0,0,0)', width=0), showlegend=False))
        
        fig.update_layout(
            xaxis_title="Frequency Spectrum (MHz)",
            yaxis_title="Power Density (dB)",
            template="plotly_dark",
            plot_bgcolor="#04060A",
            paper_bgcolor="#04060A",
            margin=dict(l=20, r=20, t=20, b=20),
            height=460,
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

# Execute isolated visualizer loop
render_live_spectrum()
