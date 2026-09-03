import streamlit as st
import numpy as np
import scipy.signal as signal
import plotly.graph_objects as go
import io
import wave
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
        padding: 20px;
        border-radius: 12px;
        background: #080C16;
    }
    </style>
""", unsafe_allow_html=True)

# Load AI Whisper model once and cache it to memory
@st.cache_resource
def load_speech_model():
    return whisper.load_model("tiny")

whisper_model = load_speech_model()

st.title("📡 SOPHIA-IQ // PLEROMA SPECTRAL COMMAND")
st.caption("Wideband SDR Signal Intercept • Neural Transcription • Audio Capture Engine")


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
enable_ai_transcription = st.sidebar.checkbox("Enable Neural Speech Transcription (Whisper AI)", value=True)


# ==============================================================================
# [SECTION 3] DSP & AI TRANSCRIPTION FUNCTIONS
# ==============================================================================

def generate_simulated_iq(sample_rate=2.4e6, num_samples=16384):
    """Generates complex I/Q spectral data for visualization."""
    t = np.arange(num_samples) / sample_rate
    noise = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) * 0.05
    carrier = 0.6 * np.exp(1j * 2 * np.pi * (translated_offset * 1e3) * t)
    return noise + carrier

def transcribe_audio_buffer(audio_bytes):
    """Transcribes raw audio bytes into text using Whisper AI."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name

        result = whisper_model.transcribe(tmp_path)
        transcript = result.get("text", "").strip()
        return transcript if transcript else "No intelligible speech detected in audio capture."
    except Exception as e:
        return f"Transcription error: {str(e)}"


# ==============================================================================
# [SECTION 4] LIVE AUDIO CAPTURE & RECORDING SUITE
# ==============================================================================

col1, col2 = st.columns([1.1, 1.9])

with col1:
    st.subheader("🎙️ Live Sound & Signal Capture")
    st.caption("Record live audio/transmissions directly from your hardware microphone:")
    
    # Native browser audio recorder
    captured_audio = st.audio_input("Click microphone to record intercept:")

    audio_data = None
    if captured_audio is not None:
        audio_data = captured_audio.read()
        st.subheader("🔊 Audio Playback & Export")
        st.audio(audio_data, format="audio/wav")
        
        st.download_button(
            label="💾 Download Intercept (.WAV)",
            data=audio_data,
            file_name=f"sophia_intercept_{(target_freq_mhz + (translated_offset/1000)):.3f}MHz.wav",
            mime="audio/wav",
            use_container_width=True
        )

        st.subheader("📝 Live AI Audio Transcript")
        if enable_ai_transcription:
            with st.spinner("Whisper AI processing captured audio..."):
                transcript = transcribe_audio_buffer(audio_data)
                st.text_area("Neural Speech-to-Text Output", value=transcript, height=120)
        else:
            st.info("AI Transcription disabled in sidebar controls.")
    else:
        st.info("Record audio above to activate playback, AI transcription, and file saving.")


# ==============================================================================
# [SECTION 5] ISOLATED SPECTRAL VISUALIZER (NO FLICKERING)
# ==============================================================================

@st.fragment(run_every=1.0)
def render_smooth_spectrum():
    iq_data = generate_simulated_iq()
    fft_vals = np.abs(np.fft.fftshift(np.fft.fft(iq_data))) ** 2
    freqs = np.linspace(target_freq_mhz - 1.2, target_freq_mhz + 1.2, len(fft_vals))
    fft_db = 10 * np.log10(fft_vals + 1e-12)

    power_dbm = 10 * np.log10(np.mean(fft_vals) + 1e-12)
    peak_dbm = 10 * np.log10(np.max(fft_vals) + 1e-12)
    snr = peak_dbm - power_dbm

    if snr > 12:
        color = "#00FFCC"
        glow = "rgba(0, 255, 204, 0.4)"
        status_text = "SIGNAL BURST DETECTED"
    else:
        color = "#0099FF"
        glow = "rgba(0, 153, 255, 0.2)"
        status_text = "WIDEBAND SCANNING"

    with col2:
        st.subheader("🤖 Receiver Status")
        st.markdown(f"""
            <div class="status-panel" style="border: 2px solid {color}; box-shadow: 0 0 20px {glow};">
                <h3 style="color: {color}; margin: 0; font-family: monospace;">{status_text}</h3>
                <p style="color: #A0AAB5; margin: 8px 0 0 0;">Tuned Frequency: <b>{(target_freq_mhz + (translated_offset/1000)):.3f} MHz</b> | SNR: <b>{snr:.1f} dB</b></p>
            </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        st.subheader("📊 High-Definition Spectral Visualizer")

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
            height=440,
            xaxis=dict(gridcolor="#121826", zerolinecolor="#121826"),
            yaxis=dict(gridcolor="#121826", zerolinecolor="#121826")
        )

        st.plotly_chart(fig, use_container_width=True, config={
            'toImageButtonOptions': {
                'format': 'png',
                'filename': f'spectrum_snapshot_{target_freq_mhz:.3f}MHz',
                'height': 720,
                'width': 1280,
                'scale': 2
            }
        })
        st.caption("📷 *Click the camera icon at the top right of the spectrum chart to download a PNG image snapshot.*")

render_smooth_spectrum()
