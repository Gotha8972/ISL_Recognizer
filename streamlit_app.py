import streamlit as st
import cv2
import av
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from cmain import PureCV_ISLRecognizer

# --- Page Configuration ---
st.set_page_config(
    page_title="High-Speed ISL Recognition",
    page_icon="🤟",
    layout="wide"
)

st.title("🤟 Real-Time ISL Recognition (Optimized 30+ FPS)")
st.markdown("""
**Performance Mode:** Using `video_frame_callback` and `av` frames to minimize latency. 
No artificial frame-rate capping.
""")

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Configuration")

# Path to your gesture templates
template_path = st.sidebar.text_input(
    "Template Folder Path",
    value="templates"
)

# Stability timing (how long you must hold the sign)
countdown_time = st.sidebar.slider(
    "Gesture Stability (seconds)",
    min_value=0.5,
    max_value=3.0,
    value=1.5,
    step=0.1
)

show_mask = st.sidebar.checkbox("Show Performance Metrics & Mask", value=True)

# --- Initialize Recognizer ---
@st.cache_resource
def load_recognizer(path):
    """Caches the recognizer object so templates aren't reloaded every frame."""
    return PureCV_ISLRecognizer(path)

try:
    recognizer = load_recognizer(template_path)
    # Inject user-defined settings into the recognizer
    recognizer.countdown_target = countdown_time
    recognizer.show_skin_mask = show_mask
except Exception as e:
    st.error(f"Error loading templates: {e}")
    st.stop()

# --- Reset Button ---
if st.sidebar.button("🗑 Clear Current Word"):
    recognizer.current_word = []
    recognizer.countdown_active = False

# --- High-Performance Video Callback ---
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    """
    This function runs in a dedicated video thread.
    It processes every camera frame as fast as possible.
    """
    # 1. Convert to OpenCV format (BGR)
    img = frame.to_ndarray(format="bgr24")
    
    # 2. Mirror for natural interaction
    img = cv2.flip(img, 1)

    # 3. Process frame through the Pure CV Pipeline (Skin mask -> Hu Moments -> Match)
    # Using the existing logic from your cmain.py
    processed_img = recognizer.process_frame(img)

    # 4. Return the processed frame back to the browser
    return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

# --- Streamer Component ---
webrtc_streamer(
    key="isl-30fps",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 640},
            "height": {"ideal": 480},
            "frameRate": {"ideal": 30}
        },
        "audio": False
    },
    async_processing=True, # Critical for high FPS
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)

# --- Live Data Dashboard ---
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Current Sentence")
    # Display current word progress
    current = "".join(recognizer.current_word)
    st.write(f"### `{current if current else '[Waiting for sign...]'}`")

with col2:
    st.subheader("📚 Word History")
    if recognizer.words_history:
        for word in reversed(recognizer.words_history[-5:]):
            st.write(f"- {word}")
    else:
        st.caption("No words finalized yet.")

# --- Footer ---
st.caption("Developed with Pure Computer Vision (No Deep Learning)")