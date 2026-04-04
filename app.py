import streamlit as st
import pandas as pd
import cv2
import pytesseract
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from rapidfuzz import process, fuzz

# --- UI Styling ---
st.set_page_config(page_title="Pro Letter Sorter", layout="wide")
st.markdown("""
    <style>
    .poc-display {
        background-color: #007bff; color: white; padding: 30px;
        border-radius: 15px; text-align: center; font-size: 55px;
        font-weight: bold; box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }
    .invitee-label { color: #6c757d; text-align: center; font-size: 20px; margin-top: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("📸 Pro Sorter: High-Quality OCR")

# --- 1. Data Loading ---
st.sidebar.header("Setup")
uploaded_file = st.sidebar.file_uploader("Upload POC List", type=["xlsx", "csv"])

@st.cache_data
def load_data(file):
    try:
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        return df.dropna(subset=['Name'])
    except Exception as e:
        st.error(f"Error: {e}")
        return None

if not uploaded_file:
    st.info("Upload your list to start.")
    st.stop()

df = load_data(uploaded_file)
invitee_list = df['Name'].astype(str).tolist()

# --- 2. Flash & Camera Controls ---
st.sidebar.subheader("Camera Settings")
use_flash = st.sidebar.checkbox("Turn Flash (Torch) On")

# --- 3. Image Processing for High Quality ---
def improve_image(img):
    # Convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Bilateral Filter: Removes noise but keeps edges (letters) sharp
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Adaptive Thresholding: Handles uneven lighting/shadows on the letter
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, 11, 2)
    
    # 2x Zoom for OCR clarity
    height, width = gray.shape
    gray = cv2.resize(gray, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
    return gray

# --- 4. The OCR Engine ---
def scan_for_name(img):
    processed = improve_image(img)
    # PSM 4: Assume a single column of text of variable sizes
    custom_config = r'--oem 3 --psm 4'
    raw_text = pytesseract.image_to_string(processed, config=custom_config)
    
    lines = [l.strip() for l in raw_text.split('\n') if len(l.strip()) > 3]
    
    best_match = None
    highest_score = 0

    for line in lines:
        # We use Token Set Ratio to ignore "Ms." or "Ji" and find the core name
        match = process.extractOne(line, invitee_list, scorer=fuzz.token_set_ratio)
        if match and match[1] > 70:
            if match[1] > highest_score:
                highest_score = match[1]
                best_match = (match[0], line)
    return best_match

# --- 5. High-Resolution Capture ---
# We use WebRTC for the torch toggle, but process frames individually
ctx = webrtc_streamer(
    key="pro-scanner",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={
        "video": {
            "facingMode": "environment", # Use back camera
            "width": {"ideal": 1280},    # High resolution
            "height": {"ideal": 720},
            "torch": use_flash           # Attempt to turn on Flash
        },
        "audio": False,
    },
)

if ctx.video_transformer:
    # Button to trigger a high-quality capture from the live stream
    if st.button("📸 SCAN NOW"):
        # Note: In webrtc, we capture the current frame
        frame = ctx.video_transformer.last_frame 
        if frame is not None:
            res = scan_for_name(frame)
            if res:
                final_name, original_text = res
                poc = df.loc[df['Name'] == final_name, 'Main POC Name'].values[0]
                st.markdown(f'<div class="poc-display">{poc}</div>', unsafe_allow_html=True)
               st.markdown(f'<div class="invitee-label">Matched: {final_name}</div>', unsafe_allow_html=True)
