import streamlit as st
import pandas as pd
import cv2
import pytesseract
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from rapidfuzz import process, fuzz

st.set_page_config(page_title="Letter Sorter", layout="wide")
st.title("📇 Fast POC Letter Sorter")

# --- 1. Excel Upload Section ---
st.sidebar.header("Setup")
uploaded_file = st.sidebar.file_uploader("Upload POC Excel Sheet", type=["xlsx"])

@st.cache_data
def load_data(file):
    try:
        df = pd.read_excel(file)
        # Ensure the required columns exist
        required_cols = ["Name", "Main POC Name"]
        if not all(col in df.columns for col in required_cols):
            st.error(f"Excel must have columns: {required_cols}")
            return pd.DataFrame()
        
        return df.dropna(subset=['Name'])
    except Exception as e:
        st.error(f"Error reading Excel: {e}")
        return pd.DataFrame()

# Global variables for the transformer to access
df = pd.DataFrame()
invitee_names = []

if uploaded_file:
    df = load_data(uploaded_file)
    if not df.empty:
        invitee_names = df['Name'].astype(str).tolist()
        st.success(f"Loaded {len(df)} contacts. Ready to scan!")
else:
    st.info("👈 Please upload your 'invitees.xlsx' file in the sidebar to begin.")
    st.stop()

# --- 2. OCR & Matching Logic ---
class OCRVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.last_poc = "Searching..."
        self.last_name = ""

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Tesseract config for label reading
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(gray, config=custom_config)
        
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # As requested: specifically use Line 2 of the detected text
        if len(lines) >= 2:
            scanned_name = lines[1] 
            
            # Use fuzzy matching to handle camera grain/OCR errors
            match = process.extractOne(scanned_name, invitee_names, scorer=fuzz.WRatio, score_cutoff=70)
            
            if match:
                matched_name = match[0]
                poc_name = df.loc[df['Name'] == matched_name, 'Main POC Name'].values[0]
                self.last_name = matched_name
                self.last_poc = f"POC: {poc_name}"
            else:
                self.last_poc = "No match found"
        else:
            self.last_poc = "Focus on Line 2..."

        # Visual Feedback on screen
        cv2.rectangle(img, (0, 0), (700, 110), (0, 0, 0), -1)
        cv2.putText(img, self.last_poc, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(img, f"Read: {self.last_name}", (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return img

# --- 3. Camera Interface ---
webrtc_streamer(
    key="ocr-scanner",
    video_transformer_factory=OCRVideoTransformer,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False}
)
