import streamlit as st
import pandas as pd
import cv2
import pytesseract
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from rapidfuzz import process, fuzz

st.set_page_config(page_title="Letter Sorter", layout="wide")
st.title("📇 Fast POC Letter Sorter")

# 1. Load the Excel Data
# Replace 'invitees.xlsx' with your actual file name
@st.cache_data
def load_data():
    try:
        # Assuming the columns are exactly "Name" and "Main POC Name"
        df = pd.read_excel("invitees.xlsx")
        # Drop rows with missing names to avoid matching errors
        df = df.dropna(subset=['Name']) 
        return df
    except Exception as e:
        st.error(f"Error loading Excel file: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.warning("Please upload or ensure 'invitees.xlsx' is in the directory.")
    st.stop()

# Extract list of names for rapid fuzzy matching
invitee_names = df['Name'].astype(str).tolist()

# 2. Define the Video Transformer for Continuous Scanning
class OCRVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.last_poc = "Scanning..."
        self.last_name = ""

    def transform(self, frame):
        # Convert video frame to numpy array
        img = frame.to_ndarray(format="bgr24")
        
        # Pre-process image for better OCR (grayscale)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Run Tesseract OCR
        # --psm 6 assumes a single uniform block of text (good for labels)
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(gray, config=custom_config)
        
        # Process the text: split by newlines and remove empty lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Check if we have at least 2 lines (to grab Line 2)
        if len(lines) >= 2:
            # Index 1 is Line 2 (0-indexed)
            scanned_name = lines[1] 
            
            # Fuzzy match the scanned name against our Excel list
            # Requires at least a 75% match to be considered valid
            match = process.extractOne(scanned_name, invitee_names, scorer=fuzz.WRatio, score_cutoff=75)
            
            if match:
                matched_name = match[0]
                # Look up the POC for the matched name
                poc_name = df.loc[df['Name'] == matched_name, 'Main POC Name'].values[0]
                
                self.last_name = matched_name
                self.last_poc = f"POC: {poc_name}"
            else:
                self.last_poc = "No match found"
        else:
            self.last_poc = "Waiting for label..."

        # 3. Draw the result directly on the camera feed for fast sorting
        # Create a background rectangle for text visibility
        cv2.rectangle(img, (0, 0), (640, 100), (0, 0, 0), -1)
        
        # Put POC Name on the screen (Large Green Text)
        cv2.putText(img, self.last_poc, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        # Put the matched Invitee Name underneath (Smaller White Text)
        cv2.putText(img, f"Found: {self.last_name}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return img

# 4. Initialize the WebRTC Streamer
st.write("Grant camera permissions to start continuous scanning.")
webrtc_streamer(
    key="ocr-scanner",
    video_transformer_factory=OCRVideoTransformer,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)
