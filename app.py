import streamlit as st
import pandas as pd
import cv2
import pytesseract
import numpy as np
from PIL import Image
from rapidfuzz import process, fuzz

st.set_page_config(page_title="Letter Sorter", layout="wide")

# --- UI Styling ---
st.markdown("""
    <style>
    .poc-display {
        background-color: #28a745;
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        font-size: 50px;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .invitee-label {
        color: #6c757d;
        text-align: center;
        font-size: 18px;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("✉️ Invitee Sorter: Fast Capture")

# --- 1. Load Excel Data ---
st.sidebar.header("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload POC Excel (Columns: Name, Main POC Name)", type=["xlsx"])

@st.cache_data
def load_data(file):
    try:
        df = pd.read_excel(file)
        if not all(col in df.columns for col in ["Name", "Main POC Name"]):
            st.error("Missing columns: 'Name' or 'Main POC Name'")
            return None
        return df.dropna(subset=['Name'])
    except Exception as e:
        st.error(f"Excel Error: {e}")
        return None

if not uploaded_file:
    st.info("Please upload your Excel file in the sidebar to start sorting.")
    st.stop()

df = load_data(uploaded_file)
if df is None: st.stop()
invitee_list = df['Name'].astype(str).tolist()

# --- 2. Camera Input ---
# This opens the native camera on your phone for a quick snap
captured_image = st.camera_input("Scan the Label")

if captured_image:
    # Process Image
    img = Image.open(captured_image)
    img_array = np.array(img)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # OCR specifically looking for Line 2
    # --psm 6 tells Tesseract to treat the image as a single block of text
    custom_config = r'--oem 3 --psm 6'
    raw_text = pytesseract.image_to_string(gray, config=custom_config)
    
    lines = [l.strip() for l in raw_text.split('\n') if l.strip()]
    
    if len(lines) >= 2:
        # TARGET: Line 2 as requested
        detected_name = lines[1]
        
        # Fuzzy match (minimum 70% match to prevent false positives)
        match = process.extractOne(detected_name, invitee_list, scorer=fuzz.WRatio, score_cutoff=70)
        
        if match:
            final_name = match[0]
            poc = df.loc[df['Name'] == final_name, 'Main POC Name'].values[0]
            
            # --- 3. Result Display ---
            st.markdown(f'<div class="poc-display">{poc}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="invitee-label">Detected: {final_name}</div>', unsafe_allow_html=True)
        else:
            st.warning(f"Could not match '{detected_name}' to any POC. Try re-snapping.")
    else:
        st.error("Could not find Line 2 on the label. Ensure the label is centered and clear.")

st.sidebar.write(f"Database contains **{len(df)}** invitees.")
