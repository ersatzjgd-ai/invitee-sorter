import streamlit as st
import pandas as pd
import cv2
import pytesseract
import numpy as np
from PIL import Image
from rapidfuzz import process, fuzz

st.set_page_config(page_title="Letter Sorter", layout="wide")

# --- 1. Custom Styling for Speed ---
st.markdown("""
    <style>
    .poc-box {
        background-color: #00C851;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .name-box {
        background-color: #33b5e5;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-size: 20px;
    }
    </style>
    """, unsafe_index=True)

st.title("📇 Fast Capture POC Sorter")

# --- 2. Excel Upload Section ---
st.sidebar.header("Setup")
uploaded_file = st.sidebar.file_uploader("Upload POC Excel Sheet", type=["xlsx"])

@st.cache_data
def load_data(file):
    try:
        df = pd.read_excel(file)
        required_cols = ["Name", "Main POC Name"]
        if not all(col in df.columns for col in required_cols):
            st.error(f"Excel must have columns: {required_cols}")
            return pd.DataFrame()
        return df.dropna(subset=['Name'])
    except Exception as e:
        st.error(f"Error reading Excel: {e}")
        return pd.DataFrame()

if uploaded_file:
    df = load_data(uploaded_file)
    invitee_names = df['Name'].astype(str).tolist()
    st.sidebar.success(f"Loaded {len(df)} contacts.")
else:
    st.info("👈 Please upload your 'invitees.xlsx' file in the sidebar.")
    st.stop()

# --- 3. Fast Picture Capture ---
# This widget opens the native phone camera
img_file = st.camera_input("Capture Label", label_visibility="hidden")

if img_file:
    # Convert the file to an opencv image
    img = Image.open(img_file)
    img_array = np.array(img)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Run OCR
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(gray, config=custom_config)
    
    # Process lines
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    if len(lines) >= 2:
        scanned_name = lines[1] # Target Line 2
        
        # Fuzzy match
        match = process.extractOne(scanned_name, invitee_names, scorer=fuzz.WRatio, score_cutoff=70)
        
        if match:
            matched_name = match[0]
            poc_name = df.loc[df['Name'] == matched_name, 'Main POC Name'].values[0]
            
            # --- 4. Large Visual Output ---
            st.markdown(f'<div class="poc-box">{poc_name}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="name-box">Invitee: {matched_name}</div>', unsafe_allow_html=True)
            
            # Show a snippet of the image to confirm focus
            st.image(img, caption="Last Scanned Label", width=300)
        else:
            st.warning(f"Could not find a match for: '{scanned_name}'")
            st.info("Try getting closer to the label or improving the light.")
    else:
        st.error("OCR failed to detect multiple lines. Make sure the full label is in frame.")
