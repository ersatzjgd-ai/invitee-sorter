import streamlit as st
import pandas as pd
import cv2
import pytesseract
import numpy as np
from PIL import Image
from rapidfuzz import process, fuzz

st.set_page_config(page_title="Letter Sorter Pro", layout="wide")

st.markdown("""
    <style>
    .poc-display {
        background-color: #28a745; color: white; padding: 30px;
        border-radius: 15px; text-align: center; font-size: 55px;
        font-weight: bold; box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }
    .invitee-label { color: #6c757d; text-align: center; font-size: 20px; margin-top: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("✉️ Smart POC Sorter")

# --- 1. Data Loading ---
st.sidebar.header("Setup")
uploaded_file = st.sidebar.file_uploader("Upload POC List (CSV or Excel)", type=["xlsx", "csv"])

@st.cache_data
def load_data(file):
    try:
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        if not all(col in df.columns for col in ["Name", "Main POC Name"]):
            st.error("File must have 'Name' and 'Main POC Name' columns.")
            return None
        return df.dropna(subset=['Name'])
    except Exception as e:
        st.error(f"Error: {e}")
        return None

if not uploaded_file:
    st.info("Please upload your contact list to begin.")
    st.stop()

df = load_data(uploaded_file)
if df is None: st.stop()
invitee_list = df['Name'].astype(str).tolist()

# --- 2. Advanced OCR Logic ---
captured_image = st.camera_input("Capture Label")

if captured_image:
    img = Image.open(captured_image)
    img_array = np.array(img)
    
    # --- PRE-PROCESSING ---
    # 1. Grayscale & Denoise (Removes phone camera grain)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    # 2. Rescale (3x Zoom for better character definition)
    height, width = gray.shape
    gray = cv2.resize(gray, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC)
    
    # 3. Sharpening & Threshold
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # Run OCR
    custom_config = r'--oem 3 --psm 4'
    raw_text = pytesseract.image_to_string(gray, config=custom_config)
    
    # Filter out empty or very short junk lines (like 'Ss')
    lines = [l.strip() for l in raw_text.split('\n') if len(l.strip()) > 3]

    # Display what the OCR "sees" for troubleshooting
    if lines:
        with st.expander("Diagnostic: What the scanner read"):
            st.write(lines)
    
    match_found = False
    best_match = None
    highest_score = 0

    # We check all detected lines to find the best name match
    for line in lines:
        # WRatio handles "Ms. Varsha Sharma Ji" much better by looking for keywords
        match = process.extractOne(line, invitee_list, scorer=fuzz.WRatio)
        
        if match and match[1] > 65: # Lowered threshold slightly for better catch rate
            if match[1] > highest_score:
                highest_score = match[1]
                best_match = (match[0], line)

    if best_match:
        final_name, original_text = best_match
        poc = df.loc[df['Name'] == final_name, 'Main POC Name'].values[0]
        
        # --- 3. Result Display ---
        st.markdown(f'<div class="poc-display">{poc}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="invitee-label">Matched: <b>{final_name}</b> from label text: "{original_text}"</div>', unsafe_allow_html=True)
        match_found = True

    if not match_found:
        st.error("Could not identify a name. Try holding the phone steadier or getting more light on the label.")
        if lines:
            st.write("The scanner saw these lines but couldn't match them to your Excel:", lines)

st.sidebar.write(f"Database: {len(df)} entries")
