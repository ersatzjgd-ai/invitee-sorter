import streamlit as st
import pandas as pd
import cv2
import pytesseract
import numpy as np
from PIL import Image
from rapidfuzz import process, fuzz

st.set_page_config(page_title="Letter Sorter Pro", layout="wide")

# --- UI Styling ---
st.markdown("""
    <style>
    .poc-display {
        background-color: #28a745;
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        font-size: 60px;
        font-weight: bold;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }
    .invitee-label {
        color: #6c757d;
        text-align: center;
        font-size: 20px;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("✉️ Smart Invitee Sorter")

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

# --- 2. Enhanced OCR Logic ---
captured_image = st.camera_input("Capture Label")

if captured_image:
    # Open and convert
    img = Image.open(captured_image)
    img_array = np.array(img)
    
    # --- PRE-PROCESSING FOR ACCURACY ---
    # 1. Convert to Grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # 2. Rescale Image (2x Zoom) - This helps Tesseract read small text much better
    height, width = gray.shape
    gray = cv2.resize(gray, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
    
    # 3. Apply threshold to make text pop (Black & White)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # Run OCR with PSM 4 (Assume a single column of text)
    custom_config = r'--oem 3 --psm 4'
    raw_text = pytesseract.image_to_string(gray, config=custom_config)
    
    # Clean up lines: Ignore empty lines and "noisy" lines with < 3 characters (like 'Ss')
    lines = [l.strip() for l in raw_text.split('\n') if len(l.strip()) > 2]
    
    # Debugging Info (Collapsed by default)
    with st.expander("View OCR Debug Info"):
        st.write("Lines detected after cleaning:", lines)

    match_found = False
    
    if len(lines) >= 1:
        # We try to match Line 2 first (index 1), then Line 1, then Line 3
        # This ensures that even if 'Ms.' is on a separate line, we find it.
        indices_to_check = [1, 0, 2] # Priority: Line 2, then Line 1, then Line 3
        
        best_overall_match = None
        highest_score = 0
        
        for idx in indices_to_check:
            if idx < len(lines):
                candidate = lines[idx]
                # We use token_set_ratio which is great for "Ms. Varsha Sharma Ji" vs "Varsha Sharma"
                match = process.extractOne(candidate, invitee_list, scorer=fuzz.token_set_ratio)
                
                if match and match[1] > 75: # If score is high enough
                    if match[1] > highest_score:
                        highest_score = match[1]
                        best_overall_match = (match[0], candidate)
        
        if best_overall_match:
            final_name, original_text = best_overall_match
            poc = df.loc[df['Name'] == final_name, 'Main POC Name'].values[0]
            
            # --- 3. Result Display ---
            st.markdown(f'<div class="poc-display">{poc}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="invitee-label">Detected: "{original_text}" → Matched: {final_name}</div>', unsafe_allow_html=True)
            match_found = True

    if not match_found:
        st.warning("Could not identify a name. Please ensure the label is in focus and try again.")
        if len(lines) > 0:
            st.write("OCR saw:", lines)

st.sidebar.write(f"Database size: {len(df)} entries")
