import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import easyocr
from rapidfuzz import process, fuzz

# --- UI Styling ---
st.set_page_config(page_title="Letter Sorter", layout="centered")

st.markdown("""
    <style>
    .big-poc-card {
        background-color: #198754; color: white; padding: 40px 20px;
        border-radius: 12px; text-align: center; font-size: 50px;
        font-weight: 900; box-shadow: 0 4px 10px rgba(0,0,0,0.2); margin-bottom: 10px;
    }
    .match-text { color: #495057; font-size: 20px; text-align: center; font-weight: 500; }
    </style>
    """, unsafe_allow_html=True)

st.title("Letter Sorter")
st.write("Scan the letter sticker with your mobile camera")

# --- 1. Data Loading ---
st.sidebar.header("📁 Data Source")
uploaded_file = st.sidebar.file_uploader("Upload POC List (CSV/Excel)", type=["xlsx", "csv"])

@st.cache_data
def load_data(file):
    try:
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        return df.dropna(subset=['Name'])
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

if not uploaded_file:
    st.info("👈 Please upload your contact list to activate the scanner.")
    st.stop()

df = load_data(uploaded_file)
if df is None: st.stop()
invitee_list = df['Name'].astype(str).tolist()

# --- 2. AI Model Initialization ---
@st.cache_resource
def load_ocr():
    # Downloads the model on first run; cached thereafter
    return easyocr.Reader(['en'])

reader = load_ocr()

# --- 3. Capture & Multi-Line Logic ---
captured_image = st.camera_input("Snap a photo of the label", label_visibility="collapsed")

if captured_image:
    with st.spinner("Analyzing first and second lines..."):
        img = Image.open(captured_image)
        img_array = np.array(img)
        
        # EasyOCR returns: [ ([bbox], text, confidence), ... ]
        # Results are usually returned in top-to-bottom order
        results = reader.readtext(img_array)
        
        # Filter for "real" text (ignore artifacts, symbols, or tiny noise)
        valid_lines = [res[1].strip() for res in results if len(res[1].strip()) > 2]
        
        # Focus specifically on the FIRST and SECOND detected lines
        lines_to_check = valid_lines[:2] 
        
        match_found = False
        best_overall_match = None
        highest_score = 0

        # Iterate through the first two lines to find the best match in the Excel
        for line in lines_to_check:
            # token_set_ratio handles "Ms. Varsha Sharma Ji" vs "Varsha Sharma" perfectly
            match = process.extractOne(line, invitee_list, scorer=fuzz.token_set_ratio)
            
            if match and match[1] > 75: # Standard confidence threshold
                if match[1] > highest_score:
                    highest_score = match[1]
                    best_overall_match = (match[0], line)

        # --- 4. Enhanced UI Results ---
        st.markdown("---")
        if best_overall_match:
            final_name, scanned_text = best_overall_match
            poc = df.loc[df['Name'] == final_name, 'Main POC Name'].values[0]
            
            st.markdown(f'<div class="big-poc-card">{poc}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="match-text">Matched: <b>{final_name}</b></div>', unsafe_allow_html=True)
            st.caption(f"Scanned text: '{scanned_text}' | Match Score: {highest_score}%")
        else:
            st.error("⚠️ No match found in the first two lines.")
            with st.expander("Diagnostic: What the AI detected"):
                st.write("First two lines identified:", lines_to_check)
                if len(valid_lines) > 2:
                    st.write("Other lines found:", valid_lines[2:])

st.sidebar.write(f"Database: {len(df)} names loaded.")
