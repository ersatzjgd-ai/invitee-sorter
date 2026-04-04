import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import easyocr
from rapidfuzz import process, fuzz

# --- UI Styling ---
st.set_page_config(page_title="Letter Sorter for POCs", layout="centered")

st.markdown("""
    <style>
    .big-poc-card {
        background-color: #198754; color: white; padding: 40px 20px;
        border-radius: 12px; text-align: center; font-size: 48px;
        font-weight: 900; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 10px;
    }
    .match-text { color: #495057; font-size: 18px; text-align: center; font-weight: 500; }
    </style>
    """, unsafe_allow_html=True)

st.title("OCR Letter Sorter")
st.write("Scan the letter sticker with the app.")

# --- 1. Load Data ---
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
    st.info("👈 Please upload your contact list to activate the camera.")
    st.stop()

df = load_data(uploaded_file)
if df is None: st.stop()
invitee_list = df['Name'].astype(str).tolist()

# --- 2. Load EasyOCR Model ---
# We use @st.cache_resource so the heavy AI model only loads once, making subsequent scans fast.
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en']) # 'en' stands for English

reader = load_ocr()

# --- 3. Camera Capture & Processing ---
st.markdown("### 📸 Scan Label")
captured_image = st.camera_input("Take a picture", label_visibility="collapsed")

st.markdown("---") 

if captured_image:
    with st.spinner("AI is analyzing the label..."):
        img = Image.open(captured_image)
        img_array = np.array(img)
        
        # EasyOCR reads directly from the array. No need for OpenCV thresholding!
        # It returns a list of tuples: (bounding_box, text, confidence_score)
        results = reader.readtext(img_array)
        
        # Extract the text from the results, ignoring low-confidence guesses or tiny artifacts
        lines = [res[1].strip() for res in results if res[2] > 0.25 and len(res[1].strip()) > 3]
        
        match_found = False
        best_match = None
        highest_score = 0

        # Scan detected lines for the best name match
        for line in lines:
            match = process.extractOne(line, invitee_list, scorer=fuzz.token_set_ratio)
            if match and match[1] > 75: 
                if match[1] > highest_score:
                    highest_score = match[1]
                    best_match = (match[0], line)

        # --- 4. Results Display ---
        if best_match:
            final_name, original_text = best_match
            poc = df.loc[df['Name'] == final_name, 'Main POC Name'].values[0]
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f'<div class="big-poc-card">{poc}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="match-text">Verified Match: <b>{final_name}</b></div>', unsafe_allow_html=True)
            with col2:
                st.success("Match Found!")
                st.caption(f"Camera read: '{original_text}'")
                st.caption(f"Confidence: {highest_score}%")
                
        else:
            st.error("⚠️ Could not match any text to your POC list.")
            with st.expander("Diagnostic: What the AI saw"):
                if lines:
                    st.write("Detected text:", lines)
                else:
                    st.write("No clear text detected. Try adjusting the distance.")
