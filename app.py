import streamlit as st
import pandas as pd
import easyocr
from PIL import Image
import numpy as np

# Page Config
st.set_page_config(page_title="Invitee Sorter", layout="centered")
st.title("✉️ Invitee to POC Sorter")

# Initialize OCR reader (downloads model on first run)
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'])

reader = load_ocr()

# 1. Upload the Excel Database
uploaded_file = st.sidebar.file_uploader("Upload POC Excel Sheet", type=["xlsx", "csv"])

if uploaded_file:
    # Load data
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('xlsx') else pd.read_csv(uploaded_file)
    
    # Clean data: Ensure columns exist and remove extra spaces
    df.columns = df.columns.str.strip()
    if 'Name' in df.columns and 'Main POC Name' in df.columns:
        df['Name_Clean'] = df['Name'].astype(str).str.lower().str.strip()
        st.sidebar.success("Database Loaded!")
    else:
        st.error("Excel must have 'Name' and 'Main POC Name' columns.")
else:
    st.info("Please upload your Excel file in the sidebar to begin.")

# 2. Camera Input
img_file = st.camera_input("Scan Invitee Label")

if img_file and uploaded_file:
    # Process Image
    img = Image.open(img_file)
    img_np = np.array(img)
    
    with st.spinner('Scanning label...'):
        # OCR Detection
        results = reader.readtext(img_np)
        
        # Logic to pick the "Name"
        # Usually, the name is the first or largest text block. 
        # We will check each detected line against our database.
        found_poc = None
        detected_name = ""

        for (bbox, text, prob) in results:
            clean_text = text.lower().strip()
            # Match check: See if detected text exists in our 'Name' column
            match = df[df['Name_Clean'] == clean_text]
            
            if not match.empty:
                found_poc = match['Main POC Name'].values[0]
                detected_name = text
                break

    # 3. Display Result
    if found_poc:
        st.balloons()
        st.markdown(f"### Found Invitee: **{detected_name}**")
        st.markdown(f"## 📍 Stack with POC: ")
        st.markdown(f"<h1 style='text-align: center; color: #ff4b4b; font-size: 80px;'>{found_poc}</h1>", unsafe_allow_html=True)
    else:
        st.warning("Could not find a matching name in the database. Try getting closer to the label.")
        if results:
            st.write("Detected text on label:", [res[1] for res in results])
