import streamlit as st
from PIL import Image
import numpy as np

from ocr.easy_ocr import load_reader, extract_with_easyocr

st.title("EasyOCR Test (Modular)")

@st.cache_resource
def get_reader():
    return load_reader(languages=("en",), gpu=False)

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded")

    reader = get_reader()
    out = extract_with_easyocr(img, reader=reader, paragraph=True, detail=1)

    st.subheader("Extracted Text")
    st.text_area("Text", out["text"], height=200)

    st.caption(f"Average confidence: {out['avg_confidence']:.2f}")
