import streamlit as st
from PIL import Image
import numpy as np

from ocr.paddle_ocr import load_paddle_ocr, extract_with_paddleocr

st.set_page_config(page_title="OCR App", layout="centered")
st.title("Extract Text from Images (PaddleOCR)")

@st.cache_resource
def get_ocr():
    return load_paddle_ocr(lang="en")

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    ocr = get_ocr()
    out = extract_with_paddleocr(img, ocr)

    st.subheader("Extracted Text")
    st.text_area("Text", out["text"], height=250)

    st.caption(f"Average confidence: {out['avg_confidence']:.2f}")

