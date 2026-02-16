# import streamlit as st
# from PIL import Image
# import numpy as np

# from ocr.paddle_ocr import load_paddle_ocr, extract_with_paddleocr

# st.set_page_config(page_title="OCR App", layout="centered")
# st.title("Extract Text from Images (PaddleOCR)")

# @st.cache_resource
# def get_ocr():
#     return load_paddle_ocr(lang="en")

# uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# if uploaded:
#     img = Image.open(uploaded).convert("RGB")
#     st.image(img, caption="Uploaded image", use_container_width=True)

#     ocr = get_ocr()
#     out = extract_with_paddleocr(img, ocr)

#     st.subheader("Extracted Text")
#     st.text_area("Text", out["text"], height=250)

#     st.caption(f"Average confidence: {out['avg_confidence']:.2f}")

import streamlit as st
from PIL import Image

from ocr.paddle_ocr import load_paddle_ocr, extract_with_paddleocr

st.set_page_config(page_title="OCR App", layout="wide")
st.title("Extract Text from Images (PaddleOCR)")

@st.cache_resource
def get_ocr():
    return load_paddle_ocr(lang="en")

# Two-column layout
left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Input")
    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    img = None
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded image", use_container_width=True)

with right:
    st.subheader("Output")
    if uploaded and img is not None:
        with st.spinner("Extracting text..."):
            ocr = get_ocr()
            out = extract_with_paddleocr(img, ocr)

        st.text_area("Extracted Text", out["text"], height=350)
        st.caption(f"Average confidence: {out['avg_confidence']:.2f}")
    else:
        st.info("Upload an image on the left to see extracted text here.")