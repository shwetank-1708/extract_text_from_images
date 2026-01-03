import streamlit as st
from PIL import Image
import numpy as np
import easyocr

st.title("OCR Test App")

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded Image")

    reader = easyocr.Reader(["en"], gpu=False)
    result = reader.readtext(np.array(image), detail=0, paragraph=True)

    st.subheader("Extracted Text")
    st.write("\n".join(result))
