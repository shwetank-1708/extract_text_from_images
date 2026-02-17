import re
import json
import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from jiwer import wer
from rapidfuzz.distance import Levenshtein

from ocr.paddle_ocr import load_paddle_ocr, extract_with_paddleocr


# ---------------- Page ---------------- #
st.set_page_config(page_title="OCR App", layout="wide")
st.title("Extract Text from Images (PaddleOCR)")

@st.cache_resource
def get_ocr():
    return load_paddle_ocr(lang="en")


# ---------------- Helpers ---------------- #
def normalize_text(s: str) -> str:
    """Normalize for fair WER/CER comparisons."""
    s = s or ""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)  # collapse whitespace
    return s


def compute_cer(gt: str, pred: str) -> float:
    """CER = edit_distance(chars) / len(gt_chars)"""
    gt_n = normalize_text(gt)
    pr_n = normalize_text(pred)

    if len(gt_n) == 0:
        return 0.0 if len(pr_n) == 0 else 1.0

    dist = Levenshtein.distance(gt_n, pr_n)
    return dist / len(gt_n)


def preprocess_image(
    img: Image.Image,
    use_grayscale: bool,
    contrast: float,
    sharpen: bool,
    threshold: int | None,
) -> Image.Image:
    """Simple OCR-friendly preprocessing using PIL only."""
    out = img.convert("RGB")

    if use_grayscale:
        out = out.convert("L")  # grayscale

    # contrast: 1.0 means no change
    if contrast and abs(contrast - 1.0) > 1e-6:
        out = ImageEnhance.Contrast(out).enhance(contrast)

    if sharpen:
        out = out.filter(ImageFilter.SHARPEN)

    if threshold is not None:
        # ensure grayscale before thresholding
        if out.mode != "L":
            out = out.convert("L")
        out = out.point(lambda p: 255 if p >= threshold else 0)

    return out


# ---------------- Layout ---------------- #
left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Input")

    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    st.markdown("### Preprocessing (optional)")
    use_grayscale = st.checkbox("Grayscale", value=True)
    contrast = st.slider("Contrast", min_value=0.5, max_value=2.5, value=1.3, step=0.1)
    sharpen = st.checkbox("Sharpen", value=False)

    use_threshold = st.checkbox("Threshold (B/W)", value=False)
    threshold = st.slider("Threshold value", 0, 255, 150, 5) if use_threshold else None

    st.markdown("### Ground Truth (for evaluation)")
    gt_text = st.text_area(
        "Paste the actual/correct text here (optional)",
        placeholder="Type/paste the correct text from the image...",
        height=180,
    )

    img = None
    processed_img = None

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        processed_img = preprocess_image(
            img=img,
            use_grayscale=use_grayscale,
            contrast=contrast,
            sharpen=sharpen,
            threshold=threshold,
        )

        st.markdown("### Preview")
        tab1, tab2 = st.tabs(["Original", "Processed"])
        with tab1:
            st.image(img, caption="Original", use_container_width=True)
        with tab2:
            st.image(processed_img, caption="Processed", use_container_width=True)


with right:
    st.subheader("Output")

    if uploaded and processed_img is not None:
        with st.spinner("Extracting text..."):
            ocr = get_ocr()
            out = extract_with_paddleocr(processed_img, ocr)

        extracted = out.get("text", "") or ""
        avg_conf = float(out.get("avg_confidence", 0.0) or 0.0)

        st.text_area("Extracted Text", extracted, height=260)
        st.caption(f"Average confidence: {avg_conf:.2f}")

        # -------- Evaluation -------- #
        if gt_text.strip():
            pred_n = normalize_text(extracted)
            gt_n = normalize_text(gt_text)

            cer_val = compute_cer(gt_text, extracted)
            wer_val = wer(gt_n, pred_n)  # jiwer WER

            m1, m2, m3 = st.columns(3)
            m1.metric("CER (lower is better)", f"{cer_val:.3f}")
            m2.metric("WER (lower is better)", f"{wer_val:.3f}")
            m3.metric("Avg confidence", f"{avg_conf:.2f}")

            st.info(
                "Tip: CER is good for character-level mistakes (spelling/letters). "
                "WER is stricter and penalizes word changes more."
            )
        else:
            st.info("Paste Ground Truth text on the left to calculate CER/WER.")

        # -------- Download buttons -------- #
        st.markdown("### Download")
        st.download_button(
            label="Download extracted text (.txt)",
            data=extracted.encode("utf-8"),
            file_name="extracted_text.txt",
            mime="text/plain",
            use_container_width=True,
        )

        # Optional: download full JSON result
        st.download_button(
            label="Download OCR result (.json)",
            data=json.dumps(out, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="ocr_result.json",
            mime="application/json",
            use_container_width=True,
        )

    else:
        st.info("Upload an image on the left to see extracted text here.")
