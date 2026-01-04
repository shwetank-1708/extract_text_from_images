# ocr/easy_ocr.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

import easyocr


BBox = List[List[float]]  # 4 points: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]


@dataclass
class OCRItem:
    bbox: BBox
    text: str
    confidence: float


def _to_numpy_rgb(image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
    """
    Convert input image to an RGB numpy array suitable for EasyOCR.
    Supports:
      - file path (str)
      - PIL Image
      - numpy array (RGB or BGR; best effort)
    """
    if isinstance(image, str):
        # EasyOCR can read paths, but we keep everything consistent as np RGB
        pil_img = Image.open(image).convert("RGB")
        return np.array(pil_img)

    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))

    if isinstance(image, np.ndarray):
        arr = image
        # If grayscale, convert to 3-channel RGB-like
        if arr.ndim == 2:
            return np.stack([arr, arr, arr], axis=-1)

        # If it looks like BGR from OpenCV (common), caller might pass BGR.
        # We can't reliably detect, but a simple heuristic:
        # if mean of channel 0 is much higher than channel 2, likely BGR.
        # We'll leave it as-is unless user converts beforehand.
        return arr

    raise TypeError("Unsupported image type. Provide a file path, PIL Image, or numpy array.")


def load_reader(languages: Sequence[str] = ("en",), gpu: bool = False) -> easyocr.Reader:
    """
    Load EasyOCR reader. Call once and reuse.
    In Streamlit, you should wrap this using st.cache_resource.
    """
    return easyocr.Reader(list(languages), gpu=gpu)


def extract_with_easyocr(
    image: Union[str, Image.Image, np.ndarray],
    reader: easyocr.Reader,
    paragraph: bool = True,
    detail: int = 1,
) -> Dict[str, Any]:
    """
    Run EasyOCR and return a consistent dict:
      {
        "engine": "easyocr",
        "text": "...",
        "avg_confidence": 0.83,
        "items": [
            {"bbox": [...], "text": "...", "confidence": 0.91},
            ...
        ]
      }

    detail=1 returns (bbox, text, confidence) per item.
    paragraph=True merges nearby lines into paragraph-style chunks.
    """
    img_rgb = _to_numpy_rgb(image)

    results = reader.readtext(img_rgb, detail=detail, paragraph=paragraph)

    items: List[OCRItem] = []
    texts: List[str] = []
    confs: List[float] = []

    # Expected format (detail=1): [ (bbox, text, conf), ... ]
    # If paragraph=True, still typically includes conf; but handle safely.
    for r in results:
        if isinstance(r, (list, tuple)) and len(r) >= 2:
            bbox = r[0]
            text = r[1]
            conf = float(r[2]) if len(r) >= 3 and r[2] is not None else 0.0

            if text is None:
                continue
            text_str = str(text).strip()
            if not text_str:
                continue

            # Ensure bbox is list-of-lists (JSON-friendly)
            bbox_ll = [[float(x), float(y)] for x, y in bbox]

            items.append(OCRItem(bbox=bbox_ll, text=text_str, confidence=conf))
            texts.append(text_str)
            confs.append(conf)

    combined_text = "\n".join(texts).strip()
    avg_conf = float(np.mean(confs)) if confs else 0.0

    return {
        "engine": "easyocr",
        "text": combined_text,
        "avg_confidence": avg_conf,
        "items": [item.__dict__ for item in items],
    }
