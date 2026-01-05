from __future__ import annotations

from typing import Any, Dict, List, Union
import numpy as np
from PIL import Image

from paddleocr import PaddleOCR


def load_paddle_ocr(lang: str = "en") -> PaddleOCR:
    """
    Load PaddleOCR reader once and reuse.
    """
    return PaddleOCR(
        use_angle_cls=True,
        lang=lang,
        show_log=False
    )


def extract_with_paddleocr(
    image: Union[str, Image.Image, np.ndarray],
    ocr: PaddleOCR
) -> Dict[str, Any]:
    if isinstance(image, Image.Image):
        img = np.array(image.convert("RGB"))
    elif isinstance(image, str):
        img = image
    elif isinstance(image, np.ndarray):
        img = image
    else:
        raise TypeError("Unsupported image type")

    result = ocr.ocr(img, cls=True)

    texts: List[str] = []
    confs: List[float] = []

    for line in result:
        for box, (text, conf) in line:
            texts.append(text)
            confs.append(float(conf))

    return {
        "engine": "paddleocr",
        "text": "\n".join(texts),
        "avg_confidence": float(np.mean(confs)) if confs else 0.0,
    }
