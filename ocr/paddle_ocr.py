# from __future__ import annotations
# from typing import Any, Dict, List, Union

# import numpy as np
# from PIL import Image


# def load_paddle_ocr(lang: str = "en") -> Any:
#     from paddleocr import PaddleOCR  # ✅ lazy import
#     return PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)


# def extract_with_paddleocr(
#     image: Union[str, Image.Image, np.ndarray],
#     ocr: Any
# ) -> Dict[str, Any]:
#     if isinstance(image, Image.Image):
#         img = np.array(image.convert("RGB"))
#     elif isinstance(image, str):
#         img = image
#     elif isinstance(image, np.ndarray):
#         img = image
#     else:
#         raise TypeError("Unsupported image type")

#     result = ocr.ocr(img, cls=True)

#     texts: List[str] = []
#     confs: List[float] = []

#     for line in result:
#         for box, (text, conf) in line:
#             texts.append(text)
#             confs.append(float(conf))

#     return {
#         "engine": "paddleocr",
#         "text": "\n".join(texts),
#         "avg_confidence": float(np.mean(confs)) if confs else 0.0,
#     }

# ocr/paddle_ocr.py

from __future__ import annotations
from typing import Any, Dict, List, Union

import numpy as np
from PIL import Image

from paddleocr import PaddleOCR


def load_paddle_ocr(lang: str = "en") -> PaddleOCR:
    # Most stable settings for Streamlit Cloud CPU
    return PaddleOCR(
        use_angle_cls=False,     # IMPORTANT: disable cls
        lang=lang,
        use_gpu=False,           # Streamlit Cloud CPU
        enable_mkldnn=False,     # avoid MKL/oneDNN issues
        show_log=False
    )


def _to_numpy_rgb(image: Union[Image.Image, np.ndarray]) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    if isinstance(image, np.ndarray):
        return image
    raise TypeError("Unsupported image type")


def extract_with_paddleocr(image: Union[Image.Image, np.ndarray], ocr: PaddleOCR) -> Dict[str, Any]:
    img = _to_numpy_rgb(image)

    # IMPORTANT: cls must be False (since we disabled angle cls)
    result = ocr.ocr(img, cls=False)

    texts: List[str] = []
    confs: List[float] = []

    # result format: [[ [box], (text, conf) ], ...]
    if result and result[0]:
        for line in result[0]:
            if len(line) >= 2:
                text, conf = line[1][0], float(line[1][1])
                texts.append(text)
                confs.append(conf)

    combined_text = "\n".join(texts).strip()
    avg_conf = float(np.mean(confs)) if confs else 0.0

    return {"engine": "paddleocr", "text": combined_text, "avg_confidence": avg_conf}
