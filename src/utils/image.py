from pathlib import Path
from re import Pattern
from typing import Optional, Tuple

import cv2
import numpy as np

from ..ocr import parse_ocr_texts
from .shared import state

logger = state.logger
BASE_W, BASE_H = 1920, 1080


def load_template(template_name: str, *, grayscale: bool):
    template_path = Path(__file__).parents[2] / "assets" / template_name
    if not template_path.exists():
        raise FileNotFoundError(f"template not found: {template_path}")
    buf = np.fromfile(str(template_path), dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)  # alpha
    if image is None:
        raise ValueError(f"template is invalid: {template_path}")

    mask = None

    if image.ndim == 3 and image.shape[2] == 4:
        alpha = image[:, :, 3]
        mask = np.where(alpha > 0, np.uint8(255), np.uint8(0))
        image = image[:, :, :3]

    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if mask is not None and grayscale:
            pass
    return image, mask


def match_template(
    haystack: np.ndarray,
    template: np.ndarray,
    threshold: float,
    grayscale: bool,
    mask: Optional[np.ndarray] = None,
    log_score: bool = False,
    ocr_check: Optional[list[Tuple[str | Pattern, float]]] = None,
    region: Optional[Tuple[float, float, float, float]] = None,
):
    if grayscale:
        haystack_proc = (
            cv2.cvtColor(haystack, cv2.COLOR_BGR2GRAY)
            if haystack.ndim == 3
            else haystack
        )
        use_mask = mask
    else:
        haystack_proc = haystack
        use_mask = None
    if region is not None:
        x1, x2, y1, y2 = region
        h, w = haystack_proc.shape[:2]
        x1 = int(x1 * w)
        x2 = int(x2 * w)
        y1 = int(y1 * h)
        y2 = int(y2 * h)
        haystack_proc = haystack_proc[y1:y2, x1:x2]
    if template is not None:
        template, use_mask = normalize_template(
            template, use_mask, screen_size=haystack.shape[:2]
        )
        result = cv2.matchTemplate(
            haystack_proc, template, cv2.TM_CCOEFF_NORMED, mask=use_mask
        )
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if not np.isfinite(max_val):
            result = cv2.matchTemplate(
                haystack_proc, template, cv2.TM_CCORR_NORMED, mask=use_mask
            )
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if region is not None:
            max_loc = (max_loc[0] + x1, max_loc[1] + y1)
        if log_score:
            logger.info(f"模板比對結果: max_val={max_val:.4f} at {max_loc}")
        if not np.isfinite(max_val):
            max_val = 1.0
            if grayscale:
                return None
    else:
        if region is None:
            raise ValueError("region 參數必須與空模板一起使用")
        max_val = 0
        max_loc = ((x1 + x2) // 2, (y1 + y2) // 2)
    if ocr_check is not None:
        # 有設置ocr 則以ocr為主，且必須同時滿足比對分數門檻和ocr驗證才算成功
        if template is not None:
            th, tw = template.shape[:2]
            x, y = int(max_loc[0]), int(max_loc[1])
            r = haystack[y : y + th, x : x + tw]
        else:
            r = haystack[y1:y2, x1:x2]
        texts = "\n".join(parse_ocr_texts(state.ocr.predict(r)))
        if log_score:
            cv2.imshow("haystack", haystack)
            cv2.imshow("OCR Check Region", r)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        for text, score in ocr_check:
            found = (
                bool(text.search(texts)) if isinstance(text, Pattern) else text in texts
            )
            if texts and log_score:
                logger.debug(
                    f"[OCR] expected='{text}', found={texts}, score={score:.4f}, "
                    f"max_val={max_val:.4f} at {max_loc}"
                )
            if found and max_val >= score:
                return max_loc, max_val
        if log_score:
            logger.debug(
                f"[OCR] Failed: texts={texts}, ocr_check={ocr_check}, "
                f"max_val={max_val:.4f} at {max_loc}"
            )
        return None
    if max_val >= threshold:
        return max_loc, max_val
    return None


def normalize_template(
    template: np.ndarray, mask: Optional[np.ndarray], *, screen_size: Tuple[int, int]
) -> np.ndarray:
    sh, sw = screen_size
    if sh == BASE_H and sw == BASE_W:
        return template, mask

    h, w = template.shape[:2]
    scale_x = sw / BASE_W
    scale_y = sh / BASE_H
    scale = min(scale_x, scale_y)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    logger.debug(
        "normalize_template: "
        f"screen_size={screen_size}, original_template_size={(w, h)}, "
        f"scale=({scale_x:.4f}, {scale_y:.4f}), new_template_size=({new_w}, {new_h})",
    )
    resized = cv2.resize(template, (new_w, new_h))
    resized_mask = None
    if mask is not None:
        resized_mask = cv2.resize(mask, (new_w, new_h))
    return resized, resized_mask
