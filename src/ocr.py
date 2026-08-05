import importlib
import logging
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="paddle")
warnings.filterwarnings("ignore", category=UserWarning, module="requests")

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_onednn_skip_unknown_arch"] = "1"
os.environ["PADDLEX_LOGGING_LEVEL"] = "ERROR"

_stderr = sys.stderr
sys.stderr = open(os.devnull, "w")
PaddleOCR = importlib.import_module("paddleocr").PaddleOCR
sys.stderr = _stderr

for _name in ("ppocr", "paddlex", "paddle"):
    logging.getLogger(_name).setLevel(logging.ERROR)


def get_paddle_ocr(lang="chinese_cht"):
    _so, _se = sys.stdout, sys.stderr
    _fd_out = os.dup(1)
    _fd_err = os.dup(2)
    _devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull, 1)
    os.dup2(_devnull, 2)
    os.close(_devnull)
    sys.stdout = sys.stderr = open(os.devnull, "w")
    try:
        instance = PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    finally:
        sys.stdout, sys.stderr = _so, _se
        os.dup2(_fd_out, 1)
        os.dup2(_fd_err, 2)
        os.close(_fd_out)
        os.close(_fd_err)
    return instance


def _to_int(value) -> int:
    return int(round(float(value)))


def _polygon_to_box(polygon) -> tuple[int, int, int, int]:
    xs = [_to_int(pt[0]) for pt in polygon]
    ys = [_to_int(pt[1]) for pt in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _box_like_to_box(box) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return _to_int(x1), _to_int(y1), _to_int(x2), _to_int(y2)


def _boxes_from_mapping(
    mapping,
) -> list[tuple[str, tuple[int, int, int, int]]]:
    boxes: list[tuple[str, tuple[int, int, int, int]]] = []
    rec_texts = mapping.get("rec_texts")
    rec_boxes = mapping.get("rec_boxes")
    rec_polys = mapping.get("rec_polys")
    if isinstance(rec_texts, list) and rec_boxes is not None:
        for text, box in zip(rec_texts, rec_boxes):
            if text:
                boxes.append((str(text), _box_like_to_box(box)))
    elif isinstance(rec_texts, list) and rec_polys is not None:
        for text, poly in zip(rec_texts, rec_polys):
            if text:
                boxes.append((str(text), _polygon_to_box(poly)))
    else:
        t = mapping.get("rec_text") or mapping.get("text", "")
        poly = mapping.get("rec_poly") or mapping.get("box")
        if t and poly is not None:
            boxes.append((str(t), _polygon_to_box(poly)))
    return boxes


def _boxes_from_json(data) -> list[tuple[str, tuple[int, int, int, int]]]:
    boxes: list[tuple[str, tuple[int, int, int, int]]] = []
    for block in data if isinstance(data, list) else [data]:
        if not isinstance(block, dict):
            continue
        payload = block.get("res") if "res" in block else block
        if not isinstance(payload, dict):
            continue
        boxes.extend(_boxes_from_mapping(payload))
    return boxes


def parse_ocr_boxes(result) -> list[tuple[str, tuple[int, int, int, int]]]:
    boxes: list[tuple[str, tuple[int, int, int, int]]] = []
    if result is None:
        return boxes
    for item in result:
        if item is None:
            continue
        if isinstance(item, list):
            for line in item:
                if isinstance(line, (list, tuple)) and len(line) >= 2:
                    polygon, text_info = line[0], line[1]
                    if isinstance(text_info, (list, tuple)):
                        text = str(text_info[0])
                    elif isinstance(text_info, str):
                        text = text_info
                    else:
                        continue
                    boxes.append((text, _polygon_to_box(polygon)))
                elif isinstance(line, dict):
                    t = line.get("rec_text") or line.get("text", "")
                    poly = line.get("rec_poly") or line.get("box")
                    if t and poly is not None:
                        boxes.append((t, _polygon_to_box(poly)))
        elif hasattr(item, "get"):
            found = _boxes_from_mapping(item)
            if found:
                boxes.extend(found)
            elif hasattr(item, "json"):
                boxes.extend(_boxes_from_json(item.json))
        elif hasattr(item, "json"):
            boxes.extend(_boxes_from_json(item.json))
    return boxes


def parse_ocr_texts(result) -> list[str]:
    texts: list[str] = []
    if result is None:
        return texts
    for item in result:
        if item is None:
            continue
        if isinstance(item, list):
            for line in item:
                if isinstance(line, (list, tuple)) and len(line) >= 2:
                    text_info = line[1]
                    if isinstance(text_info, (list, tuple)):
                        texts.append(str(text_info[0]))
                    elif isinstance(text_info, str):
                        texts.append(text_info)
                elif isinstance(line, dict):
                    t = line.get("rec_text") or line.get("text", "")
                    if t:
                        texts.append(t)
        elif isinstance(item, dict):
            rec_texts = item.get("rec_texts")
            if isinstance(rec_texts, list):
                texts.extend(str(t) for t in rec_texts if t)
            else:
                t = item.get("rec_text") or item.get("text", "")
                if t:
                    texts.append(t)
        elif hasattr(item, "json"):
            data = item.json
            for block in data if isinstance(data, list) else [data]:
                if isinstance(block, dict):
                    t = block.get("rec_text") or block.get("text", "")
                    if t:
                        texts.append(t)
    return texts
