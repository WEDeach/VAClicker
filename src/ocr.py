import os
import sys
import logging
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="paddle")
warnings.filterwarnings("ignore", category=UserWarning, module="requests")

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_onednn_skip_unknown_arch"] = "1"
os.environ["PADDLEX_LOGGING_LEVEL"] = "ERROR"

_stderr = sys.stderr
sys.stderr = open(os.devnull, "w")
from paddleocr import PaddleOCR
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
