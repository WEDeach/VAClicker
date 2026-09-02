import time

from ... import AI
from ...ocr import parse_ocr_boxes
from ...utils.clicker import click_at
from ...utils.shared import state
from ...utils.text_map import get_text_mapping
from ...utils.window import get_window_rect, get_window_screen

# 伺服器錯誤畫面: 中央「返回標題畫面」按鈕
SERVER_ERROR_BTN_REGION = (0.42, 0.60, 0.52, 0.60)
SERVER_ERROR_TEXT_KEY = "btn.server_error.return_title"
SERVER_ERROR_BTN_KEYWORDS = ("返回標題", "返回标题")


class ServerError(AI):
    def __init__(self, *, delay_click: float = 2.0):
        super().__init__()
        self.delay_click = delay_click

    def check(self) -> bool:
        _screen = get_window_screen()
        if _screen is None or _screen.size == 0:
            return False
        _mapping_text = get_text_mapping(SERVER_ERROR_TEXT_KEY)
        _keywords = SERVER_ERROR_BTN_KEYWORDS + (_mapping_text,)
        _height, _width = _screen.shape[:2]
        _x1f, _x2f, _y1f, _y2f = SERVER_ERROR_BTN_REGION
        _x1, _x2 = int(_x1f * _width), int(_x2f * _width)
        _y1, _y2 = int(_y1f * _height), int(_y2f * _height)
        _crop = _screen[_y1:_y2, _x1:_x2]
        if state.ocr is None or _crop.size == 0:
            return False
        try:
            _boxes = parse_ocr_boxes(state.ocr.predict(_crop))
        except Exception:
            state.logger.warning("伺服器錯誤畫面 OCR 失敗")
            return False
        for _text, (_bx1, _by1, _bx2, _by2) in _boxes:
            _normalized = "".join(str(_text).split())
            if any(_keyword in _normalized for _keyword in _keywords):
                _left, _top, _, _ = get_window_rect()
                _point = (
                    _left + _x1 + (_bx1 + _bx2) // 2,
                    _top + _y1 + (_by1 + _by2) // 2,
                )
                state.logger.info("偵測到伺服器錯誤，點擊返回標題畫面")
                click_at(_point)
                time.sleep(self.delay_click)
                return True
        return False
