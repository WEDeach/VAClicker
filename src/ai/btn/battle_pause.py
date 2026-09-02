import time

from ... import AI
from ...ocr import parse_ocr_boxes
from ...utils.clicker import click_at
from ...utils.shared import state
from ...utils.window import get_window_rect, get_window_screen

# 戰鬥暫停畫面: 中央「Pause」+ 左「倍速」+ 右「自動」同時存在才判定
PAUSE_TEXT_REGION = (0.40, 0.62, 0.45, 0.55)
SPEED_BTN_REGION = (0.30, 0.45, 0.64, 0.72)
AUTO_BTN_REGION = (0.55, 0.70, 0.64, 0.72)
PAUSE_KEYWORDS = ("pause",)
SPEED_KEYWORDS = ("倍速",)
AUTO_KEYWORDS = ("自動",)


def _normalise_pause_text(text) -> str:
    if text is None:
        return ""
    return "".join(str(text).split())


def _region_has_keyword(screen, region, keywords) -> bool:
    height, width = screen.shape[:2]
    x1f, x2f, y1f, y2f = region
    x1, x2 = int(x1f * width), int(x2f * width)
    y1, y2 = int(y1f * height), int(y2f * height)
    crop = screen[y1:y2, x1:x2]
    if state.ocr is None or crop.size == 0:
        return False
    try:
        boxes = parse_ocr_boxes(state.ocr.predict(crop))
    except Exception:
        return False
    normalized_keywords = tuple(keyword.casefold() for keyword in keywords)
    for text, _ in boxes:
        normalized = _normalise_pause_text(text).casefold()
        if any(keyword in normalized for keyword in normalized_keywords):
            return True
    return False


class BattlePause(AI):
    def __init__(self, *, delay_click: float = 1.0):
        super().__init__()
        self.delay_click = delay_click

    def check(self) -> bool:
        screen = get_window_screen()
        if screen is None or screen.size == 0:
            return False
        if not _region_has_keyword(screen, PAUSE_TEXT_REGION, PAUSE_KEYWORDS):
            return False
        if not _region_has_keyword(screen, SPEED_BTN_REGION, SPEED_KEYWORDS):
            return False
        if not _region_has_keyword(screen, AUTO_BTN_REGION, AUTO_KEYWORDS):
            return False
        height, width = screen.shape[:2]
        left, top, _, _ = get_window_rect()
        state.logger.info("偵測到戰鬥暫停畫面，點擊畫面中央")
        click_at((left + width // 2, top + height // 2))
        time.sleep(self.delay_click)
        return True
