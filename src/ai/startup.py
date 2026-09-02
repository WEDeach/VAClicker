"""Startup screen handlers.

Each handler recognizes one boot-time screen with two small OCR regions and
performs a single action (center click or close-button click). All coordinates
are normalized to the window size; OCR crops stay small to bound latency.
"""

import re
import time

from .. import AI
from ..ocr import parse_ocr_boxes
from ..utils.clicker import click_at
from ..utils.shared import state
from ..utils.window import get_window_rect, get_window_screen

# 須知畫面: 標題「請注意」與「未成年玩家須知」各取一個窄行
NOTICE_TITLE_REGION = (0.45, 0.55, 0.29, 0.36)
NOTICE_UNDERAGE_REGION = (0.43, 0.57, 0.53, 0.60)
NOTICE_TITLE_KEYWORDS = ("請注意", "請註意")
NOTICE_UNDERAGE_KEYWORDS = ("未成年玩家",)

# 登入畫面: 中央下方「Tap to Start」與右下角「Version」
LOGIN_TAP_REGION = (0.42, 0.58, 0.82, 0.89)
LOGIN_VERSION_REGION = (0.80, 0.93, 0.95, 1.00)
LOGIN_TAP_KEYWORDS = ("TaptoStart", "Tap to Start", "Taptostart")
LOGIN_VERSION_KEYWORD = "Version"
LOGIN_VERSION_RE = re.compile(r"Version[:：]?\s*([0-9]+\.[0-9]+(?:\.[0-9]+)?)")

# 登入時公告: 上方標題「公告」與下方「關閉」按鈕
ANNOUNCE_TITLE_REGION = (0.45, 0.55, 0.06, 0.14)
ANNOUNCE_CLOSE_REGION = (0.45, 0.55, 0.86, 0.94)
ANNOUNCE_TITLE_KEYWORDS = ("公告", "公告欄", "维护公告")
ANNOUNCE_CLOSE_KEYWORDS = ("關閉", "关闭", "X關閉", "X关闭")


def _normalise_startup_text(text) -> str:
    if text is None:
        return ""
    return "".join(str(text).split())


def _ocr_boxes(screen, region):
    height, width = screen.shape[:2]
    x1f, x2f, y1f, y2f = region
    x1, x2 = int(x1f * width), int(x2f * width)
    y1, y2 = int(y1f * height), int(y2f * height)
    crop = screen[y1:y2, x1:x2]
    if state.ocr is None or crop.size == 0:
        return [], True
    try:
        result = state.ocr.predict(crop)
        boxes = parse_ocr_boxes(result)
    except Exception:
        state.logger.warning("啟動畫面 OCR 失敗: region=%s", region)
        return [], False
    return [
        (text, (x1 + bx1, y1 + by1, x1 + bx2, y1 + by2))
        for text, (bx1, by1, bx2, by2) in boxes
    ], True


def _find_text_box(screen, region, keywords):
    normalized_keywords = tuple(
        _normalise_startup_text(keyword) for keyword in keywords
    )
    boxes, ocr_ok = _ocr_boxes(screen, region)
    for text, box in boxes:
        normalized = _normalise_startup_text(text)
        if any(keyword in normalized for keyword in normalized_keywords):
            return normalized, box, ocr_ok
    return None, None, ocr_ok


def _click_box_center(box):
    left, top, _, _ = get_window_rect()
    x1, y1, x2, y2 = box
    click_at((left + (x1 + x2) // 2, top + (y1 + y2) // 2))


def _click_center(screen):
    height, width = screen.shape[:2]
    left, top, _, _ = get_window_rect()
    click_at((left + width // 2, top + height // 2))


class NoticeScreen(AI):
    """須知畫面: 偵測「請注意」+「未成年玩家」後點擊畫面中心一次。"""

    def __init__(self, *, delay_after_click: float = 8.0):
        super().__init__()
        self.delay_after_click = delay_after_click
        self._handled = False

    def check(self) -> bool:
        screen = get_window_screen()
        if screen is None or screen.size == 0:
            return False
        title, _, title_ok = _find_text_box(
            screen, NOTICE_TITLE_REGION, NOTICE_TITLE_KEYWORDS
        )
        underage, _, underage_ok = _find_text_box(
            screen, NOTICE_UNDERAGE_REGION, NOTICE_UNDERAGE_KEYWORDS
        )
        if not (title_ok and underage_ok):
            return self._handled
        present = title is not None and underage is not None
        if not present:
            self._handled = False
            return False
        if not self._handled:
            self._handled = True
            state.logger.info("偵測到須知畫面，點擊畫面中心")
            _click_center(screen)
            time.sleep(self.delay_after_click)
        return True


class LoginScreen(AI):
    """登入畫面: 偵測「Tap to Start」+「Version」後列印版本並點擊中心。"""

    def __init__(self, *, delay_after_click: float = 5.0):
        super().__init__()
        self.delay_after_click = delay_after_click
        self._handled = False
        self._version_logged = False

    def check(self) -> bool:
        screen = get_window_screen()
        if screen is None or screen.size == 0:
            return False
        tap, _, tap_ok = _find_text_box(screen, LOGIN_TAP_REGION, LOGIN_TAP_KEYWORDS)
        version_box, _, version_ok = _find_text_box(
            screen, LOGIN_VERSION_REGION, (LOGIN_VERSION_KEYWORD,)
        )
        if not (tap_ok and version_ok):
            return self._handled
        present = tap is not None and version_box is not None
        if not present:
            self._handled = False
            self._version_logged = False
            return False
        if not self._handled:
            match = LOGIN_VERSION_RE.search(version_box)
            if match is None:
                return False
            self._handled = True
            version = match.group(1)
            state.logger.info("偵測到登入畫面，版本: %s", version)
            if not self._version_logged:
                print(f"遊戲版本: {version}")
                self._version_logged = True
            _click_center(screen)
            time.sleep(self.delay_after_click)
        return True


class AnnouncementScreen(AI):
    """登入公告: 偵測「公告」+「關閉」後點擊關閉按鈕。

    公告可能有多則: 關閉一則後下一則會接著出現, 因此只要公告畫面
    存在就持續點擊(以 delay_after_click 間隔抑制, 避免同一幀重複點)。
    """

    def __init__(self, *, delay_after_click: float = 1.0):
        super().__init__()
        self.delay_after_click = delay_after_click
        # 負值: 第一次偵測到公告時立即點擊
        self._last_click_at = -delay_after_click

    def check(self) -> bool:
        screen = get_window_screen()
        if screen is None or screen.size == 0:
            return False
        title, _, title_ok = _find_text_box(
            screen, ANNOUNCE_TITLE_REGION, ANNOUNCE_TITLE_KEYWORDS
        )
        close, close_box, close_ok = _find_text_box(
            screen, ANNOUNCE_CLOSE_REGION, ANNOUNCE_CLOSE_KEYWORDS
        )
        if not (title_ok and close_ok):
            return False
        present = title is not None and close is not None
        if not present:
            return False
        now = time.monotonic()
        if now - self._last_click_at >= self.delay_after_click:
            self._last_click_at = now
            state.logger.info("偵測到公告畫面，點擊關閉")
            _click_box_center(close_box)
        return True
