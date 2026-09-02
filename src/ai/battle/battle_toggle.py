"""戰鬥按鈕列: 「自動」與「倍速」toggle 偵測與點擊。

兩個按鈕都是單擊切換, 位置依畫面比例固定, 狀態依顏色改變
(off 為灰, on 為黃)。欄位控制是否自動點擊, 預設皆開啟:
按鈕為 off(灰)時點擊開啟, 為 on(黃)時跳過。
"""

import time
from typing import Tuple

import cv2

from ... import AI
from ...ocr import parse_ocr_boxes
from ...utils.clicker import click_at
from ...utils.shared import state
from ...utils.window import get_window_rect, get_window_screen

# 自動按鈕區域(涵蓋「自動」文字)與點擊點(文字中心)
AUTO_BTN_REGION: Tuple[float, float, float, float] = (0.615, 0.660, 0.645, 0.710)
AUTO_CLICK_POINT: Tuple[float, float] = (0.640, 0.678)
AUTO_TEXT_REGION: Tuple[float, float, float, float] = (0.620, 0.660, 0.660, 0.695)
# 倍速按鈕區域(涵蓋「倍速」文字與黃色指示)與點擊點(文字中心)
SPEED_BTN_REGION: Tuple[float, float, float, float] = (0.300, 0.370, 0.640, 0.710)
SPEED_CLICK_POINT: Tuple[float, float] = (0.357, 0.675)
SPEED_TEXT_REGION: Tuple[float, float, float, float] = (0.335, 0.375, 0.660, 0.695)
# on(黃)狀態: 區域內黃色像素比例門檻
TOGGLE_ON_YELLOW_MIN_RATIO = 0.05
TOGGLE_YELLOW_LOWER = (10, 80, 110)
TOGGLE_YELLOW_UPPER = (50, 255, 255)
TOGGLE_TEXT_KEYWORDS = ("自動", "倍速")
# 暫停畫面檢查: 有「Pause」時不動作(暫停畫面由 BattlePause 處理,
# 這裡只處理戰鬥進行中的按鈕列)
PAUSE_CHECK_REGION: Tuple[float, float, float, float] = (0.40, 0.62, 0.45, 0.55)
PAUSE_KEYWORDS = ("pause",)


def _crop_region(screen, region):
    height, width = screen.shape[:2]
    x1, x2 = int(region[0] * width), int(region[1] * width)
    y1, y2 = int(region[2] * height), int(region[3] * height)
    return screen[y1:y2, x1:x2]


def _yellow_ratio(screen, region) -> float:
    crop = _crop_region(screen, region)
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, TOGGLE_YELLOW_LOWER, TOGGLE_YELLOW_UPPER)
    # cv2.inRange 輸出 0/255, countNonZero/size 才是真正的 [0,1] 比例
    return float(cv2.countNonZero(mask)) / float(mask.size)


def _region_has_keyword(screen, region, keywords) -> bool:
    """OCR 檢查區域內是否出現任一關鍵字。"""
    crop = _crop_region(screen, region)
    if crop.size == 0 or state.ocr is None:
        return False
    try:
        boxes = parse_ocr_boxes(state.ocr.predict(crop))
    except Exception:
        return False
    normalized_keywords = tuple("".join(str(k).split()).casefold() for k in keywords)
    for text, _ in boxes:
        normalized = "".join(str(text).split()).casefold()
        if any(keyword in normalized for keyword in normalized_keywords):
            return True
    return False


def _has_toggle_text(screen, region) -> bool:
    """OCR 確認按鈕列存在(找到「自動」或「倍速」文字)。

    避免在非戰鬥畫面(按鈕列不存在)誤點擊固定區域。
    """
    return _region_has_keyword(screen, region, TOGGLE_TEXT_KEYWORDS)


class BattleToggle(AI):
    """自動/倍速 toggle: off(灰)時點擊開啟, on(黃)時跳過。

    click_auto / click_speed 控制是否自動點擊對應按鈕(預設 True)。
    僅在按鈕列存在(OCR 找到「自動」/「倍速」)時動作, 避免誤點其他畫面。
    """

    def __init__(
        self,
        *,
        click_auto: bool = True,
        click_speed: bool = True,
        delay_after_click: float = 0.8,
    ):
        super().__init__()
        self.click_auto = click_auto
        self.click_speed = click_speed
        self.delay_after_click = delay_after_click

    def _is_on(self, screen, region) -> bool:
        return _yellow_ratio(screen, region) >= TOGGLE_ON_YELLOW_MIN_RATIO

    def _click_point(self, screen, point):
        height, width = screen.shape[:2]
        left, top, _, _ = get_window_rect()
        click_at((left + int(point[0] * width), top + int(point[1] * height)))

    def check(self) -> bool:
        screen = get_window_screen()
        if screen is None or screen.size == 0:
            return False
        # 暫停畫面不動作(由 BattlePause 處理)
        if _region_has_keyword(screen, PAUSE_CHECK_REGION, PAUSE_KEYWORDS):
            return False
        # 按鈕列存在檢查: 「自動」在自動區域、「倍速」在倍速區域
        # (各區域必須出現對應文字, 避免單一誤判授權兩個點擊)
        if not (
            _region_has_keyword(screen, AUTO_TEXT_REGION, ("自動",))
            and _region_has_keyword(screen, SPEED_TEXT_REGION, ("倍速",))
        ):
            return False
        clicked = False
        if self.click_auto and not self._is_on(screen, AUTO_BTN_REGION):
            state.logger.info("偵測到「自動」按鈕 off，點擊開啟")
            self._click_point(screen, AUTO_CLICK_POINT)
            clicked = True
        if self.click_speed and not self._is_on(screen, SPEED_BTN_REGION):
            state.logger.info("偵測到「倍速」按鈕 off，點擊開啟")
            self._click_point(screen, SPEED_CLICK_POINT)
            clicked = True
        if clicked:
            time.sleep(self.delay_after_click)
        return clicked
