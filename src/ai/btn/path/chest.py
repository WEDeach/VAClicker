import time
from enum import IntEnum

from .... import AI
from ....utils.clicker import calculate_click_point, click_at
from ....utils.image import load_template, match_template
from ....utils.shared import state
from ....utils.window import get_window_rect, get_window_screen

v1 = "路徑_寶箱.png"
v2 = "路徑_寶箱_v2.png"
v3 = "路徑_標記.png"

# 寶箱路徑 icon 偵測區域 (右上角)
CHEST_ICON_REGION = (0.91, 0.99, 0.15, 0.25)
# 路標按鈕位置 (右上角)
MARK_BUTTON_POINT = (0.95, 0.22)
# 路標 icon 偵測 (MARK_ONLY 模式啟用條件, 避免在其他畫面誤點)
MARK_ICON_REGION = (0.91, 0.99, 0.15, 0.25)
MARK_ICON_THRESHOLD = 0.9
# 找不到寶箱訊息的 OCR 檢查區域
NOT_FOUND_OCR_REGION = (0.36, 0.65, 0.41, 0.57)


class PathMode(IntEnum):
    """寶箱路徑模式。

    CHEST_AND_MARK = 0: 先點擊寶箱路徑, 偵測到「找不到寶箱」才點擊路標 (預設)
    MARK_ONLY = 1:      一開始直接點擊路標路徑, 不尋找寶箱
    """

    CHEST_AND_MARK = 0
    MARK_ONLY = 1


class ChestPath(AI):
    def __init__(
        self,
        *,
        mode: PathMode = PathMode.CHEST_AND_MARK,
        delay_not_found: float = 15.0,
        delay_check_not_found: float = 1.2,
    ):
        super().__init__()
        self.mode = PathMode(mode)
        self.delay_not_found = delay_not_found
        self.delay_check_not_found = delay_check_not_found
        self.not_found_end_at = 0

    def _chest_icon_present(self, screen) -> bool:
        _tmpl, _mask = load_template(v2, grayscale=True)
        return (
            match_template(
                screen,
                _tmpl,
                0.9,
                True,
                _mask,
                region=CHEST_ICON_REGION,
            )
            is not None
        )

    def _mark_icon_present(self, screen) -> bool:
        """路標 icon 存在(僅 MARK_ONLY 用, 作為點擊路標前的啟用條件)。"""
        _tmpl, _mask = load_template(v3, grayscale=True)
        return (
            match_template(
                screen,
                _tmpl,
                MARK_ICON_THRESHOLD,
                True,
                _mask,
                region=MARK_ICON_REGION,
            )
            is not None
        )

    def _click_mark(self) -> None:
        left, top, width, height = get_window_rect()
        click_at(
            (
                int(left + width * MARK_BUTTON_POINT[0]),
                int(top + height * MARK_BUTTON_POINT[1]),
            )
        )

    def _chest_icon_match(self, screen):
        _tmpl, _mask = load_template(v2, grayscale=True)
        match = match_template(
            screen, _tmpl, 0.9, True, _mask, region=CHEST_ICON_REGION
        )
        if match is None:
            return None
        loc, score = match
        return loc, score, (_tmpl.shape[1], _tmpl.shape[0])

    def check(self, *, check_only: bool = False) -> bool:
        # 有冷卻 則只點一次路標, 別問為啥 因為我爽
        if not check_only and time.time() < self.not_found_end_at:
            return False

        _screen = get_window_screen()

        if self.mode == PathMode.MARK_ONLY:
            # 直接點路標路徑, 不尋找寶箱; 但需先確認路標 icon 存在
            # (避免在城鎮/登入等沒有路徑選擇的畫面誤點固定位置)
            if not self._mark_icon_present(_screen):
                return False
            if check_only:
                return True
            self._click_mark()
            self.not_found_end_at = time.time() + self.delay_not_found
            return True

        # ChestAndMark: 先找寶箱路徑 icon
        match = self._chest_icon_match(_screen)
        if match is None:
            return False
        if check_only:
            return True
        loc, score, (tpl_w, tpl_h) = match
        point = calculate_click_point(loc, (tpl_w, tpl_h))
        click_at(point)

        self.not_found_path()
        return True

    def not_found_path(self) -> bool:
        # TODO: ASYNC TO CHECK
        time.sleep(self.delay_check_not_found)  # update screen
        _screen = get_window_screen()
        _match = match_template(
            _screen,
            None,
            1,
            False,
            None,
            ocr_check=[("找不到寶箱", 0), ("找不到前往目的地的路線", 0)],
            region=NOT_FOUND_OCR_REGION,
        )
        if _match:
            state.logger.debug("找不到路線，點擊路標")
            self.not_found_end_at = time.time() + self.delay_not_found
            self._click_mark()
            return True
        return False
