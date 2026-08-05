import random
import time

import cv2
import numpy as np

from ... import AI
from ...ocr import parse_ocr_boxes
from ...utils.clicker import calculate_click_point, click_at
from ...utils.image import load_template, match_template
from ...utils.shared import state
from ...utils.window import get_window_screen

BLESSING_MENU_REGION = (0.30, 0.70, 0.40, 0.82)
BLESSING_TEXT = "加護"

BLESSING_WHITE_TIER = 0
BLESSING_GREEN_TIER = 1
BLESSING_BLUE_TIER = 2

WHITE_MAX_SATURATION = 60
WHITE_MIN_VALUE = 180
GREEN_HUE_RANGE = (35, 85)
BLUE_HUE_RANGE = (95, 130)
COLOR_MIN_SATURATION = 80
COLOR_MIN_VALUE = 80
MIN_COLOR_PIXEL_RATIO = 0.05


class Return(AI):
    def __init__(
        self,
        *,
        need_ret_inn: bool = False,
        current_battle_num: int = 1,
        max_battle_num: int = 7,
        delay_return: float = 5.0,
        delay_blessing: float = 5.0,
    ):
        super().__init__()
        self.need_ret_inn = need_ret_inn
        self.current_battle_num = current_battle_num
        self.max_battle_num = max_battle_num
        self.delay_return = delay_return
        self.delay_blessing = delay_blessing

    def check(self) -> bool:
        _screen = get_window_screen()
        _tmpl, _mask = load_template("BTN_歸還.png", grayscale=True)
        _match = match_template(
            _screen,
            _tmpl,
            0.8,
            True,
            _mask,
            ocr_check=[("歸還", 0.7)],
            region=(0.34, 0.65, 0.72, 0.79),
        )
        if _match:
            loc, score = _match
            point = calculate_click_point(loc, (_tmpl.shape[1], _tmpl.shape[0]))
            click_at(point)
            time.sleep(self.delay_return)
            self.increment_battle()
            return True
        return self._select_blessing(_screen)

    def _select_blessing(self, screen: np.ndarray) -> bool:
        h, w = screen.shape[:2]
        x1f, x2f, y1f, y2f = BLESSING_MENU_REGION
        x1, x2 = int(x1f * w), int(x2f * w)
        y1, y2 = int(y1f * h), int(y2f * h)
        crop = screen[y1:y2, x1:x2]
        if crop.size == 0:
            return False

        boxes = parse_ocr_boxes(state.ocr.predict(crop))
        candidates = [(text, box) for text, box in boxes if BLESSING_TEXT in text]
        if not candidates:
            return False

        ranked = []
        for text, (bx1, by1, bx2, by2) in candidates:
            full_x1, full_y1 = x1 + bx1, y1 + by1
            full_x2, full_y2 = x1 + bx2, y1 + by2
            tier = self._classify_tier(screen[full_y1:full_y2, full_x1:full_x2])
            if tier is None:
                state.logger.debug("[Return] 無法辨識加護顏色，跳過候選: %s", text)
                continue
            ranked.append((tier, text, (full_x1, full_y1, full_x2, full_y2)))

        if not ranked:
            state.logger.warning("[Return] 找不到有效的加護候選")
            return False

        top_tier = max(tier for tier, _, _ in ranked)
        top_candidates = [c for c in ranked if c[0] == top_tier]
        _, chosen_text, (cx1, cy1, cx2, cy2) = random.choice(top_candidates)
        state.logger.info(
            "[Return] 選擇加護: %s (tier=%d, 候選數=%d)",
            chosen_text,
            top_tier,
            len(top_candidates),
        )

        point = calculate_click_point((cx1, cy1), (cx2 - cx1, cy2 - cy1))
        click_at(point)
        time.sleep(self.delay_blessing)
        return True

    def _classify_tier(self, box_bgr: np.ndarray):
        if box_bgr.size == 0:
            return None
        hsv = cv2.cvtColor(box_bgr, cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        total = hue.size

        white_mask = (sat <= WHITE_MAX_SATURATION) & (val >= WHITE_MIN_VALUE)
        green_mask = (
            (hue >= GREEN_HUE_RANGE[0])
            & (hue <= GREEN_HUE_RANGE[1])
            & (sat >= COLOR_MIN_SATURATION)
            & (val >= COLOR_MIN_VALUE)
        )
        blue_mask = (
            (hue >= BLUE_HUE_RANGE[0])
            & (hue <= BLUE_HUE_RANGE[1])
            & (sat >= COLOR_MIN_SATURATION)
            & (val >= COLOR_MIN_VALUE)
        )

        counts = {
            BLESSING_BLUE_TIER: int(np.count_nonzero(blue_mask)),
            BLESSING_GREEN_TIER: int(np.count_nonzero(green_mask)),
            BLESSING_WHITE_TIER: int(np.count_nonzero(white_mask)),
        }
        best_tier = max(counts, key=counts.get)
        if counts[best_tier] / total < MIN_COLOR_PIXEL_RATIO:
            return None
        return best_tier

    def increment_battle(self) -> None:
        self.current_battle_num += 1
        if self.current_battle_num >= self.max_battle_num:
            self.need_ret_inn = True
