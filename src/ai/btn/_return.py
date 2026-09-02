import random
import time

import cv2
import numpy as np

from ... import AI
from ...ocr import parse_ocr_boxes
from ...utils.clicker import calculate_click_point, click_at
from ...utils.image import load_template, match_template
from ...utils.shared import state
from ...utils.window import get_window_rect, get_window_screen

BLESSING_MENU_REGION = (0.30, 0.70, 0.40, 0.82)
BLESSING_ROW_REGIONS = (
    (0.35, 0.65, 0.60, 0.66),
    (0.35, 0.65, 0.67, 0.73),
    (0.35, 0.65, 0.74, 0.80),
)
BLESSING_NOTHING_REGION = (0.35, 0.65, 0.84, 0.90)
BLESSING_NOTHING_TEXT = "都不做"
# 回歸選單防呆: 同時有「什麼都不做」與「歸還/回歸」時屬於回歸畫面, 不是加護選單
RETURN_MARKER_REGION = (0.24, 0.76, 0.50, 0.80)
RETURN_MARKER_KEYWORDS = ("歸還", "回歸")

BLESSING_WHITE_TIER = 0
BLESSING_GREEN_TIER = 1
BLESSING_BLUE_TIER = 2
BLESSING_PURPLE_TIER = 3
BLESSING_RED_TIER = 4

BLESSING_INFO_ICON_TEXT = frozenset({"i", "1", "l", "|"})
WHITE_MAX_SATURATION = 60
WHITE_MIN_VALUE = 180
GREEN_HUE_RANGE = (35, 85)
BLUE_HUE_RANGE = (95, 130)
PURPLE_HUE_RANGE = (131, 169)
RED_HUE_RANGES = ((0, 15), (170, 179))
COLOR_MIN_SATURATION = 80
COLOR_MIN_VALUE = 80
MIN_COLOR_PIXEL_RATIO = 0.05


def _normalise_blessing_text(text) -> str:
    if text is None:
        return ""
    text = "".join(str(text).split())
    if not text or text.casefold() in BLESSING_INFO_ICON_TEXT:
        return ""
    if not any(
        "\u3400" <= char <= "\u4dbf"
        or "\u4e00" <= char <= "\u9fff"
        or "\uf900" <= char <= "\ufaff"
        for char in text
    ):
        return ""
    return text


def _bounded_blessing_box(box, crop_shape, screen_shape, offset):
    try:
        if not isinstance(box, (tuple, list, np.ndarray)) or len(box) != 4:
            return None
        values = [float(value) for value in box]
        if not all(np.isfinite(value) for value in values):
            return None
    except (TypeError, ValueError, OverflowError):
        return None

    crop_height, crop_width = crop_shape[:2]
    if not (
        0 <= values[0] <= crop_width
        and 0 <= values[1] <= crop_height
        and 0 <= values[2] <= crop_width
        and 0 <= values[3] <= crop_height
    ):
        return None

    converted = [int(round(value)) for value in values]
    local_x1, local_y1, local_x2, local_y2 = converted
    if not (
        0 <= local_x1 <= crop_width
        and 0 <= local_y1 <= crop_height
        and 0 <= local_x2 <= crop_width
        and 0 <= local_y2 <= crop_height
        and local_x2 > local_x1
        and local_y2 > local_y1
    ):
        return None

    full_x1, full_y1 = offset[0] + local_x1, offset[1] + local_y1
    full_x2, full_y2 = offset[0] + local_x2, offset[1] + local_y2
    screen_height, screen_width = screen_shape[:2]
    if not (
        0 <= full_x1 < full_x2 <= screen_width
        and 0 <= full_y1 < full_y2 <= screen_height
    ):
        return None
    return full_x1, full_y1, full_x2, full_y2


def _choose_blessing_candidate(candidates):
    if not candidates:
        return None
    meaningful = []
    for candidate in candidates:
        if not isinstance(candidate, (tuple, list)) or len(candidate) != 2:
            continue
        text = _normalise_blessing_text(candidate[0])
        if text:
            meaningful.append((text, candidate[1]))
    if len(meaningful) != 1:
        return None
    return meaningful[0]


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
        if state.ocr is None:
            state.logger.debug("[Return] blessing OCR skipped: state.ocr is None")
            return False

        if not self._has_blessing_nothing_button(screen):
            return False

        return_box = self._has_return_marker(screen)
        if return_box is not False:
            if return_box is not None:
                self._click_return_menu(return_box)
            return True

        selected_rows = []
        filled_rows = []
        invalid_rows = []
        for row_number, region in enumerate(BLESSING_ROW_REGIONS, start=1):
            selected = self._select_blessing_row(screen, row_number, region)
            if selected is None:
                invalid_rows.append(row_number)
                continue
            text, (full_x1, full_y1, full_x2, full_y2) = selected
            tier, color_info = self._classify_tier(
                screen[full_y1:full_y2, full_x1:full_x2]
            )
            if tier is None:
                invalid_rows.append(row_number)
                state.logger.debug(
                    "[Return] 無法辨識加護顏色，跳過列: row=%d %s (%s)",
                    row_number,
                    text,
                    color_info,
                )
                continue

            filled_rows.append(row_number)
            selected_rows.append((row_number, tier, text, selected[1]))
            state.logger.debug(
                "[Return] 辨識加護顏色: row=%d %s -> tier=%d (%s)",
                row_number,
                text,
                tier,
                color_info,
            )

        state.logger.debug(
            "[Return] blessing filled rows: %s; invalid rows: %s",
            filled_rows,
            invalid_rows,
        )
        if invalid_rows or len(selected_rows) != len(BLESSING_ROW_REGIONS):
            state.logger.warning("[Return] blessing menu incomplete or ambiguous")
            return False

        ranked = [(tier, text, box) for _, tier, text, box in selected_rows]
        top_tier = max(tier for tier, _, _ in ranked)
        top_candidates = [c for c in ranked if c[0] == top_tier]
        state.logger.debug(
            "[Return] 所有加護候選: %s",
            ", ".join(f"{text}(tier={tier})" for tier, text, _ in ranked),
        )
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

    def _has_blessing_nothing_button(self, screen) -> bool:
        height, width = screen.shape[:2]
        x1f, x2f, y1f, y2f = BLESSING_NOTHING_REGION
        x1, x2 = int(x1f * width), int(x2f * width)
        y1, y2 = int(y1f * height), int(y2f * height)
        crop = screen[y1:y2, x1:x2]
        if crop.size == 0:
            state.logger.debug(
                "[Return] blessing menu gate skipped: empty crop "
                "region=(%.3f, %.3f, %.3f, %.3f)",
                x1f,
                x2f,
                y1f,
                y2f,
            )
            return False
        try:
            boxes = parse_ocr_boxes(state.ocr.predict(crop))
            box_items = list(boxes or [])
        except Exception:
            state.logger.warning("[Return] blessing menu gate OCR failed")
            return False
        for item in box_items:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                continue
            text = _normalise_blessing_text(item[0])
            if text and BLESSING_NOTHING_TEXT in text:
                state.logger.debug("[Return] blessing menu gate found: %s", text)
                return True
        return False

    def _click_return_menu(self, box) -> None:
        left, top, _, _ = get_window_rect()
        x1, y1, x2, y2 = box
        state.logger.info("偵測到回歸選單，點擊歸還")
        click_at((left + (x1 + x2) // 2, top + (y1 + y2) // 2))
        time.sleep(self.delay_return)
        self.increment_battle()

    def _has_return_marker(self, screen):
        """Return the return menu marker box when confirmed present.

        Returns None when the marker cannot be checked (OCR unavailable,
        empty crop, or OCR failure) so callers fail closed instead of
        proceeding to blessing selection.
        """
        if state.ocr is None:
            return None
        height, width = screen.shape[:2]
        x1f, x2f, y1f, y2f = RETURN_MARKER_REGION
        x1, x2 = int(x1f * width), int(x2f * width)
        y1, y2 = int(y1f * height), int(y2f * height)
        crop = screen[y1:y2, x1:x2]
        if crop.size == 0:
            state.logger.warning("[Return] return marker region empty")
            return None
        try:
            boxes = parse_ocr_boxes(state.ocr.predict(crop))
            box_items = list(boxes or [])
        except Exception:
            state.logger.warning("[Return] return marker OCR failed")
            return None
        for item in box_items:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                continue
            text = _normalise_blessing_text(item[0])
            if text and any(keyword in text for keyword in RETURN_MARKER_KEYWORDS):
                state.logger.debug("[Return] return menu marker found: %s", text)
                return (
                    x1 + item[1][0],
                    y1 + item[1][1],
                    x1 + item[1][2],
                    y1 + item[1][3],
                )
        state.logger.debug(
            "[Return] return menu marker not found: boxes=%d texts=%s",
            len(box_items),
            [item[0] for item in box_items if isinstance(item, (tuple, list))],
        )
        return False

    def _select_blessing_row(self, screen, row_number, region):
        height, width = screen.shape[:2]
        x1f, x2f, y1f, y2f = region
        x1, x2 = int(x1f * width), int(x2f * width)
        y1, y2 = int(y1f * height), int(y2f * height)
        crop = screen[y1:y2, x1:x2]
        if crop.size == 0:
            state.logger.debug(
                "[Return] blessing row=%d skipped OCR: empty crop "
                "region=(%.3f, %.3f, %.3f, %.3f)",
                row_number,
                x1f,
                x2f,
                y1f,
                y2f,
            )
            return None

        try:
            boxes = parse_ocr_boxes(state.ocr.predict(crop))
            box_items = list(boxes or [])
        except Exception:
            state.logger.warning("[Return] blessing row=%d OCR failed", row_number)
            return None

        row_candidates = []
        malformed_boxes = False
        for item in box_items:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                malformed_boxes = True
                continue
            text, box = item
            full_box = _bounded_blessing_box(box, crop.shape, screen.shape, (x1, y1))
            if full_box is None:
                malformed_boxes = True
                continue
            text = _normalise_blessing_text(text)
            if text:
                row_candidates.append((text, full_box))

        selected = None
        if not malformed_boxes:
            selected = _choose_blessing_candidate(row_candidates)

        state.logger.debug(
            "[Return] blessing row=%d OCR: crop=%s boxes=%d usable=%d "
            "filled=%s malformed=%s texts=%s",
            row_number,
            crop.shape,
            len(box_items),
            len(row_candidates),
            bool(selected),
            malformed_boxes,
            [text for text, _ in row_candidates],
        )
        return selected

    def _classify_tier(self, box_bgr: np.ndarray):
        if box_bgr.size == 0:
            return None, "total=0"
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
        purple_mask = (
            (hue >= PURPLE_HUE_RANGE[0])
            & (hue <= PURPLE_HUE_RANGE[1])
            & (sat >= COLOR_MIN_SATURATION)
            & (val >= COLOR_MIN_VALUE)
        )
        red_hue_mask = (
            (hue >= RED_HUE_RANGES[0][0]) & (hue <= RED_HUE_RANGES[0][1])
        ) | ((hue >= RED_HUE_RANGES[1][0]) & (hue <= RED_HUE_RANGES[1][1]))
        red_mask = (
            red_hue_mask & (sat >= COLOR_MIN_SATURATION) & (val >= COLOR_MIN_VALUE)
        )

        counts = {
            BLESSING_WHITE_TIER: int(np.count_nonzero(white_mask)),
            BLESSING_GREEN_TIER: int(np.count_nonzero(green_mask)),
            BLESSING_BLUE_TIER: int(np.count_nonzero(blue_mask)),
            BLESSING_PURPLE_TIER: int(np.count_nonzero(purple_mask)),
            BLESSING_RED_TIER: int(np.count_nonzero(red_mask)),
        }
        best_count = max(counts.values())
        if best_count == 0:
            best_tier = BLESSING_BLUE_TIER
            tied_tiers = False
        else:
            best_tiers = [tier for tier, count in counts.items() if count == best_count]
            tied_tiers = len(best_tiers) > 1
            best_tier = best_tiers[0] if not tied_tiers else None
        best_name = {
            BLESSING_WHITE_TIER: "白",
            BLESSING_GREEN_TIER: "綠",
            BLESSING_BLUE_TIER: "藍",
            BLESSING_PURPLE_TIER: "紫",
            BLESSING_RED_TIER: "紅",
        }.get(best_tier, "平手")
        white = counts[BLESSING_WHITE_TIER]
        green = counts[BLESSING_GREEN_TIER]
        blue = counts[BLESSING_BLUE_TIER]
        purple = counts[BLESSING_PURPLE_TIER]
        red = counts[BLESSING_RED_TIER]
        best_ratio = best_count / total
        color_info = (
            f"白={white}({white / total * 100:.1f}%) "
            f"綠={green}({green / total * 100:.1f}%) "
            f"藍={blue}({blue / total * 100:.1f}%) "
            f"紫={purple}({purple / total * 100:.1f}%) "
            f"紅={red}({red / total * 100:.1f}%) "
            f"總={total} 最高={best_name}({best_ratio:.2%}) "
            f"閾值={MIN_COLOR_PIXEL_RATIO:.2%}"
        )
        if best_ratio < MIN_COLOR_PIXEL_RATIO or tied_tiers:
            return None, color_info
        return best_tier, color_info

    def increment_battle(self) -> None:
        self.current_battle_num += 1
        if self.current_battle_num >= self.max_battle_num:
            self.need_ret_inn = True
        state.logger.debug(
            "increment_battle called, current_battle_num=%d, need_ret_inn=%s",
            self.current_battle_num,
            self.need_ret_inn,
        )
