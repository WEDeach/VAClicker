import time

import cv2

from ... import AI
from ...ocr import parse_ocr_boxes
from ...utils.clicker import calculate_click_point, click_at
from ...utils.shared import state
from ...utils.text_map import get_text_mapping
from ...utils.window import get_window_screen

PLAYER_DEATH_REGION = (0.30, 0.70, 0.40, 0.65)
REVIVE_CHARGE_REGION = (0.35, 0.65, 0.00, 0.15)
REVIVE_CHARGE_HSV_LOWER = (10, 80, 120)
REVIVE_CHARGE_HSV_UPPER = (45, 255, 255)
REVIVE_CHARGE_MIN_WIDTH = 0.02
REVIVE_CHARGE_MAX_WIDTH = 0.04
REVIVE_CHARGE_MIN_HEIGHT = 0.03
REVIVE_CHARGE_MAX_HEIGHT = 0.07
REVIVE_CHARGE_MIN_AREA = 0.0004
REVIVE_CHARGE_MAX_AREA = 0.0012
PLAYER_DEATH_TEXT_KEYS = (
    "view.death.player.revive",
    "view.death.player.accept",
)


class DeathView(AI):
    def __init__(
        self,
        *,
        allow_revive: bool = False,
        delay_revive: float = 10.0,
        min_revive_charges: int = 1,
        player_death_region=PLAYER_DEATH_REGION,
        revive_charge_region=REVIVE_CHARGE_REGION,
    ):
        super().__init__()
        self.allow_revive = allow_revive
        self.delay_revive = delay_revive
        self.min_revive_charges = min_revive_charges
        self.player_death_region = player_death_region
        self.revive_charge_region = revive_charge_region

    def check(self) -> bool:
        screen = get_window_screen()
        revive_box = self._find_player_death_revive_box(screen)
        if revive_box is not None and self.allow_revive:
            charges = self.revive_charge_count(screen)
            state.logger.debug(
                "玩家死亡, 復活之火: %d",
                charges,
            )
            if charges < self.min_revive_charges:
                state.logger.warning(
                    ("復活之火不足 (%d < %d)，不執行再起"),
                    charges,
                    self.min_revive_charges,
                )
            else:
                click_at(calculate_click_point(revive_box[:2], revive_box[2:]))
                state.logger.info("已使用復活之火，等待 %.1f 秒", self.delay_revive)
                time.sleep(self.delay_revive)
            return True
        return False

    def _find_player_death_revive_box(self, screen):
        player_death_keywords = tuple(
            get_text_mapping(key) for key in PLAYER_DEATH_TEXT_KEYS
        )
        boxes = self._ocr_boxes(screen, self.player_death_region)
        revive_box = None
        found_keywords = set()
        for text, box in boxes:
            for keyword in player_death_keywords:
                if keyword in text:
                    found_keywords.add(keyword)
            if player_death_keywords[0] in text:
                revive_box = box
        if all(keyword in found_keywords for keyword in player_death_keywords):
            return revive_box
        return None

    def is_character_death(self) -> bool:
        # TODO: Implement character death detection
        return False

    def _ocr_boxes(self, screen, region):
        height, width = screen.shape[:2]
        x1f, x2f, y1f, y2f = region
        x1, x2 = int(x1f * width), int(x2f * width)
        y1, y2 = int(y1f * height), int(y2f * height)
        crop = screen[y1:y2, x1:x2]
        if state.ocr is None or crop.size == 0:
            return []
        return [
            (text, (x1 + bx1, y1 + by1, bx2 - bx1, by2 - by1))
            for text, (bx1, by1, bx2, by2) in parse_ocr_boxes(state.ocr.predict(crop))
        ]

    def revive_charge_count(self, screen) -> int:
        height, width = screen.shape[:2]
        x1f, x2f, y1f, y2f = self.revive_charge_region
        x1, x2 = int(x1f * width), int(x2f * width)
        y1, y2 = int(y1f * height), int(y2f * height)
        crop = screen[y1:y2, x1:x2]
        if crop.size == 0:
            return 0
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, REVIVE_CHARGE_HSV_LOWER, REVIVE_CHARGE_HSV_UPPER)
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        return sum(
            REVIVE_CHARGE_MIN_WIDTH * width
            <= component_width
            <= REVIVE_CHARGE_MAX_WIDTH * width
            and REVIVE_CHARGE_MIN_HEIGHT * height
            <= component_height
            <= REVIVE_CHARGE_MAX_HEIGHT * height
            and REVIVE_CHARGE_MIN_AREA * width * height
            <= area
            <= REVIVE_CHARGE_MAX_AREA * width * height
            for _, _, component_width, component_height, area in stats[1:count]
        )
