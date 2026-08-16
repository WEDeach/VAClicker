import time
from typing import Optional, Tuple

import vgamepad

from ... import AI
from ...ocr import parse_ocr_boxes
from ...utils.clicker import calculate_click_point, click_at, click_by_gamepad
from ...utils.image import match_template
from ...utils.shared import state
from ...utils.window import dump4log, get_window_screen
from ..btn._return import Return

# 對話本文所在區域，避開底部 AUTO / 自動選擇控制項
INN_DIALOG_BODY_REGION = (0.25, 0.75, 0.70, 0.90)
# 底部 AUTO / 自動選擇控制項所在區域
INN_DIALOG_CONTROL_REGION = (0.25, 0.75, 0.90, 1.00)
INN_DIALOG_MAX_STEPS = 24
INN_DIALOG_ABSENT_CONFIRMATIONS = 2
# 相對於視窗截圖的安全備援座標，保證落在控制項區域之外
INN_DIALOG_SAFE_FALLBACK_POINT = (0.50, 0.82)
# 住宿結算時彈出的補給通知所在區域，與技能區域 (0.61-0.67) 略有重疊
INN_DIALOG_SUPPLY_REGION = (0.25, 0.75, 0.45, 0.70)
# 旅館住宿選單「住宿」選項所在區域，與 check_inn_stay() 的判定區域一致
INN_STAY_OPTION_REGION = (0.54, 0.67, 0.41, 0.47)
INN_DIALOG_SUPPLY_KEYWORDS = (
    "補充",
    "補給",
)
INN_DIALOG_CONTROL_KEYWORDS = (
    "AUTO",
    "自動",
    "選擇",
    "选择",
    "點擊後關閉",
    "下一",
    "關閉",
)


def _region_to_pixels(
    region: Tuple[float, float, float, float], width: int, height: int
) -> Tuple[int, int, int, int]:
    x1f, x2f, y1f, y2f = region
    return int(x1f * width), int(x2f * width), int(y1f * height), int(y2f * height)


def _is_control_text(text: str) -> bool:
    return any(keyword in text for keyword in INN_DIALOG_CONTROL_KEYWORDS)


def _is_supply_text(text: str) -> bool:
    return any(keyword in text for keyword in INN_DIALOG_SUPPLY_KEYWORDS)


class TownView(AI):
    def __init__(
        self,
        *,
        delay_inn_entry: float = 5.0,
        delay_inn_stay: float = 2.0,
        delay_inn_stay_select: float = 2.0,
        delay_inn_stay_confirm: float = 10.0,
        delay_inn_stay_dialog: float = 5.0,
        delay_inn_exit: float = 5.0,
        delay_dungeon_entry: float = 5.0,
        enable_dungeon: bool = True,
        inn_dialog_max_steps: int = INN_DIALOG_MAX_STEPS,
        inn_dialog_absent_confirmations: int = INN_DIALOG_ABSENT_CONFIRMATIONS,
        delay_inn_dialog_poll: float = 2.0,
    ):
        super().__init__()

        self.enable_dungeon = enable_dungeon
        self.delay_inn_entry = delay_inn_entry
        self.delay_inn_stay = delay_inn_stay
        self.delay_inn_stay_select = delay_inn_stay_select
        self.delay_inn_stay_confirm = delay_inn_stay_confirm
        self.delay_inn_stay_dialog = delay_inn_stay_dialog
        self.delay_inn_exit = delay_inn_exit
        self.delay_dungeon_entry = delay_dungeon_entry
        self.inn_dialog_max_steps = inn_dialog_max_steps
        self.inn_dialog_absent_confirmations = inn_dialog_absent_confirmations
        self.delay_inn_dialog_poll = delay_inn_dialog_poll
        self._inn_dialog_pending = False

    def check(self) -> bool:
        if self._inn_dialog_pending:
            return self._resume_inn_dialog()
        if self.check_inn():
            return True
        return False

    def check_inn(self) -> bool:
        _screen = get_window_screen()
        _match = match_template(
            _screen,
            None,
            0.8,
            False,
            None,
            ocr_check=[("旅店", 0)],
            region=(0.34, 0.39, 0.37, 0.43),
        )
        if _match:
            _return = self.find(Return)
            if _return is None:
                state.logger.warning("找不到 Return AI，無法處理旅館")
                return False
            if _return.need_ret_inn:
                loc, score = _match
                point = calculate_click_point(loc, (0, 0))
                click_at(point)
                time.sleep(self.delay_inn_entry)
                self.check_inn_stay()

            if not _return.need_ret_inn and self.enable_dungeon:
                return self.check_dungeon()
        return False

    def _find_inn_stay_match(self, screen):
        return match_template(
            screen,
            None,
            1,
            False,
            None,
            ocr_check=[("宿", 0)],
            region=INN_STAY_OPTION_REGION,
        )

    def check_inn_stay(self) -> bool:
        _screen = get_window_screen()
        _match = self._find_inn_stay_match(_screen)
        if _match:
            loc, score = _match
            point = calculate_click_point(loc, (0, 0))
            click_at(point)
            time.sleep(self.delay_inn_stay)
            return self.check_inn_stay_select()
        else:
            dump4log(_screen, "找不到住宿選項")
        return False

    def check_inn_stay_select(self) -> bool:
        # 0.50, 0.60
        # 馬房: 0.38, 0.42
        # 一般房: 0.43, 0.48
        _screen = get_window_screen()
        _match = match_template(
            _screen,
            None,
            1,
            False,
            None,
            ocr_check=[("房", 0)],
            region=(0.50, 0.60, 0.42, 0.47),
        )
        if _match:
            loc, score = _match
            point = calculate_click_point(loc, (0, 0))
            click_at(point)
            time.sleep(self.delay_inn_stay_select)
            return self.check_inn_stay_confirm()
        else:
            dump4log(_screen, "找不到房間選項")
        return False

    def check_inn_stay_confirm(self) -> bool:
        _screen = get_window_screen()
        _match = match_template(
            _screen,
            None,
            1,
            False,
            None,
            ocr_check=[("定", 0)],
            region=(0.50, 0.60, 0.58, 0.62),
        )
        if _match:
            loc, score = _match
            point = calculate_click_point(loc, (0, 0))
            click_at(point)
            time.sleep(self.delay_inn_stay_confirm)
            self._inn_dialog_pending = True
            return self._resume_inn_dialog()
        else:
            dump4log(_screen, "找不到確定按鈕")
        return False

    def _resume_inn_dialog(self) -> bool:
        # 只有在對話確定完全結算後，才允許重置戰鬥次數並按下 B 離開
        if not self._settle_inn_dialog():
            return False

        _return = self.find(Return)
        if _return:
            _return.current_battle_num = 1
            _return.need_ret_inn = False
            state.logger.info("已回旅館休息，重置戰鬥次數")
        self._inn_dialog_pending = False
        click_by_gamepad(vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_B)  # 返回
        time.sleep(self.delay_inn_exit)
        return True

    def _settle_inn_dialog(self) -> bool:
        for _ in range(self.inn_dialog_max_steps):
            state.logger.info("處理旅館對話: %d/%d", _ + 1, self.inn_dialog_max_steps)
            screen = get_window_screen()

            skill_point = self._find_skill_close_point(screen)
            if skill_point:
                state.logger.debug("發現新技能對話，點擊關閉 %s", skill_point)
                click_at(skill_point)
                time.sleep(self.delay_inn_stay_dialog)
                continue

            levelup_point = self._find_levelup_close_point(screen)
            if levelup_point:
                state.logger.debug("發現升級對話，點擊關閉 %s", levelup_point)
                click_at(levelup_point)
                time.sleep(self.delay_inn_stay_dialog)
                continue

            supply_point = self._find_supply_notification_point(screen)
            if supply_point:
                state.logger.debug("發現補給通知，點擊 %s", supply_point)
                click_at(supply_point)
                time.sleep(self.delay_inn_stay_dialog)
                continue

            if self._find_inn_stay_match(screen):
                state.logger.debug("住宿對話完成，返回旅館住宿選項")
                return True

            body_point = self._find_inn_dialog_body_point(screen)
            if body_point:
                click_at(body_point)
                time.sleep(self.delay_inn_stay_dialog)
                continue

            if self._inn_dialog_control_present(screen):
                fallback_point = self._inn_dialog_fallback_point(screen)
                state.logger.debug(
                    "找不到對話本文，改用安全備援座標點擊 %s", fallback_point
                )
                click_at(fallback_point)
                time.sleep(self.delay_inn_stay_dialog)
                continue

            time.sleep(self.delay_inn_dialog_poll)

        state.logger.warning("旅館對話結算逾時，仍有殘留對話或控制項")
        return False

    def _find_skill_close_point(self, screen) -> Optional[Tuple[int, int]]:
        match_skill = match_template(
            screen,
            None,
            1,
            False,
            None,
            ocr_check=[("點擊後關閉", 0)],
            region=(0.44, 0.57, 0.61, 0.67),
        )
        if not match_skill:
            return None
        loc_skill, _ = match_skill
        return calculate_click_point(loc_skill, (0, 0))

    def _find_levelup_close_point(self, screen) -> Optional[Tuple[int, int]]:
        match_levelup = match_template(
            screen,
            None,
            1,
            False,
            None,
            ocr_check=[("下一", 0), ("關閉", 0)],
            region=(0.03, 0.13, 0.90, 0.95),
        )
        if not match_levelup:
            return None
        loc_levelup, _ = match_levelup
        return calculate_click_point(loc_levelup, (0, 0))

    def _find_supply_notification_point(self, screen) -> Optional[Tuple[int, int]]:
        h, w = screen.shape[:2]
        x1, x2, y1, y2 = _region_to_pixels(INN_DIALOG_SUPPLY_REGION, w, h)
        crop = screen[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        boxes = parse_ocr_boxes(state.ocr.predict(crop))
        for text, (bx1, by1, bx2, by2) in boxes:
            if not _is_supply_text(text):
                continue
            abs_x1, abs_y1 = x1 + bx1, y1 + by1
            box_w, box_h = bx2 - bx1, by2 - by1
            return calculate_click_point((abs_x1, abs_y1), (box_w, box_h))
        return None

    def _find_inn_dialog_body_point(self, screen) -> Optional[Tuple[int, int]]:
        h, w = screen.shape[:2]
        x1, x2, y1, y2 = _region_to_pixels(INN_DIALOG_BODY_REGION, w, h)
        crop = screen[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        boxes = parse_ocr_boxes(state.ocr.predict(crop))
        for text, (bx1, by1, bx2, by2) in boxes:
            if _is_control_text(text):
                continue
            abs_x1, abs_y1 = x1 + bx1, y1 + by1
            box_w, box_h = bx2 - bx1, by2 - by1
            return calculate_click_point((abs_x1, abs_y1), (box_w, box_h))
        return None

    def _inn_dialog_control_present(self, screen) -> bool:
        h, w = screen.shape[:2]
        x1, x2, y1, y2 = _region_to_pixels(INN_DIALOG_CONTROL_REGION, w, h)
        crop = screen[y1:y2, x1:x2]
        if crop.size == 0:
            return False

        boxes = parse_ocr_boxes(state.ocr.predict(crop))
        return any(_is_control_text(text) for text, _ in boxes)

    def _inn_dialog_fallback_point(self, screen) -> Tuple[int, int]:
        h, w = screen.shape[:2]
        fx, fy = INN_DIALOG_SAFE_FALLBACK_POINT
        return calculate_click_point((int(fx * w), int(fy * h)), (0, 0))

    def check_dungeon(self) -> bool:
        _screen = get_window_screen()
        _match = match_template(
            _screen,
            None,
            1,
            False,
            None,
            ocr_check=[("郊外", 0)],
            region=(0.59, 0.65, 0.54, 0.57),
        )
        if _match:
            loc, score = _match
            point = calculate_click_point(loc, (0, 0))
            click_at(point)
            time.sleep(self.delay_dungeon_entry)
            return True
        else:
            state.logger.warning("找不到郊外選項，無法回郊外")
            dump4log(_screen, "找不到郊外選項")
        return False
