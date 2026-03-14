import time

import vgamepad

from ... import AI
from ...utils.clicker import calculate_click_point, click_at, click_by_gamepad
from ...utils.image import match_template
from ...utils.shared import state
from ...utils.window import dump4log, get_window_screen
from ..btn._return import Return


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
    ):
        super().__init__()

        self.delay_inn_entry = delay_inn_entry
        self.delay_inn_stay = delay_inn_stay
        self.delay_inn_stay_select = delay_inn_stay_select
        self.delay_inn_stay_confirm = delay_inn_stay_confirm
        self.delay_inn_stay_dialog = delay_inn_stay_dialog
        self.delay_inn_exit = delay_inn_exit
        self.delay_dungeon_entry = delay_dungeon_entry

    def check(self) -> bool:
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
            if _return.need_ret_inn:
                loc, score = _match
                point = calculate_click_point(loc, (0, 0))
                click_at(point)
                time.sleep(self.delay_inn_entry)
                self.check_inn_stay()

            if not _return.need_ret_inn:
                return self.check_dungeon()
        return False

    def check_inn_stay(self) -> bool:
        _screen = get_window_screen()
        _match = match_template(
            _screen,
            None,
            1,
            False,
            None,
            ocr_check=[("宿", 0)],
            region=(0.54, 0.67, 0.41, 0.47),
        )
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
            click_at(point)  # 對話
            time.sleep(self.delay_inn_stay_dialog)
            click_at(point)  # 補給
            time.sleep(self.delay_inn_stay_dialog)
            click_at(point)  # 對話
            time.sleep(self.delay_inn_stay_dialog)

            def _check_new_skill():
                screen_check = get_window_screen()
                match_skill = match_template(
                    screen_check,
                    None,
                    1,
                    False,
                    None,
                    ocr_check=[("點擊後關閉", 0)],
                    region=(0.44, 0.57, 0.61, 0.67),
                )
                if match_skill:
                    loc_skill, score_skill = match_skill
                    point_skill = calculate_click_point(loc_skill, (0, 0))
                    state.logger.debug(
                        "發現新技能對話，點擊關閉 %s (score=%.3f)",
                        point_skill,
                        score_skill,
                    )
                    click_at(point_skill)
                    time.sleep(self.delay_inn_stay_dialog)
                    _check_new_skill()  # loop check
                else:
                    _check_levelup()

            def _check_levelup():
                screen_check = get_window_screen()
                match_levelup = match_template(
                    screen_check,
                    None,
                    1,
                    False,
                    None,
                    ocr_check=[("下一", 0), ("關閉", 0)],
                    region=(0.03, 0.13, 0.90, 0.95),
                )
                if match_levelup:
                    loc_levelup, score_levelup = match_levelup
                    point_levelup = calculate_click_point(loc_levelup, (0, 0))
                    state.logger.debug(
                        "發現升級對話，點擊關閉 %s (score=%.3f)",
                        point_levelup,
                        score_levelup,
                    )
                    click_at(point_levelup)
                    time.sleep(self.delay_inn_stay_dialog)
                    _check_new_skill()  # loop check

            _check_new_skill()
            time.sleep(self.delay_inn_stay_dialog)

            _return = self.find(Return)
            if _return:
                _return.current_battle_num = 1
                _return.need_ret_inn = False
                state.logger.info("已回旅館休息，重置戰鬥次數")
            click_by_gamepad(vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_B)  # 返回
            time.sleep(self.delay_inn_exit)
            return True
        else:
            dump4log(_screen, "找不到確定按鈕")
        return False

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
