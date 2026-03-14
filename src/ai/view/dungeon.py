import time

import vgamepad

from ... import AI
from ...utils.clicker import click_by_gamepad
from ...utils.image import load_template, match_template
from ...utils.shared import state
from ...utils.window import get_window_screen
from ..btn._return import Return
from ..btn.chest_action import ChestAction


class DungeonView(AI):
    def __init__(self, *, delay_need_ret_inn: float = 10.0):
        super().__init__()
        self.delay_need_ret_inn = delay_need_ret_inn

    def check(self) -> bool:
        _screen = get_window_screen()
        _temp, _mask = load_template("郊外_城鎮.png", grayscale=False)
        _match = match_template(
            _screen,
            _temp,
            0.7,
            False,
            _mask,
            region=(0.3, 0.7, 0, 1),
        )
        if _match:
            state.logger.info("已回到郊外...")

            _return = self.find(Return)
            if _return:
                if _return.need_ret_inn:
                    state.logger.info("嘗試回旅館休息...")
                    click_by_gamepad(vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_B)
                    time.sleep(self.delay_need_ret_inn)

            chest_action = self.find(ChestAction)
            if chest_action:
                chest_action.reset_select_index()

            return True
        return False
