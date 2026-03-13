import time

import vgamepad as vg

from ... import AI
from ...utils.clicker import calculate_click_point, click_at, click_by_gamepad
from ...utils.image import load_template, match_template
from ...utils.window import get_window_screen


class Recovery(AI):
    def __init__(
        self, *, delay_recovery: float = 8.0, delay_character_exit: float = 2.0
    ):
        super().__init__()
        self.delay_recovery = delay_recovery
        self.delay_character_exit = delay_character_exit

    def check(self) -> bool:
        _screen = get_window_screen()
        _tmpl, _mask = load_template("BTN_回復.png", grayscale=False)
        _match = match_template(_screen, _tmpl, 0.8, False, _mask)
        if _match:
            loc, score = _match
            point = calculate_click_point(loc, (_tmpl.shape[1], _tmpl.shape[0]))
            click_at(point)
            time.sleep(self.delay_recovery)
            click_by_gamepad(vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
            time.sleep(self.delay_character_exit)
            click_by_gamepad(vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
            return True
        return False
