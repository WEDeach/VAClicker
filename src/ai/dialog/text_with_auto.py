import time

from ... import AI
from ...utils.clicker import calculate_click_point, click_at
from ...utils.image import load_template, match_template
from ...utils.window import get_window_screen


class TextWithAuto(AI):
    def __init__(self, *, delay_pre_dialog: float = 5.0):
        super().__init__()
        self.delay_pre_dialog = delay_pre_dialog

    def check(self, loop: bool = True) -> bool:
        _screen = get_window_screen()
        _tmpl, _mask = load_template("對話框.png", grayscale=False)
        match_e = match_template(_screen, _tmpl, 0.7, False, _mask)
        if match_e:
            loc, score = match_e
            point = calculate_click_point(loc, (_tmpl.shape[1], _tmpl.shape[0]))
            click_at(point)
            time.sleep(self.delay_pre_dialog)
            if loop:
                self.check(loop=loop)
            return True
        return False
