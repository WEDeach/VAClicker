import time

from ... import AI
from ...utils.clicker import calculate_click_point, click_at
from ...utils.image import load_template, match_template
from ...utils.window import get_window_screen


class Return(AI):
    def __init__(
        self,
        *,
        need_ret_inn: bool = False,
        current_battle_num: int = 1,
        max_battle_num: int = 7,
        delay_return: float = 5.0,
    ):
        super().__init__()
        self.need_ret_inn = need_ret_inn
        self.current_battle_num = current_battle_num
        self.max_battle_num = max_battle_num
        self.delay_return = delay_return

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
            self.current_battle_num += 1
            if self.current_battle_num >= self.max_battle_num:
                self.need_ret_inn = True
            return True
        return False
