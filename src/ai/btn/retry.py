import time

from ... import AI
from ...utils.clicker import calculate_click_point, click_at
from ...utils.image import load_template, match_template
from ...utils.text_map import get_text_mapping
from ...utils.window import get_window_screen


class Retry(AI):
    def __init__(self, *, delay_retry: float = 3.0):
        super().__init__()
        self.delay_retry = delay_retry

    def check(self) -> bool:
        _screen = get_window_screen()
        _tmpl, _mask = load_template("BTN_重試.png", grayscale=False)
        _match = match_template(
            _screen,
            _tmpl,
            0.8,
            False,
            _mask,
            ocr_check=[(get_text_mapping("btn.retry"), 0.6)],
            region=(0.33, 0.67, 0.38, 0.60),
        )
        if _match:
            loc, score = _match
            point = calculate_click_point(loc, (_tmpl.shape[1], _tmpl.shape[0]))
            click_at(point)
            time.sleep(self.delay_retry)
            return True
        return False
