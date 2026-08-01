from typing import Tuple

from ... import AI
from ...utils.clicker import calculate_click_point, click_at
from ...utils.image import load_template, match_template
from ...utils.window import get_window_screen


class AutoMode(AI):
    def __init__(
        self,
        *,
        template: str = "ICON_自動.png",
        threshold: float = 0.8,
        region: Tuple[float, float, float, float] = (0.62, 0.66, 0.66, 0.73),
    ):
        super().__init__()
        self.template = template
        self.threshold = threshold
        self.region = region

    def check(self) -> bool:
        _screen = get_window_screen()
        _tmpl, _mask = load_template(self.template, grayscale=False)
        _match = match_template(
            _screen,
            _tmpl,
            self.threshold,
            False,
            _mask,
            region=self.region,
        )
        if _match:
            loc, score = _match
            point = calculate_click_point(loc, (_tmpl.shape[1], _tmpl.shape[0]))
            click_at(point)
            return True
        return False
