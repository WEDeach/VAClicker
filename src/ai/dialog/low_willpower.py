import time

from ... import AI
from ...utils.clicker import calculate_click_point, click_at
from ...utils.image import match_template
from ...utils.shared import state
from ...utils.window import get_window_screen


class LowWillpowerDialog(AI):
    def __init__(
        self,
        *,
        target_text: str = "前往地下城",
        button_region: tuple[float, float, float, float] = (0.52, 0.60, 0.56, 0.61),
        delay_post_click: float = 5.0,
    ):
        super().__init__()
        self.target_text = target_text
        self.button_region = button_region
        self.delay_post_click = delay_post_click

    def check(self) -> bool:
        _screen = get_window_screen()
        _match = match_template(
            _screen,
            None,
            0.0,
            False,
            None,
            ocr_check=[(self.target_text, 0)],
            region=self.button_region,
        )
        if not _match:
            return False

        _loc, _ = _match
        _point = calculate_click_point(_loc, (0, 0))
        state.logger.warning("當前意志力較低, 仍選擇前往地下城")
        click_at(_point)
        time.sleep(self.delay_post_click)
        return True
