import time

from .... import AI
from ....utils.clicker import calculate_click_point, click_at
from ....utils.image import load_template, match_template
from ....utils.shared import state
from ....utils.window import get_window_rect, get_window_screen

v1 = "路徑_寶箱.png"
v2 = "路徑_寶箱_v2.png"


class ChestPath(AI):
    def __init__(
        self, *, delay_not_found: float = 15.0, delay_check_not_found: float = 1.2
    ):
        super().__init__()
        self.delay_not_found = delay_not_found
        self.delay_check_not_found = delay_check_not_found
        self.not_found_end_at = 0

    def check(self, *, check_only: bool = False) -> bool:
        # 有冷卻 則只點一次路標, 別問為啥 因為我爽
        if not check_only and time.time() < self.not_found_end_at:
            return False

        _screen = get_window_screen()
        _tmpl, _mask = load_template(v2, grayscale=True)
        _match = match_template(
            _screen, _tmpl, 0.9, True, _mask, region=(0.91, 0.99, 0.15, 0.25)
        )
        if _match:
            if check_only:
                return True
            loc, score = _match
            point = calculate_click_point(loc, (_tmpl.shape[1], _tmpl.shape[0]))
            click_at(point)

            self.not_found_path()
            return True
        return False

    def not_found_path(self) -> bool:
        # TODO: ASYNC TO CHECK
        time.sleep(self.delay_check_not_found)  # update screen
        _screen = get_window_screen()
        _match = match_template(
            _screen,
            None,
            1,
            False,
            None,
            ocr_check=[("找不到寶箱", 0), ("找不到前往目的地的路線", 0)],
            region=(0.36, 0.65, 0.41, 0.57),
        )
        if _match:
            state.logger.debug("找不到路線，點擊路標")
            self.not_found_end_at = time.time() + self.delay_not_found
            left, top, width, height = get_window_rect()
            _x = int(left + width * 0.95)
            _y = int(top + height * 0.22)
            _point = (_x, _y)
            click_at(_point)
            return True
        return False
