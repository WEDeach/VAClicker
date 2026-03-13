import re
import time

from ... import AI
from ...utils.clicker import calculate_click_point, click_at
from ...utils.image import load_template, match_template
from ...utils.shared import state
from ...utils.text_map import get_text_mapping
from ...utils.window import get_window_rect, get_window_screen


class ChestAction(AI):
    def __init__(
        self,
        *,
        select_index: int = 0,
        select_retry_cooltime: float = 12.0,
        delay_chest_unlock: float = 4.0,
    ):
        super().__init__()
        self.select_index = select_index
        self._select_index = select_index
        self.select_retry_cooltime = select_retry_cooltime
        self.last_open_time = 0
        self.last_unlock_time = 0
        self.delay_chest_unlock = delay_chest_unlock

    def reset_select_index(self):
        self._select_index = self.select_index

    def check(self) -> bool:
        _screen = get_window_screen()
        _match = match_template(
            _screen,
            None,
            1,
            False,
            None,
            ocr_check=[(re.compile(get_text_mapping("btn.chest.action.open")), 0)],
            region=(0.35, 0.65, 0.75, 0.88),
        )
        if _match:
            _loc, _ = _match
            left, top, width, height = get_window_rect()
            _x = int(left + width // 2)
            _y = int(top + height * 0.81)
            _point = (_x, _y)
            click_at(_point)
            return True
        return self.select_opener()

    def select_opener(self) -> bool:
        _screen = get_window_screen()
        _temp, _mask = load_template("寶箱_開啟選擇.png", grayscale=True)
        _match = match_template(
            _screen,
            _temp,
            0.7,
            True,
            _mask,
            ocr_check=[("要讓誰開啟", 0.8)],
        )
        if _match:
            loc, score = _match
            if score == 1.0:
                state.logger.warning(
                    "[select_opener] OCR準確度過高，可能是誤判，放棄點擊"
                )
                return False

            now = time.time()
            idx = self._select_index
            state.logger.debug(
                "[select_opener] SELECT_INDEX=#%d at %.3f seconds,"
                " SELECT_RETRY_COOLTIME=%.3f",
                idx,
                now - self.last_open_time,
                self.select_retry_cooltime,
            )

            if (
                self.last_open_time > 0
                and (now - self.last_open_time) <= self.select_retry_cooltime
            ):
                idx = (idx + 1) % 6
                self._select_index = idx

            self.last_open_time = now

            left, top, _, _ = get_window_rect()
            th, tw = _temp.shape[:2]
            container_abs = (left + loc[0], top + loc[1], tw, th)

            cx, cy, cw, ch = container_abs
            gx = cx + int(0.05 * cw)
            gy = cy + int(0.35 * ch)
            gw = int((0.95 - 0.05) * cw)
            gh = int((0.75 - 0.35) * ch)

            gap_x = 0
            gap_y = 0
            cols = 3
            rows = 2

            # 總間距寬高扣掉後，再均分給每顆按鈕
            btn_w = max(1, (gw - gap_x * (cols - 1)) // cols)
            btn_h = max(1, (gh - gap_y * (rows - 1)) // rows)

            rects = []
            for row in range(rows):
                for col in range(cols):
                    rects.append(
                        (
                            gx + col * (btn_w + gap_x),
                            gy + row * (btn_h + gap_y),
                            btn_w,
                            btn_h,
                        )
                    )

            if 0 <= idx < len(rects):
                bx, by, bw, bh = rects[idx]
                click_pt = (bx + bw // 2, by + bh // 2)
                click_at(click_pt)
                state.logger.info("開啟寶箱 (By #%d)", idx)
            return True
        return self.chest_unlock()

    def chest_unlock(self) -> bool:
        _screen = get_window_screen()
        _temp, _mask = load_template("寶箱_解除.png", grayscale=True)
        _match = match_template(
            _screen,
            _temp,
            0.7,
            True,
            _mask,
        )
        if _match:
            loc, score = _match
            point = calculate_click_point(loc, (_temp.shape[1], _temp.shape[0]))
            click_at(point)
            time.sleep(self.delay_chest_unlock)
            self.last_unlock_time = time.time()
        return False
