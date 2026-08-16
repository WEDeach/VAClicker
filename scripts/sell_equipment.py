"""自動販賣裝備 (Sell equipment automatically).

前置條件:
 - 畫面 1920x1080 | 遊戲亮度 100%
 - 請先手動進入販售介面: 商店 → 出售道具 (畫面上方標題為「出售道具」)
 - 裝備上鎖判定用的參考圖 assets/lock_icon_locked.png / lock_icon_unlocked.png

運作流程:
 1. 辨識販售介面 (標題「出售道具」/「持有清單」)
 2. 依序點擊每個道具列 (點擊即勾選並顯示右側詳細訊息)
 3. 檢查詳細訊息左上方固定位置的鎖頭圖示:
    - 上鎖 → 單擊鎖頭圖示解鎖並驗證
 4. 全部列處理完後點擊下方「出售」按鈕
 5. 彈窗判定:
    - 「包含鎖定中的裝備」失敗 → 關閉彈窗 → 重新掃描一輪 (雙擊歸正勾選狀態)
    - 「出售確認」→ 點擊「確定」
 6. 出售完成等待 5 秒, 回到步驟 1 重複執行
"""

import sys
import time
from collections import defaultdict
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from src import AI
from src.ai.warehouse.parser import find_text_boxes, offset_ocr_boxes
from src.ocr import parse_ocr_boxes
from src.utils.clicker import click_at
from src.utils.shared import state
from src.utils.window import get_window_rect, get_window_screen

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

DEBUG = True

# 販售介面判定文字
TITLE_MARKERS = ("出售道具", "持有清單")
# 重新尋找販售介面時只需涵蓋標題、持有清單與道具列。
SELL_FIND_REGION = (0.33, 0.60, 0.00, 0.80)
# 出售成功後已知仍在販售流程,只需重新解析道具列。
SELL_ROWS_REGION = (0.34, 0.46, 0.20, 0.80)
# 道具列區域 (名稱文字範圍)
SELL_LIST_REGION = (0.36, 0.44, 0.235, 0.75)
SELL_ROW_CLICK_X = 0.42
# 下方「出售」按鈕 (OCR 搜尋區域 + 固定點)
SELL_BUTTON_REGION = (0.40, 0.60, 0.88, 0.99)
SELL_BUTTON_FALLBACK = (0.50, 0.948)
# 詳細訊息面板左上方鎖頭圖示 (位置固定, 由截圖比對得出)
LOCK_REGION = (985, 241, 1018, 275)
LOCK_CLICK = (
    (LOCK_REGION[0] + LOCK_REGION[2]) // 2,
    (LOCK_REGION[1] + LOCK_REGION[3]) // 2,
)
LOCK_MATCH_MAX_DIFF = 30.0
LOCK_SEARCH_RANGE = 12
# 彈窗判定。只 OCR 文字區域,避免再次掃描整個販售畫面。
# 確定按鈕不在此區域,會使用固定 fallback 座標點擊。
SELL_POPUP_REGION = (0.39, 0.61, 0.34, 0.53)
SUCCESS_MARKERS = ("出售確認", "沒問題吧")
FAILURE_MARKERS = ("鎖定中的", "锁定中的")
CONFIRM_TEXTS = ("確定", "确定")
CONFIRM_SUCCESS_FALLBACK = (0.555, 0.614)
CONFIRM_FAILURE_FALLBACK = (0.50, 0.556)
# 時間參數
ROW_CLICK_WAIT = 0.25
LOCK_CHECK_RETRIES = 5
LOCK_CLICK_WAIT = 0.6
SELL_CONFIRM_WAIT = 0.5
SOLD_WAIT = 3.0
UNLOCK_RETRY_LIMIT = 2
MAX_RECOVERY_ROUNDS = 2


SELL_ROW_MERGE_GAP_PX = 40


def _parse_sell_rows(boxes, width: int, height: int) -> list[tuple[str, int]]:
    """Group sell-list OCR boxes into (name, row_y) rows; each row merges the
    item name and its small tag (倉庫) by y proximity."""
    x1, x2, y1, y2 = SELL_LIST_REGION
    items = []
    for text, (bx1, by1, bx2, by2) in boxes:
        text = text.strip()
        if not any(ch.isalpha() for ch in text):
            continue
        cx = (bx1 + bx2) / 2
        cy = (by1 + by2) / 2
        if x1 * width <= cx <= x2 * width and y1 * height <= cy <= y2 * height:
            items.append((cy, text))
    items.sort()
    rows: list[list] = []
    for cy, text in items:
        if rows and cy - rows[-1][1] < SELL_ROW_MERGE_GAP_PX:
            if len(text) > len(rows[-1][0]):
                rows[-1][0] = text
        else:
            rows.append([text, round(cy)])
    return [(name, row_y) for name, row_y in rows]


class _SellState(str, Enum):
    FIND_SELL = "find_sell"
    SELECT_ROWS = "select_rows"
    SELL_CONFIRM = "sell_confirm"
    SOLD_DONE = "sold_done"


class SellEquipment(AI):
    """Non-blocking equipment selling workflow AI."""

    def __init__(self) -> None:
        super().__init__()
        self._state = _SellState.FIND_SELL
        self._rows = []
        self._row_cursor = 0
        self._recovery_rounds = 0
        self._last_unlock_log = 0.0
        self._lock_refs = None
        self._lock_mask = None
        self._wait_until = 0.0
        self._after_sale_scan = False

    # ----- 鎖頭判定 (固定區域像素比對) -----

    def _load_lock_refs(self):
        if self._lock_refs is None:
            base = Path(__file__).parents[1] / "assets"
            ref_locked = cv2.imread(str(base / "lock_icon_locked.png"))
            ref_unlocked = cv2.imread(str(base / "lock_icon_unlocked.png"))
            if ref_locked is None or ref_unlocked is None:
                raise RuntimeError(
                    "缺少鎖頭參考圖 assets/lock_icon_locked.png / lock_icon_unlocked.png"
                )
            self._lock_refs = (ref_locked, ref_unlocked)
            diff = (
                np.abs(ref_locked.astype(np.int16) - ref_unlocked.astype(np.int16)).max(
                    axis=2
                )
                > 25
            )
            self._lock_mask = diff.astype(np.float32)
        return self._lock_refs, self._lock_mask

    def lock_state(self, screen) -> str | None:
        """Return 'locked' / 'unlocked' / None (panel or icon not visible)."""
        (ref_locked, ref_unlocked), mask = self._load_lock_refs()
        x1, y1, x2, y2 = LOCK_REGION
        h, w = y2 - y1, x2 - x1
        best_locked = float("inf")
        best_unlocked = float("inf")
        for dy in range(-LOCK_SEARCH_RANGE, LOCK_SEARCH_RANGE + 1):
            for dx in range(-LOCK_SEARCH_RANGE, LOCK_SEARCH_RANGE + 1):
                yy, xx = y1 + dy, x1 + dx
                if (
                    yy < 0
                    or xx < 0
                    or yy + h > screen.shape[0]
                    or xx + w > screen.shape[1]
                ):
                    continue
                region = screen[yy : yy + h, xx : xx + w].astype(np.float32)
                diff_locked = (
                    np.abs(region - ref_locked.astype(np.float32)).max(axis=2) * mask
                )
                diff_unlocked = (
                    np.abs(region - ref_unlocked.astype(np.float32)).max(axis=2) * mask
                )
                best_locked = min(best_locked, float(diff_locked.sum() / mask.sum()))
                best_unlocked = min(
                    best_unlocked, float(diff_unlocked.sum() / mask.sum())
                )
        if min(best_locked, best_unlocked) > LOCK_MATCH_MAX_DIFF:
            return None
        return "locked" if best_locked < best_unlocked else "unlocked"

    # ----- 主流程 -----

    def _click_screen_point(self, point) -> None:
        left, top, _, _ = get_window_rect()
        click_at((left + point[0], top + point[1]))

    def _click_normalized(self, point, width, height) -> None:
        self._click_screen_point((round(point[0] * width), round(point[1] * height)))

    def _ocr_boxes(self, screen, region=None):
        height, width = screen.shape[:2]
        if region is None:
            crop = screen
            offset = (0, 0)
        else:
            x1f, x2f, y1f, y2f = region
            x1, x2 = round(x1f * width), round(x2f * width)
            y1, y2 = round(y1f * height), round(y2f * height)
            crop = screen[y1:y2, x1:x2]
            offset = (x1, y1)
        if crop.size == 0:
            return []
        return offset_ocr_boxes(parse_ocr_boxes(state.ocr.predict(crop)), offset)

    def check(self) -> bool:
        now = time.monotonic()
        if self._wait_until > now:
            return False
        if self._state is _SellState.SELECT_ROWS:
            _, _, width, height = get_window_rect()
            if self._row_cursor < len(self._rows):
                return self._select_rows(None, width, height)
            point = (
                round(SELL_BUTTON_FALLBACK[0] * width),
                round(SELL_BUTTON_FALLBACK[1] * height),
            )
            state.logger.info("所有列處理完成, 點擊出售按鈕 (固定點 %s)", point)
            self._click_screen_point(point)
            time.sleep(SELL_CONFIRM_WAIT)
            self._state = _SellState.SELL_CONFIRM
            return True
        if self._state is _SellState.SOLD_DONE:
            _, _, width, height = get_window_rect()
            return self._sold_done(None, width, height)
        screen = get_window_screen()
        height, width = screen.shape[:2]
        if self._state is _SellState.FIND_SELL and self._after_sale_scan:
            boxes = self._ocr_boxes(screen, SELL_ROWS_REGION)
            rows = _parse_sell_rows(boxes, width, height)
            if rows:
                self._rows = rows
                self._row_cursor = 0
                self._after_sale_scan = False
                state.logger.info(
                    "出售完成後重新載入道具列, 列數=%d 項目=%s",
                    len(rows),
                    [name for name, _ in rows[:6]],
                )
                self._state = _SellState.SELECT_ROWS
                return True
            state.logger.debug("出售完成後尚未取得道具列, 等待下一輪")
            return False
        boxes = self._ocr_boxes(
            screen,
            (
                SELL_POPUP_REGION
                if self._state is _SellState.SELL_CONFIRM
                else SELL_FIND_REGION
                if self._state is _SellState.FIND_SELL
                else None
            ),
        )

        # 彈窗優先處理 (無論狀態)
        popup = self._handle_popup(boxes, width, height)
        if popup is not None:
            return popup

        if self._state is _SellState.FIND_SELL:
            return self._find_sell(boxes, width, height)
        if self._state is _SellState.SELECT_ROWS:
            return self._select_rows(boxes, width, height)
        if self._state is _SellState.SELL_CONFIRM:
            return self._sell_confirm(boxes, width, height)
        return False

    def _is_sell_ui(self, boxes) -> bool:
        for text, _ in boxes:
            if any(marker in text for marker in TITLE_MARKERS):
                return True
        return False

    def _find_sell(self, boxes, width, height) -> bool:
        if not self._is_sell_ui(boxes):
            return False
        self._rows = _parse_sell_rows(boxes, width, height)
        if not self._rows:
            state.logger.debug("販售清單沒有可處理的道具列")
            return False
        self._row_cursor = 0
        self._after_sale_scan = False
        mode = "恢復" if self._recovery_rounds > 0 else "初次"
        state.logger.info(
            "偵測到販售介面 (%s掃描), 列數=%d 項目=%s",
            mode,
            len(self._rows),
            [name for name, _ in self._rows[:6]],
        )
        self._state = _SellState.SELECT_ROWS
        return True

    def _select_rows(self, boxes, width, height) -> bool:
        if self._row_cursor >= len(self._rows):
            self._click_sell_button(boxes, width, height)
            self._state = _SellState.SELL_CONFIRM
            return True
        name, row_y = self._rows[self._row_cursor]
        self._row_cursor += 1
        point = (round(SELL_ROW_CLICK_X * width), row_y)
        state.logger.info("點擊道具列: 名稱=%s y=%d", name, row_y)
        self._click_screen_point(point)
        lock = None
        for _ in range(LOCK_CHECK_RETRIES):
            time.sleep(ROW_CLICK_WAIT)
            lock = self.lock_state(get_window_screen())
            if lock is not None:
                break
        if lock == "locked":
            state.logger.info("道具已上鎖, 點擊鎖頭解鎖: 名稱=%s", name)
            self._click_screen_point(LOCK_CLICK)
            time.sleep(LOCK_CLICK_WAIT)
            verified = self.lock_state(get_window_screen())
            for attempt in range(UNLOCK_RETRY_LIMIT):
                if verified == "unlocked":
                    break
                state.logger.warning(
                    "解鎖未確認 (%s), 重試 %d/%d",
                    verified,
                    attempt + 1,
                    UNLOCK_RETRY_LIMIT,
                )
                self._click_screen_point(LOCK_CLICK)
                time.sleep(LOCK_CLICK_WAIT)
                verified = self.lock_state(get_window_screen())
            if verified == "unlocked":
                state.logger.info("已解鎖: 名稱=%s", name)
            else:
                state.logger.warning("解鎖失敗: 名稱=%s 狀態=%s", name, verified)
        elif lock == "unlocked":
            if self._recovery_rounds > 0:
                # 失敗恢復輪: 再點一次還原勾選狀態
                self._click_screen_point(point)
                time.sleep(ROW_CLICK_WAIT)
        else:
            state.logger.debug("未偵測到鎖頭圖示 (面板未開啟?): 名稱=%s", name)
        return True

    def _click_sell_button(self, boxes, width, height) -> bool:
        candidates = find_text_boxes(
            boxes,
            "出售",
            region=SELL_BUTTON_REGION,
            width=width,
            height=height,
        )
        exact = [(text, box) for text, box in candidates if text == "出售"]
        if exact:
            _, box = exact[-1]
            point = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
        else:
            point = (
                round(SELL_BUTTON_FALLBACK[0] * width),
                round(SELL_BUTTON_FALLBACK[1] * height),
            )
            state.logger.warning("找不到出售按鈕, 使用固定點 %s", point)
        state.logger.info("點擊出售按鈕: %s", point)
        self._click_screen_point(point)
        time.sleep(SELL_CONFIRM_WAIT)
        return True

    def _handle_popup(self, boxes, width, height) -> bool | None:
        texts = [text for text, _ in boxes]
        if any(marker in text for marker in SUCCESS_MARKERS for text in texts):
            state.logger.info("偵測到出售確認通知, 點擊確定")
            self._click_confirm(boxes, width, height, CONFIRM_SUCCESS_FALLBACK)
            self._state = _SellState.SOLD_DONE
            self._wait_until = time.monotonic() + SOLD_WAIT
            return True
        if any(marker in text for marker in FAILURE_MARKERS for text in texts):
            state.logger.warning(
                "偵測到出售失敗通知 (包含鎖定中的裝備), 關閉後重新掃描"
            )
            self._click_confirm(boxes, width, height, CONFIRM_FAILURE_FALLBACK)
            time.sleep(SELL_CONFIRM_WAIT)
            self._rows = []
            self._row_cursor = 0
            if self._recovery_rounds < MAX_RECOVERY_ROUNDS:
                self._recovery_rounds += 1
                self._state = _SellState.FIND_SELL
            else:
                state.logger.warning("失敗恢復輪已達上限, 回到待機重新偵測")
                self._recovery_rounds = 0
                self._state = _SellState.FIND_SELL
            return True
        return None

    def _click_confirm(self, boxes, width, height, fallback) -> None:
        candidates = []
        for text, box in boxes:
            if any(confirm in text for confirm in CONFIRM_TEXTS):
                candidates.append(box)
        if candidates:
            box = candidates[-1]
            point = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
        else:
            point = (round(fallback[0] * width), round(fallback[1] * height))
            state.logger.warning("找不到確定按鈕, 使用固定點 %s", point)
        self._click_screen_point(point)

    def _sell_confirm(self, boxes, width, height) -> bool:
        # 彈窗可能仍在播放動畫;下一輪會繼續使用小區域 OCR。
        return True

    def _sold_done(self, boxes, width, height) -> bool:
        self._rows = []
        self._row_cursor = 0
        self._recovery_rounds = 0
        self._after_sale_scan = True
        self._state = _SellState.FIND_SELL
        state.logger.info("出售完成, 等待後重新掃描販售介面")
        return True


def entrypoint(core) -> None:
    core.setup(lang="chinese_cht", debug=DEBUG)
    core.register_ai(SellEquipment())
    core.run()


if __name__ == "__main__":
    from src.core import VACore

    entrypoint(VACore())
