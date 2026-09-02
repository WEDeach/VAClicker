"""Backpack organizer workflow AI (core, script-agnostic).

Flow (all regions normalized to window size):

1. Town screen: ensure the bottom party panel is expanded by clicking the
   toggle icon (two arrows) at the bottom-left when it is not already open.
2. Click the first non-empty character slot (up to 6 slots, some may be empty).
3. On the character detail screen, click the "背包" (backpack) tab.
4. On the backpack screen, look for the "其它"/"其他" (other) category:
   - When found and ``allow_deposit_other`` is enabled, click the first item
     of that category, then "放入倉庫" (deposit), wait, and repeat until the
     category has no more items.
   - When absent (or deposits are disabled), switch to the next character by
     dragging right-to-left across the backpack UI, verifying the top name
     changed. Cached names prevent reprocessing (max 6 characters).
5. When all characters are done, press B to exit and clear the pending flag.

The workflow holds the AI loop (returns True) while it is active so other AIs
(such as the recovery logic that shares the character detail screen) cannot
interfere. Town entry is skipped while ``Return.need_ret_inn`` is set so it
never conflicts with the inn-return flow.
"""

import time
from enum import Enum

import vgamepad as vg

from ... import AI
from ...ocr import parse_ocr_boxes
from ...utils.clicker import (
    click_at,
    click_by_gamepad,
    drag_hold,
)
from ...utils.shared import state
from ...utils.window import dump4log, get_window_rect, get_window_screen
from ..btn._return import Return

# Town / party panel -----------------------------------------------------
# 與 TownView 相同的城鎮確認區域(旅店)
TOWN_CONFIRM_REGION = (0.34, 0.39, 0.37, 0.43)
TOWN_CONFIRM_TEXT = "旅店"
# 左下角隊伍資訊展開/收合 ICON(兩個箭頭), x 固定、y 隨視窗高度微變
PARTY_TOGGLE_POINT = (0.375, 0.945)
# 展開後的隊伍面板(角色槽位名稱 / Lv / HP 等)
PARTY_PANEL_REGION = (0.30, 0.70, 0.75, 1.00)
PARTY_EXPANDED_KEYWORDS = ("Lv.", "HP", "MP", "SP")
# 角色槽位名稱排序用(排除欄位性文字)
PARTY_SLOT_EXCLUDE_PREFIXES = ("Lv.", "HP", "MP", "SP", "心", "G", "X", "□")
# 城鎮畫面已知文字(避免誤判為角色槽)
TOWN_KNOWN_TEXTS = (
    "旅店",
    "道具店",
    "寺院",
    "荒屋",
    "打鐵鋪",
    "寶石商",
    "郊外",
    "冒險者公會",
    "EVENT",
    "Discount",
    "冒险者公会",
    "冒险者",
)

# Character detail / backpack -------------------------------------------
# 角色詳細畫面頂部名稱(切換判定用)
CHAR_NAME_TOP_REGION = (0.40, 0.60, 0.00, 0.08)
# 角色詳細畫面底部「背包」按鈕
BACKPACK_BTN_REGION = (0.55, 0.65, 0.90, 0.98)
BACKPACK_BTN_TEXT = "背包"
# 背包畫面標題
BACKPACK_TITLE_REGION = (0.45, 0.55, 0.10, 0.16)
BACKPACK_TITLE_TEXT = "背包"
# 背包畫面底部「道具清單」標題(輔助確認)
ITEM_LIST_TITLE_REGION = (0.45, 0.55, 0.80, 0.90)
ITEM_LIST_TITLE_TEXT = "道具清單"
# 左側類別欄位列
CATEGORY_LIST_REGION = (0.28, 0.45, 0.20, 0.72)
OTHER_CATEGORY_TEXTS = ("其它", "其他")
# 其他類別下的第一個物品(其他欄位下方)
OTHER_ITEM_REGION = (0.35, 0.65, 0.46, 0.75)
CATEGORY_EXCLUDE_TEXTS = (
    "其它",
    "其他",
    "消耗品",
    "裝備中",
    "装備中",
    "道具",
    "裝備",
    "装備",
)
# 物品資訊彈窗「放入倉庫」按鈕
DEPOSIT_BTN_REGION = (0.45, 0.55, 0.78, 0.84)
DEPOSIT_BTN_TEXT = "放入倉庫"

# Character switching ----------------------------------------------------
SWITCH_DRAG_START = (0.85, 0.50)
SWITCH_DRAG_END = (0.15, 0.50)
MAX_CHARACTERS = 6


class _State(str, Enum):
    IDLE = "idle"
    ENSURE_EXPANDED = "ensure_expanded"
    SELECT_CHARACTER = "select_character"
    WAIT_DETAIL = "wait_detail"
    OPEN_BACKPACK = "open_backpack"
    WAIT_BACKPACK = "wait_backpack"
    CHECK_OTHER = "check_other"
    WAIT_ITEM_DETAIL = "wait_item_detail"
    WAIT_DEPOSIT = "wait_deposit"
    SWITCH_CHARACTER = "switch_character"
    VERIFY_SWITCH = "verify_switch"
    FINISH = "finish"


class InventoryOrganizer(AI):
    """Organize backpack "other" items into the warehouse, one character at a
    time, from the town screen. One state transition per ``check()``.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        allow_deposit_other: bool = False,
        need_organize: bool = False,
        delay_toggle_wait: float = 2.0,
        delay_detail_wait: float = 2.0,
        delay_backpack_wait: float = 2.0,
        delay_item_wait: float = 2.0,
        delay_deposit: float = 5.0,
        delay_switch_wait: float = 2.0,
        delay_action: float = 0.5,
        retry_limit: int = 5,
        max_check_duration: float = 180.0,
    ):
        super().__init__()
        self.enabled = enabled
        self.allow_deposit_other = allow_deposit_other
        self.delay_toggle_wait = delay_toggle_wait
        self.delay_detail_wait = delay_detail_wait
        self.delay_backpack_wait = delay_backpack_wait
        self.delay_item_wait = delay_item_wait
        self.delay_deposit = delay_deposit
        self.delay_switch_wait = delay_switch_wait
        self.delay_action = delay_action
        self.retry_limit = retry_limit
        self.max_check_duration = max_check_duration
        self.need_organize = need_organize
        self._state = _State.IDLE
        self._seen_names: set[str] = set()
        self._retries = 0
        self._pre_switch_name = None

    # -- public API --------------------------------------------------------

    def reset(self) -> None:
        """Force a fresh run when the pending flag is set (e.g. after the inn
        stay completes)."""
        self._state = _State.IDLE
        self._seen_names.clear()
        self._retries = 0

    # -- main loop ----------------------------------------------------------

    def check(self) -> bool:
        """Advance the workflow, chaining multiple state transitions in one
        call until an action requires an on-screen confirmation that has not
        happened yet (then yield back to the main loop)."""
        if not self.enabled or not self.need_organize:
            return False
        if self._state is _State.IDLE:
            if not self._in_town():
                return False
            self._state = _State.ENSURE_EXPANDED
            state.logger.info("[背包整理] 開始整理背包")
        started = time.monotonic()
        while self._state is not _State.IDLE:
            if time.monotonic() - started >= self.max_check_duration:
                state.logger.warning(
                    "[背包整理] 單次 check 超過 %.0f 秒，讓出控制權",
                    self.max_check_duration,
                )
                return True
            if not self._advance():
                break
        return self._state is not _State.IDLE

    def _advance(self) -> bool:
        """Run one state handler. Return True to keep chaining inside the same
        ``check()`` call, False to yield back to the main loop (waiting for an
        on-screen confirmation) or because the workflow finished."""
        handler = {
            _State.ENSURE_EXPANDED: self._ensure_expanded,
            _State.SELECT_CHARACTER: self._select_character,
            _State.WAIT_DETAIL: self._wait_detail,
            _State.OPEN_BACKPACK: self._open_backpack,
            _State.WAIT_BACKPACK: self._wait_backpack,
            _State.CHECK_OTHER: self._check_other,
            _State.WAIT_ITEM_DETAIL: self._wait_item_detail,
            _State.WAIT_DEPOSIT: self._wait_deposit,
            _State.SWITCH_CHARACTER: self._switch_character,
            _State.VERIFY_SWITCH: self._verify_switch,
            _State.FINISH: self._finish,
        }[self._state]
        return handler()

    # -- state handlers ------------------------------------------------------

    def _ensure_expanded(self) -> bool:
        if self._party_expanded():
            self._state = _State.SELECT_CHARACTER
            return True
        if self._retries >= self.retry_limit:
            state.logger.warning("[背包整理] 隊伍資訊展開逾時")
            return self._abort()
        self._retries += 1
        state.logger.debug("[背包整理] 點擊展開隊伍資訊 ICON (嘗試 %d)", self._retries)
        self._click_normalized(PARTY_TOGGLE_POINT)
        time.sleep(self.delay_toggle_wait)
        # 等待畫面更新，讓出控制權，下輪再確認
        return False

    def _select_character(self) -> bool:
        name = self._first_party_slot_name()
        if name is None:
            state.logger.warning("[背包整理] 找不到可點擊的角色槽位")
            return self._abort()
        state.logger.debug("[背包整理] 點擊角色: %s", name)
        self._click_box(name[1])
        self._seen_names.add(name[0])
        self._retries = 0
        self._state = _State.WAIT_DETAIL
        time.sleep(self.delay_detail_wait)
        return True

    def _wait_detail(self) -> bool:
        if self._find_text_point(BACKPACK_BTN_REGION, BACKPACK_BTN_TEXT) is not None:
            self._state = _State.OPEN_BACKPACK
            return True
        if self._retries >= self.retry_limit:
            state.logger.warning("[背包整理] 角色詳細畫面逾時")
            dump4log(get_window_screen(), "背包整理_角色詳細逾時")
            return self._abort()
        self._retries += 1
        time.sleep(self.delay_action)
        # 畫面尚未轉換完成，讓出控制權等待下輪
        return False

    def _open_backpack(self) -> bool:
        point = self._find_text_point(BACKPACK_BTN_REGION, BACKPACK_BTN_TEXT)
        if point is None:
            self._state = _State.WAIT_DETAIL
            return True
        state.logger.debug("[背包整理] 點擊背包按鈕")
        self._click_ocr_point(point)
        self._retries = 0
        self._state = _State.WAIT_BACKPACK
        time.sleep(self.delay_backpack_wait)
        return True

    def _wait_backpack(self) -> bool:
        if self._is_backpack_screen():
            self._state = _State.CHECK_OTHER
            return True
        if self._retries >= self.retry_limit:
            state.logger.warning("[背包整理] 背包畫面逾時")
            dump4log(get_window_screen(), "背包整理_背包畫面逾時")
            return self._abort()
        self._retries += 1
        time.sleep(self.delay_action)
        # 背包畫面尚未出現，讓出控制權等待下輪
        return False

    def _check_other(self) -> bool:
        other = self._find_category(OTHER_CATEGORY_TEXTS)
        if other is None:
            state.logger.debug("[背包整理] 背包無「其它」欄位，切換角色")
            self._state = _State.SWITCH_CHARACTER
            return True
        if not self.allow_deposit_other:
            state.logger.debug("[背包整理] 「其它」欄位存在但未允許放入倉庫，切換角色")
            self._state = _State.SWITCH_CHARACTER
            return True
        item = self._first_other_item(other)
        if item is None:
            state.logger.debug("[背包整理] 「其它」欄位下無物品，切換角色")
            self._state = _State.SWITCH_CHARACTER
            return True
        state.logger.debug("[背包整理] 點擊其它物品: %s", item[0])
        self._click_box(item[1])
        self._retries = 0
        self._state = _State.WAIT_ITEM_DETAIL
        time.sleep(self.delay_item_wait)
        return True

    def _wait_item_detail(self) -> bool:
        point = self._find_text_point(DEPOSIT_BTN_REGION, DEPOSIT_BTN_TEXT)
        if point is None:
            if self._retries >= self.retry_limit:
                state.logger.warning("[背包整理] 物品資訊逾時")
                dump4log(get_window_screen(), "背包整理_物品資訊逾時")
                return self._abort()
            self._retries += 1
            time.sleep(self.delay_action)
            # 物品資訊彈窗尚未出現，讓出控制權等待下輪
            return False
        state.logger.debug("[背包整理] 點擊放入倉庫")
        self._click_ocr_point(point)
        self._retries = 0
        self._state = _State.WAIT_DEPOSIT
        time.sleep(self.delay_deposit)
        return True

    def _wait_deposit(self) -> bool:
        # 放入倉庫後重新確認背包畫面，然後周而復始(背包 > 其他 > 放入倉庫)
        if self._is_backpack_screen():
            self._state = _State.CHECK_OTHER
            return True
        if self._retries >= self.retry_limit:
            state.logger.warning("[背包整理] 放入倉庫後背包畫面未確認")
            dump4log(get_window_screen(), "背包整理_放入後未確認")
            return self._abort()
        self._retries += 1
        time.sleep(self.delay_action)
        # 畫面尚未回到背包，讓出控制權等待下輪
        return False

    def _switch_character(self) -> bool:
        if len(self._seen_names) >= MAX_CHARACTERS:
            state.logger.info("[背包整理] 已處理 %d 個角色", len(self._seen_names))
            self._state = _State.FINISH
            return True
        current_name = self._read_top_character_name()
        if current_name is None:
            if self._retries >= self.retry_limit:
                state.logger.warning("[背包整理] 無法讀取角色名稱，放棄切換")
                return self._abort()
            self._retries += 1
            time.sleep(self.delay_action)
            return False
        self._pre_switch_name = current_name
        left, top, width, height = get_window_rect()
        start = (
            left + int(SWITCH_DRAG_START[0] * width),
            top + int(SWITCH_DRAG_START[1] * height),
        )
        end = (
            left + int(SWITCH_DRAG_END[0] * width),
            top + int(SWITCH_DRAG_END[1] * height),
        )
        state.logger.debug("[背包整理] 拖曳切換角色 (當前: %s)", current_name)
        drag_hold(start, end, hold_duration=0.3, move_duration=1.0)
        self._state = _State.VERIFY_SWITCH
        time.sleep(self.delay_switch_wait)
        return True

    def _verify_switch(self) -> bool:
        new_name = self._read_top_character_name()
        if new_name is None:
            if self._retries >= self.retry_limit:
                state.logger.warning("[背包整理] 切換後無法讀取角色名稱")
                return self._abort()
            self._retries += 1
            time.sleep(self.delay_action)
            return False
        if new_name == getattr(self, "_pre_switch_name", None):
            # 拖曳失敗(名稱未變)，重試切換
            if self._retries >= self.retry_limit:
                state.logger.warning("[背包整理] 切換角色拖曳重試耗盡: %s", new_name)
                return self._abort()
            self._retries += 1
            state.logger.debug(
                "[背包整理] 拖曳未切換角色，重試 (嘗試 %d)", self._retries
            )
            self._state = _State.SWITCH_CHARACTER
            return True
        if new_name in self._seen_names:
            state.logger.info("[背包整理] 角色 %s 已處理過，結束", new_name)
            self._state = _State.FINISH
            return True
        self._seen_names.add(new_name)
        state.logger.debug("[背包整理] 已切換到角色: %s", new_name)
        self._retries = 0
        # 切換後確認畫面:仍在背包畫面則繼續檢查，否則重新點背包
        if self._is_backpack_screen():
            self._state = _State.CHECK_OTHER
        else:
            self._state = _State.OPEN_BACKPACK
        time.sleep(self.delay_backpack_wait)
        return True

    def _finish(self) -> bool:
        state.logger.info("[背包整理] 整理完成，按 B 退出")
        click_by_gamepad(vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
        time.sleep(self.delay_action)
        self.need_organize = False
        self._state = _State.IDLE
        self._seen_names.clear()
        self._retries = 0
        return False

    def _abort(self) -> bool:
        state.logger.warning("[背包整理] 流程中止，按 B 退出")
        try:
            dump4log(get_window_screen(), "背包整理_中止")
        except Exception:
            pass
        click_by_gamepad(vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
        time.sleep(self.delay_action)
        self.need_organize = False
        self._state = _State.IDLE
        self._seen_names.clear()
        self._retries = 0
        return False

    # -- OCR / screen helpers ------------------------------------------------

    def _screen_boxes(self, region):
        screen = get_window_screen()
        if screen is None or screen.size == 0:
            return []
        height, width = screen.shape[:2]
        x1f, x2f, y1f, y2f = region
        x1, x2 = int(x1f * width), int(x2f * width)
        y1, y2 = int(y1f * height), int(y2f * height)
        crop = screen[y1:y2, x1:x2]
        if crop.size == 0:
            return []
        boxes = []
        for text, (bx1, by1, bx2, by2) in parse_ocr_boxes(state.ocr.predict(crop)):
            boxes.append((text, (x1 + bx1, y1 + by1, x1 + bx2, y1 + by2)))
        return boxes

    def _find_text_point(self, region, keyword):
        for text, (x1, y1, x2, y2) in self._screen_boxes(region):
            if keyword in text:
                return ((x1 + x2) // 2, (y1 + y2) // 2)
        return None

    def _in_town(self) -> bool:
        return self._find_text_point(TOWN_CONFIRM_REGION, TOWN_CONFIRM_TEXT) is not None

    def _party_expanded(self) -> bool:
        boxes = self._screen_boxes(PARTY_PANEL_REGION)
        for text, _ in boxes:
            if any(kw in text for kw in PARTY_EXPANDED_KEYWORDS):
                return True
        return False

    def _first_party_slot_name(self):
        """Return the first non-empty character slot (top-left-most)."""
        candidates = []
        for text, (x1, y1, x2, y2) in self._screen_boxes(PARTY_PANEL_REGION):
            text = text.strip()
            if len(text) < 2:
                continue
            if any(text.startswith(p) for p in PARTY_SLOT_EXCLUDE_PREFIXES):
                continue
            if any(kw in text for kw in ("HP", "MP", "SP", "Lv.")):
                continue
            if any(kw in text for kw in TOWN_KNOWN_TEXTS):
                continue
            # 名稱:優先取最短的非數字文字(過濾 OCR 誤讀的數字/數值/符號)
            if text.isdigit():
                continue
            if "/" in text or "×" in text or "x" in text:
                continue
            candidates.append((y1, x1, text, (x1, y1, x2, y2)))
        if not candidates:
            return None
        candidates.sort(key=lambda c: (c[0], c[1]))
        return candidates[0][2], candidates[0][3]

    def _read_top_character_name(self):
        """Read the character name at the top of the detail/backpack screen."""
        boxes = self._screen_boxes(CHAR_NAME_TOP_REGION)
        # 1) 名稱與 Lv. 在同一 box: 「不忘初心Lv.50MAX」
        for text, _ in boxes:
            text = text.strip()
            if "Lv." in text:
                name = text.split("Lv.")[0].strip()
                if name:
                    return name
        # 2) 名稱被拆開: 獨立的「不忘初心」box
        for text, _ in boxes:
            text = text.strip()
            if not text:
                continue
            if text.startswith(("Lv.", "Exp", "ExD")):
                continue
            if any(ch.isdigit() for ch in text):
                continue
            if len(text) < 2:
                continue
            return text
        return None

    def _is_backpack_screen(self) -> bool:
        if (
            self._find_text_point(BACKPACK_TITLE_REGION, BACKPACK_TITLE_TEXT)
            is not None
        ):
            return True
        if (
            self._find_text_point(ITEM_LIST_TITLE_REGION, ITEM_LIST_TITLE_TEXT)
            is not None
        ):
            return True
        return False

    def _find_category(self, keywords):
        for text, (x1, y1, x2, y2) in self._screen_boxes(CATEGORY_LIST_REGION):
            if any(kw in text for kw in keywords):
                return text, (x1, y1, x2, y2)
        return None

    def _first_other_item(self, other=None):
        """First item row below the 其它 category tab (exclude tab texts).

        ``other`` is the ``(text, box)`` result of ``_find_category``; when
        given, only rows below that tab's bottom edge are considered, so items
        belonging to the category above (equipment etc.) are never picked.
        """
        start_y = other[1][3] if other else None
        candidates = []
        for text, (x1, y1, x2, y2) in self._screen_boxes(OTHER_ITEM_REGION):
            text = text.strip()
            if len(text) < 2:
                continue
            if any(kw in text for kw in CATEGORY_EXCLUDE_TEXTS):
                continue
            if text.isdigit():
                continue
            if start_y is not None and y1 < start_y:
                continue
            candidates.append((y1, x1, text, (x1, y1, x2, y2)))
        if not candidates:
            return None
        candidates.sort(key=lambda c: (c[0], c[1]))
        return candidates[0][2], candidates[0][3]

    # -- input helpers ---------------------------------------------------------

    def _click_abs(self, point) -> None:
        click_at(point)
        time.sleep(self.delay_action)

    def _click_ocr_point(self, point) -> None:
        """OCR boxes are relative to the window screenshot; add the window
        top-left offset so clicks land on absolute screen coordinates."""
        left, top, _, _ = get_window_rect()
        self._click_abs((left + point[0], top + point[1]))

    def _click_box(self, box) -> None:
        left, top, _, _ = get_window_rect()
        x = left + (box[0] + box[2]) // 2
        y = top + (box[1] + box[3]) // 2
        self._click_abs((x, y))

    def _click_normalized(self, point) -> None:
        left, top, width, height = get_window_rect()
        self._click_abs((left + int(point[0] * width), top + int(point[1] * height)))
