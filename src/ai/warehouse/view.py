"""Non-blocking warehouse list/detail workflow AI."""

import difflib
import time
from enum import Enum
from typing import Callable, Optional

from ... import AI
from ...ocr import parse_ocr_boxes
from ...utils.clicker import click_at, drag_hold, move_cursor_to
from ...utils.shared import state
from ...utils.window import get_window_rect, get_window_screen
from .models import (
    EQUIPMENT_CATEGORIES,
    DestructiveActionPolicy,
    EquipmentDetail,
    ItemRow,
    WarehouseCategory,
    WarehouseDecision,
)
from .parser import (
    WarehouseLayout,
    find_text_boxes,
    is_detail_screen,
    is_filter_dialog,
    is_list_screen,
    normalized_point,
    normalized_region_pixels,
    offset_ocr_boxes,
    parse_detail,
    parse_list_rows,
)


class _WorkflowState(str, Enum):
    FIND_LIST = "find_list"
    SELECT_CATEGORY = "select_category"
    RESET_LIST = "reset_list"
    PROCESS_LIST = "process_list"
    WAIT_DETAIL = "wait_detail"
    CLOSE_DETAIL = "close_detail"
    WAIT_LIST = "wait_list"
    CLOSE_LIST = "close_list"
    WAIT_CLOSE_LIST = "wait_close_list"
    BLOCKED = "blocked"
    FINISHED = "finished"


class WarehouseView(AI):
    """Inspect configured equipment tabs one OCR-detected screen transition
    per check.
    """

    def __init__(
        self,
        *,
        on_item: Optional[Callable[[EquipmentDetail], WarehouseDecision]] = None,
        on_finish: Optional[Callable[[_WorkflowState], None]] = None,
        categories: tuple[WarehouseCategory, ...] = EQUIPMENT_CATEGORIES,
        layout: Optional[WarehouseLayout] = None,
        action_delay: float = 0.1,
        detail_retry_limit: int = 2,
        list_ocr_retry_limit: int = 2,
        max_pages: int = 50,
        reset_retry_limit: int = 20,
        no_progress_limit: int = 2,
        list_close_retry_limit: int = 2,
        drag_hold_duration: float = 0.05,
        drag_move_duration: float = 1.2,
        drag_settle_delay: float = 0.35,
    ) -> None:
        super().__init__()
        self.on_item = on_item or (lambda _: WarehouseDecision())
        self.on_finish = on_finish
        self._finish_notified = False
        self.categories = tuple(
            category for category in categories if category in EQUIPMENT_CATEGORIES
        )
        self.layout = layout or WarehouseLayout()
        self.action_delay = action_delay
        self.detail_retry_limit = detail_retry_limit
        self.list_ocr_retry_limit = list_ocr_retry_limit
        self.max_pages = max_pages
        self.reset_retry_limit = reset_retry_limit
        self.no_progress_limit = no_progress_limit
        self.list_close_retry_limit = list_close_retry_limit
        self.drag_hold_duration = drag_hold_duration
        self.drag_move_duration = drag_move_duration
        self.drag_settle_delay = drag_settle_delay
        self._state = _WorkflowState.FIND_LIST
        self._category_index = 0
        self._current_row: Optional[ItemRow] = None
        self._page_rows: list[ItemRow] = []
        self._page_cursor = 0
        self._page_loaded = False
        self._page_fingerprint: tuple[tuple[str, Optional[int]], ...] = ()
        self._previous_page_keys: tuple[tuple[str, Optional[int]], ...] = ()
        self._previous_page_rows: tuple[ItemRow, ...] = ()
        self._expected_page_overlap: Optional[int] = None
        self._page_count = 0
        self._stalled_pages = 0
        self._list_ocr_retries = 0
        self._detail_retries = 0
        self._close_retries = 0
        self._unknown_close_polls = 0
        self._list_close_retries = 0
        self._pending_decision = WarehouseDecision()
        self._cached_list_boxes = None
        self._cached_list_dimensions = None
        self._reset_fingerprint: tuple[str, ...] = ()
        self._reset_has_fingerprint = False
        self._reset_attempts = 0

    def _transition(self, new_state: _WorkflowState, reason: str) -> None:
        """Record meaningful workflow state transitions."""
        if self._state is not new_state:
            state.logger.info(
                "倉庫流程 %s -> %s: %s", self._state.value, new_state.value, reason
            )
        self._state = new_state
        if new_state in (_WorkflowState.BLOCKED, _WorkflowState.FINISHED):
            self._clear_cached_list()
            self._clear_reset_state()
            if self.on_finish is not None and not self._finish_notified:
                self._finish_notified = True
                self.on_finish(new_state)

    @staticmethod
    def _ocr_sample(boxes, limit: int = 8) -> str:
        """Return a compact OCR text sample for diagnostics."""
        return " | ".join(text for text, _ in boxes[:limit])

    def check(self) -> bool:
        """Advance one guarded state transition when the warehouse UI is visible."""
        if self._state in (_WorkflowState.FINISHED, _WorkflowState.BLOCKED):
            return False
        if self._cached_list_boxes is not None:
            boxes = self._cached_list_boxes
            width, height = self._cached_list_dimensions
            self._cached_list_boxes = None
            self._cached_list_dimensions = None
            state.logger.debug("倉庫重用已確認的清單 OCR")
        else:
            screen = get_window_screen()
            height, width = screen.shape[:2]
            x1, x2, y1, y2 = normalized_region_pixels(
                self.layout.ocr_region, width, height
            )
            crop = screen[y1:y2, x1:x2]
            started = time.perf_counter()
            ocr_result = state.ocr.predict(crop)
            boxes = offset_ocr_boxes(parse_ocr_boxes(ocr_result), (x1, y1))
            state.logger.debug(
                "倉庫 OCR: region=%s crop_shape=%s boxes=%d elapsed=%.3fs",
                self.layout.ocr_region,
                crop.shape,
                len(boxes),
                time.perf_counter() - started,
            )
        if is_filter_dialog(boxes):
            state.logger.warning("偵測到倉庫篩選對話，依安全規則不操作")
            return True
        if self._state is _WorkflowState.FIND_LIST:
            return self._find_list(boxes, width, height)
        if self._state is _WorkflowState.SELECT_CATEGORY:
            return self._select_category(width, height)
        if self._state is _WorkflowState.RESET_LIST:
            return self._reset_list(boxes, width, height)
        if self._state is _WorkflowState.PROCESS_LIST:
            return self._process_list(boxes, width, height)
        if self._state is _WorkflowState.WAIT_DETAIL:
            return self._wait_detail(boxes, width, height)
        if self._state is _WorkflowState.CLOSE_DETAIL:
            return self._close_detail(boxes, width, height)
        if self._state is _WorkflowState.WAIT_LIST:
            return self._wait_list(boxes, width, height)
        if self._state is _WorkflowState.CLOSE_LIST:
            return self._close_final_list(boxes, width, height)
        return self._wait_final_list_close(boxes, width, height)

    def _cache_list(self, boxes, width: int, height: int) -> None:
        self._cached_list_boxes = boxes
        self._cached_list_dimensions = (width, height)

    def _clear_cached_list(self) -> None:
        self._cached_list_boxes = None
        self._cached_list_dimensions = None

    def _clear_reset_state(self) -> None:
        self._reset_fingerprint = ()
        self._reset_has_fingerprint = False
        self._reset_attempts = 0

    def _handle_empty_list_ocr(self) -> bool:
        self._list_ocr_retries += 1
        state.logger.warning(
            "倉庫清單列 OCR 未解析到項目: 重試=%d/%d",
            self._list_ocr_retries,
            self.list_ocr_retry_limit,
        )
        if self._list_ocr_retries > self.list_ocr_retry_limit:
            self._transition(_WorkflowState.BLOCKED, "清單列 OCR 重試耗盡")
        return True

    def _find_list(self, boxes, width: int, height: int) -> bool:
        if not is_list_screen(boxes):
            return False
        warehouse = find_text_boxes(
            boxes,
            "warehouse_source",
            region=self.layout.source_region,
            width=width,
            height=height,
        )
        if warehouse:
            warehouse_box = warehouse[0][1]
            state.logger.info("倉庫來源使用 OCR box: box=%s", warehouse_box)
            self._click_box(warehouse_box)
        else:
            state.logger.info(
                "倉庫來源使用 fallback point: point=%s", self.layout.warehouse_source
            )
            self._click_normalized(self.layout.warehouse_source, width, height)
        self._transition(_WorkflowState.SELECT_CATEGORY, "已找到道具清單並選擇倉庫來源")
        return True

    def _select_category(self, width: int, height: int) -> bool:
        self._clear_cached_list()
        if self._category_index >= len(self.categories):
            self._transition(_WorkflowState.CLOSE_LIST, "所有指定類別已完成")
            return True
        self._clear_reset_state()
        self._page_rows = []
        self._page_cursor = 0
        self._page_loaded = False
        self._page_fingerprint = ()
        self._previous_page_keys = ()
        self._previous_page_rows = ()
        self._expected_page_overlap = None
        self._page_count = 0
        self._stalled_pages = 0
        self._list_ocr_retries = 0
        category = self.categories[self._category_index]
        state.logger.info("倉庫選擇類別: %s，已重設頁面掃描", category.value)
        self._click_normalized(self.layout.category_points[category], width, height)
        state.logger.info(
            "倉庫分類開始: 嘗試將清單卷軸置頂 start=%s end=%s",
            self.layout.list_reset_start,
            self.layout.list_reset_end,
        )
        self._drag_list(
            width,
            height,
            start=self.layout.list_reset_start,
            end=self.layout.list_reset_end,
            reason="分類開始置頂",
        )
        self._reset_attempts = 1
        self._transition(
            _WorkflowState.RESET_LIST, f"已選擇類別 {category.value}，等待置頂確認"
        )
        return True

    def _reset_list(self, boxes, width: int, height: int) -> bool:
        if not is_list_screen(boxes):
            state.logger.warning("倉庫置頂確認失敗：未辨識到清單")
            self._transition(_WorkflowState.BLOCKED, "置頂確認時未辨識到清單")
            return True
        rows = parse_list_rows(boxes, width, height, self.layout)
        fingerprint = tuple(row.name for row in rows)
        state.logger.info(
            "倉庫置頂確認: 嘗試=%d/%d 指紋=%s 列=%d",
            self._reset_attempts,
            self.reset_retry_limit,
            fingerprint,
            len(rows),
        )
        if not rows:
            state.logger.warning("倉庫置頂確認失敗：OCR 未解析到可見列")
            if self._reset_attempts >= self.reset_retry_limit:
                state.logger.warning("倉庫置頂確認失敗：重設嘗試上限已達")
                self._transition(_WorkflowState.BLOCKED, "置頂確認重設嘗試上限已達")
                return True
            self._reset_attempts += 1
        elif self._reset_has_fingerprint and fingerprint == self._reset_fingerprint:
            self._clear_reset_state()
            self._cache_list(boxes, width, height)
            state.logger.info("倉庫置頂確認完成：OCR 指紋穩定")
            self._transition(_WorkflowState.PROCESS_LIST, "置頂 OCR 指紋穩定")
            return True
        else:
            if self._reset_attempts >= self.reset_retry_limit:
                state.logger.warning("倉庫置頂確認失敗：重設嘗試上限已達")
                self._transition(_WorkflowState.BLOCKED, "置頂確認重設嘗試上限已達")
                return True
            self._reset_fingerprint = fingerprint
            self._reset_has_fingerprint = True
            self._reset_attempts += 1
        points = self._visible_page_drag_points(rows, height, "down")
        start = points[0] if points else self.layout.list_reset_start
        end = points[1] if points else self.layout.list_reset_end
        self._drag_list(
            width,
            height,
            start=start,
            end=end,
            reason="置頂確認重試",
        )
        return True

    def _process_list(self, boxes, width: int, height: int) -> bool:
        if not is_list_screen(boxes):
            self._transition(_WorkflowState.FIND_LIST, "處理清單時未再辨識到道具清單")
            return False
        if not self._page_loaded:
            self._page_rows = parse_list_rows(boxes, width, height, self.layout)
            if not self._page_rows:
                return self._handle_empty_list_ocr()
            self._list_ocr_retries = 0
            self._page_cursor = self._adjacent_page_overlap(
                self._previous_page_rows,
                self._page_rows,
                expected_overlap=self._expected_page_overlap,
            )
            state.logger.info(
                "倉庫載入頁面: 類別=%s 頁=%d 列=%d 重疊=%d 項目=%s",
                self.categories[self._category_index].value,
                self._page_count + 1,
                len(self._page_rows),
                self._page_cursor,
                [row.name for row in self._page_rows[:6]],
            )
            if self._page_count and self._page_rows_unchanged(self._page_rows):
                state.logger.info(
                    "倉庫到達類別尾端: 類別=%s 確認頁=%d 列=%d",
                    self.categories[self._category_index].value,
                    self._page_count + 1,
                    len(self._page_rows),
                )
                self._category_index += 1
                self._transition(
                    _WorkflowState.SELECT_CATEGORY, "已確認最後一頁，無新項目"
                )
                return True
            self._page_loaded = True
        if self._page_cursor < len(self._page_rows):
            self._current_row = self._page_rows[self._page_cursor]
            state.logger.info(
                "倉庫點擊資訊: 類別=%s 列=%d 名稱=%s 星=%s y=%d",
                self.categories[self._category_index].value,
                self._page_cursor,
                self._current_row.name,
                self._current_row.stars,
                self._current_row.row_y,
            )
            self._click_info(self._current_row, width, height)
            self._detail_retries = 0
            self._transition(_WorkflowState.WAIT_DETAIL, "已點擊項目資訊")
            return True
        visible_rows = parse_list_rows(boxes, width, height, self.layout)
        if not visible_rows:
            return self._handle_empty_list_ocr()
        self._list_ocr_retries = 0
        fingerprint = tuple((row.name, row.stars) for row in visible_rows)
        if (
            self._page_count >= self.max_pages
            or self._stalled_pages >= self.no_progress_limit
        ):
            self._category_index += 1
            self._transition(_WorkflowState.SELECT_CATEGORY, "頁面上限或無進度上限已達")
            return True
        if self._page_count and self._fingerprint_matches(
            self._page_fingerprint, fingerprint
        ):
            self._stalled_pages += 1
        else:
            self._stalled_pages = 0
        self._page_fingerprint = fingerprint
        self._previous_page_keys = fingerprint
        self._previous_page_rows = tuple(visible_rows)
        self._expected_page_overlap = self._estimate_page_overlap(
            visible_rows, height
        )
        self._page_count += 1
        self._page_loaded = False
        state.logger.info(
            "倉庫拖曳下一頁: 類別=%s 頁=%d",
            self.categories[self._category_index].value,
            self._page_count,
        )
        self._clear_cached_list()
        points = self._visible_page_drag_points(visible_rows, height, "up")
        start = points[0] if points else self.layout.list_drag_start
        end = points[1] if points else self.layout.list_drag_end
        self._drag_list(width, height, start=start, end=end)
        return True

    def _estimate_page_overlap(self, rows, height: int) -> Optional[int]:
        span = self._list_drag_span(rows, height)
        if span is None:
            return None
        _, top_y, bottom_y, pitch = span
        drag_distance = (bottom_y - top_y) + pitch
        scrolled_rows = max(1, round(drag_distance / pitch))
        return max(0, len(rows) - scrolled_rows)

    def _list_drag_span(
        self,
        rows,
        height: int,
    ) -> Optional[tuple[float, int, int, float]]:
        pitch = self._row_pitch(tuple(rows))
        if not rows or pitch <= 0:
            return None
        x = (self.layout.list_region[0] + self.layout.list_region[1]) / 2
        return x, rows[0].row_y, rows[-1].row_y, pitch

    def _visible_page_drag_points(self, rows, height: int, direction: str):
        """Drag points spanning exactly one visible page: grab the row on the
        side the gesture moves toward and release one row pitch beyond the
        opposite edge (topmost row -> bottommost row + one pitch)."""
        span = self._list_drag_span(rows, height)
        if span is None:
            return None
        x, top_y, bottom_y, pitch = span
        if direction == "down":
            start_y = top_y / height
            end_y = min(0.99, (bottom_y + pitch) / height)
        else:
            start_y = bottom_y / height
            end_y = max(0.01, (top_y - pitch) / height)
        return (x, start_y), (x, end_y)

    @staticmethod
    def _row_key(row) -> tuple[str, Optional[int]]:
        if isinstance(row, ItemRow):
            return row.name, row.stars
        return row

    @staticmethod
    def _row_pitch(rows: tuple[ItemRow, ...]) -> float:
        gaps = [
            current.row_y - previous.row_y
            for previous, current in zip(rows, rows[1:])
            if current.row_y > previous.row_y
        ]
        if not gaps:
            return 0.0
        ordered = sorted(gaps)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return float(ordered[middle])
        return (ordered[middle - 1] + ordered[middle]) / 2

    @staticmethod
    def _names_close(a: str, b: str) -> bool:
        """OCR-variance tolerant name comparison."""
        if a == b:
            return True
        if abs(len(a) - len(b)) > 1:
            return False
        return difflib.SequenceMatcher(None, a, b).ratio() >= 0.6

    @classmethod
    def _keys_close(cls, a, b) -> bool:
        name_a, stars_a = a
        name_b, stars_b = b
        if stars_a is not None and stars_b is not None and stars_a != stars_b:
            return False
        return cls._names_close(name_a, name_b)

    @classmethod
    def _fingerprint_matches(cls, previous_fp, current_fp) -> bool:
        """Tolerant fingerprint comparison across OCR passes."""
        if not previous_fp or len(previous_fp) != len(current_fp):
            return False
        close = sum(
            1
            for previous_key, current_key in zip(previous_fp, current_fp)
            if cls._keys_close(previous_key, current_key)
        )
        return close >= max(1, len(previous_fp) // 2)

    @classmethod
    def _adjacent_page_overlap(
        cls,
        previous,
        current,
        *,
        expected_overlap: Optional[int] = None,
    ) -> int:
        """Return the visible suffix/prefix overlap, including duplicate names."""
        previous_rows = tuple(previous)
        current_rows = tuple(current)
        if not previous_rows or not current_rows:
            return 0

        previous_keys = tuple(cls._row_key(row) for row in previous_rows)
        current_keys = tuple(cls._row_key(row) for row in current_rows)
        max_overlap = min(len(previous_rows), len(current_rows))
        target_overlap = (
            None
            if expected_overlap is None
            else max(0, min(max_overlap, expected_overlap))
        )
        candidates = []
        use_positions = all(
            isinstance(row, ItemRow) for row in (*previous_rows, *current_rows)
        )
        row_pitch = cls._row_pitch(previous_rows) if use_positions else 0.0
        for length in range(1, max_overlap + 1):
            if previous_keys[-length:] != current_keys[:length]:
                continue
            if (
                length < len(current_rows)
                and current_keys[length] == current_keys[length - 1]
            ):
                continue
            if not use_positions:
                score = abs(length - target_overlap) if target_overlap is not None else 0
                candidates.append((score, -length, length))
                continue

            shifts = [
                previous_rows[-length + index].row_y - current_rows[index].row_y
                for index in range(length)
            ]
            if max(shifts) - min(shifts) > max(12.0, row_pitch * 0.4):
                continue
            if target_overlap is None:
                median_shift = sorted(shifts)[len(shifts) // 2]
                scrolled_rows = (
                    max(0, round(median_shift / row_pitch)) if row_pitch else 0
                )
                expected = max(
                    0,
                    min(
                        len(previous_rows),
                        len(current_rows),
                        len(previous_rows) - scrolled_rows,
                    ),
                )
            else:
                expected = target_overlap
            candidates.append((abs(length - expected), -length, length))

        return min(candidates)[2] if candidates else 0

    def _page_rows_unchanged(self, current_rows) -> bool:
        """Return True when the current visible page matches the previous page
        up to OCR text variance, i.e. scrolling produced no new items.
        """
        rows = tuple(current_rows)
        current_keys = tuple(self._row_key(row) for row in rows)
        previous = self._previous_page_rows
        if not current_keys:
            return False
        if not previous:
            previous_keys = self._previous_page_keys
            return (
                len(current_keys) <= len(previous_keys)
                and current_keys == previous_keys[-len(current_keys) :]
            )
        pitch = self._row_pitch(previous) or self._row_pitch(rows) or 0.0
        tolerance = max(12.0, pitch * 0.4)
        by_key = {}
        for previous_row in previous:
            by_key.setdefault(self._row_key(previous_row), []).append(previous_row)
        for row in rows:
            key = self._row_key(row)
            candidates = by_key.get(key)
            if candidates and any(
                abs(candidate.row_y - row.row_y) <= tolerance
                for candidate in candidates
            ):
                continue
            position_matches = [
                candidate
                for candidate in previous
                if abs(candidate.row_y - row.row_y) <= tolerance
            ]
            if position_matches and any(
                self._names_close(candidate.name, row.name)
                for candidate in position_matches
            ):
                continue
            return False
        return True

    def _realign_list_cursor(
        self,
        boxes,
        width: int,
        height: int,
        *,
        advance: bool,
        cache: bool = True,
    ) -> Optional[bool]:
        rows = parse_list_rows(boxes, width, height, self.layout)
        if not rows:
            return None
        current = self._current_row
        previous_cursor = self._page_cursor
        if rows and current:
            matching = [
                index
                for index, row in enumerate(rows)
                if self._row_key(row) == self._row_key(current)
            ]
            if matching:
                index = min(
                    matching,
                    key=lambda candidate: abs(rows[candidate].row_y - current.row_y),
                )
                self._page_rows = rows
                self._page_loaded = True
                self._page_cursor = index + (1 if advance else 0)
                state.logger.info(
                    "倉庫返回清單重新對齊: 項目=%s 列=%d 下一列=%d",
                    current.name,
                    index,
                    self._page_cursor,
                )
                if cache:
                    self._cache_list(boxes, width, height)
                return True

        if not current:
            self._page_rows = rows
            self._page_loaded = True
            self._page_cursor = min(previous_cursor, len(rows))
            if cache:
                self._cache_list(boxes, width, height)
            return False
        state.logger.warning(
            "倉庫返回清單無法對齊目前項目: 名稱=%s，保留游標=%d",
            current.name,
            previous_cursor,
        )
        return False

    def _wait_detail(self, boxes, width: int, height: int) -> bool:
        if is_detail_screen(boxes) and self._current_row:
            detail = parse_detail(
                boxes,
                self.categories[self._category_index],
                width,
                height,
                self.layout,
            )
            self._pending_decision = self.on_item(detail) or WarehouseDecision()
            state.logger.info(
                "倉庫詳細已辨識: 名稱=%s 星=%s 加護=%s",
                detail.name,
                detail.stars,
                [
                    (blessing.name, blessing.value, blessing.is_percent)
                    for blessing in detail.blessings
                ],
            )
            self._transition(_WorkflowState.CLOSE_DETAIL, "詳細 OCR 已接受")
            return self._close_detail(boxes, width, height)
        if is_list_screen(boxes):
            self._detail_retries += 1
            aligned = self._realign_list_cursor(
                boxes,
                width,
                height,
                advance=False,
                cache=False,
            )
            if aligned is None:
                if self._detail_retries > self.detail_retry_limit:
                    self._transition(
                        _WorkflowState.BLOCKED,
                        "清單 OCR 未解析到可重試項目",
                    )
                return True
            if aligned is False:
                if self._detail_retries > self.detail_retry_limit:
                    self._transition(
                        _WorkflowState.BLOCKED,
                        "詳細頁未開啟且清單項目無法重新對齊",
                    )
                else:
                    state.logger.info(
                        "倉庫等待目前項目 OCR 對齊: 重試=%d/%d",
                        self._detail_retries,
                        self.detail_retry_limit,
                    )
                return True
            if (
                self._detail_retries <= self.detail_retry_limit
                and self._current_row
            ):
                state.logger.info(
                    "倉庫詳細頁未開啟，重試資訊按鈕: 名稱=%s 重試=%d/%d",
                    self._current_row.name,
                    self._detail_retries,
                    self.detail_retry_limit,
                )
                self._click_info(self._current_row, width, height)
                return True
            self._transition(
                _WorkflowState.BLOCKED,
                "詳細頁未開啟且清單項目無法重新對齊",
            )
            return True
        self._detail_retries += 1
        state.logger.debug(
            "倉庫詳細 OCR 未接受: 重試=%d/%d 樣本=%s",
            self._detail_retries,
            self.detail_retry_limit,
            self._ocr_sample(boxes),
        )
        if self._detail_retries <= self.detail_retry_limit and self._current_row:
            state.logger.info(
                "倉庫重試資訊按鈕: 名稱=%s 重試=%d/%d",
                self._current_row.name,
                self._detail_retries,
                self.detail_retry_limit,
            )
            self._click_info(self._current_row, width, height)
            return True
        state.logger.warning(
            "倉庫項目詳細頁未開啟且未確認回到清單，停止工作流程: %s",
            self._current_row.name if self._current_row else "未知",
        )
        self._transition(_WorkflowState.BLOCKED, "詳細 OCR 與清單回復均未確認")
        return True

    def _close_detail(self, boxes, width: int, height: int) -> bool:
        if not is_detail_screen(boxes):
            if is_list_screen(boxes):
                aligned = self._realign_list_cursor(
                    boxes,
                    width,
                    height,
                    advance=True,
                )
                if aligned is not True:
                    self._transition(
                        _WorkflowState.WAIT_LIST,
                        "已返回清單，等待項目 OCR 對齊",
                    )
                    return True
                state.logger.info("倉庫已返回清單: 游標=%d", self._page_cursor)
                self._transition(
                    _WorkflowState.PROCESS_LIST, "已確認返回清單並重新對齊游標"
                )
                return True
            self._unknown_close_polls += 1
            if self._unknown_close_polls > self.detail_retry_limit:
                state.logger.warning("倉庫詳細頁關閉狀態無法確認，停止工作流程")
                self._transition(
                    _WorkflowState.BLOCKED, "詳細關閉後畫面狀態不明且重試耗盡"
                )
            else:
                state.logger.info(
                    "倉庫等待清單恢復: 重試=%d/%d",
                    self._unknown_close_polls,
                    self.detail_retry_limit,
                )
                self._transition(_WorkflowState.WAIT_LIST, "詳細關閉後等待確認清單")
            return True
        if (
            self._pending_decision.discard
            and self._pending_decision.discard_policy is DestructiveActionPolicy.ALLOW
        ):
            discard = find_text_boxes(boxes, "丟棄", width=width, height=height)
            if discard:
                state.logger.info("倉庫執行允許的丟棄操作")
                self._click_box(discard[0][1])
                self._transition(_WorkflowState.WAIT_LIST, "已點擊允許的丟棄按鈕")
                return True
        state.logger.info("倉庫安全關閉詳細頁")
        self._click_close(boxes, width, height)
        self._close_retries = 0
        self._unknown_close_polls = 0
        self._transition(_WorkflowState.WAIT_LIST, "已請求安全關閉詳細頁")
        return True

    def _wait_list(self, boxes, width: int, height: int) -> bool:
        if is_list_screen(boxes):
            aligned = self._realign_list_cursor(
                boxes,
                width,
                height,
                advance=True,
            )
            if aligned is not True:
                self._close_retries += 1
                if self._close_retries > self.detail_retry_limit:
                    self._transition(
                        _WorkflowState.BLOCKED,
                        "返回清單後項目 OCR 對齊重試耗盡",
                    )
                else:
                    state.logger.info(
                        "倉庫等待清單項目 OCR 對齊: 重試=%d/%d",
                        self._close_retries,
                        self.detail_retry_limit,
                    )
                return True
            self._close_retries = 0
            self._unknown_close_polls = 0
            state.logger.info("倉庫確認清單恢復: 游標=%d", self._page_cursor)
            self._transition(
                _WorkflowState.PROCESS_LIST, "已確認清單恢復並重新對齊游標"
            )
            return True
        if is_detail_screen(boxes):
            if self._close_retries < self.detail_retry_limit:
                self._close_retries += 1
                state.logger.info(
                    "倉庫安全關閉重試: 重試=%d/%d",
                    self._close_retries,
                    self.detail_retry_limit,
                )
                self._click_close(boxes, width, height)
                return True
            state.logger.warning("倉庫詳細頁關閉重試耗盡，停止工作流程")
            self._transition(_WorkflowState.BLOCKED, "詳細頁關閉重試耗盡")
            return True
        self._unknown_close_polls += 1
        if self._unknown_close_polls <= self.detail_retry_limit:
            state.logger.info(
                "倉庫等待畫面 OCR 確認: 重試=%d/%d 樣本=%s",
                self._unknown_close_polls,
                self.detail_retry_limit,
                self._ocr_sample(boxes),
            )
            return True
        state.logger.warning(
            "倉庫詳細頁關閉後畫面狀態不明，停止工作流程: 樣本=%s",
            self._ocr_sample(boxes),
        )
        self._transition(_WorkflowState.BLOCKED, "清單或詳細畫面均未確認")
        return True

    def _click_close(self, boxes, width: int, height: int) -> None:
        close = find_text_boxes(boxes, "關閉", width=width, height=height)
        if close:
            self._click_box(close[-1][1])
        else:
            self._click_normalized(self.layout.close_point, width, height)

    def _close_final_list(self, boxes, width: int, height: int) -> bool:
        if not is_list_screen(boxes):
            state.logger.warning("倉庫清單關閉前無法確認清單畫面，停止工作流程")
            self._transition(_WorkflowState.BLOCKED, "關閉清單前未確認清單畫面")
            return True
        state.logger.info("倉庫請求關閉最終清單")
        self._click_list_close(boxes, width, height)
        self._list_close_retries = 0
        self._transition(_WorkflowState.WAIT_CLOSE_LIST, "已請求關閉最終清單")
        return True

    def _wait_final_list_close(self, boxes, width: int, height: int) -> bool:
        if not is_list_screen(boxes):
            state.logger.info("倉庫最終清單已消失，流程完成")
            self._transition(_WorkflowState.FINISHED, "已確認最終清單關閉")
            return True
        if self._list_close_retries < self.list_close_retry_limit:
            self._list_close_retries += 1
            state.logger.info(
                "倉庫關閉最終清單重試: 重試=%d/%d",
                self._list_close_retries,
                self.list_close_retry_limit,
            )
            self._click_list_close(boxes, width, height)
            return True
        state.logger.warning("倉庫清單關閉未確認，停止工作流程")
        self._transition(_WorkflowState.BLOCKED, "最終清單關閉重試耗盡")
        return True

    def _click_list_close(self, boxes, width: int, height: int) -> None:
        close = find_text_boxes(boxes, "關閉", width=width, height=height)
        if close:
            self._click_box(close[-1][1])
        elif is_list_screen(boxes):
            self._click_normalized(self.layout.close_point, width, height)

    def _click_info(self, row: ItemRow, width: int, height: int) -> None:
        self._click_normalized((0.625, row.row_y / height), width, height)
        self._move_cursor_to_detail_safe_point(width, height)

    def _move_cursor_to_detail_safe_point(self, width: int, height: int) -> None:
        left, top, _, _ = get_window_rect()
        safe_point = self.layout.detail_safe_cursor
        absolute_point = normalized_point(safe_point, width, height, (left, top))
        state.logger.info(
            "倉庫詳細 OCR 前移開游標: normalized=%s absolute=%s",
            safe_point,
            absolute_point,
        )
        move_cursor_to(absolute_point)

    def _drag_list(
        self,
        width: int,
        height: int,
        *,
        start: Optional[tuple[float, float]] = None,
        end: Optional[tuple[float, float]] = None,
        reason: Optional[str] = None,
    ) -> None:
        """Drag the list by a bounded, slow gesture and wait for it to settle."""
        normalized_start = start or self.layout.list_drag_start
        normalized_end = end or self.layout.list_drag_end
        left, top, _, _ = get_window_rect()
        absolute_start = normalized_point(normalized_start, width, height, (left, top))
        absolute_end = normalized_point(normalized_end, width, height, (left, top))
        state.logger.info(
            "倉庫清單拖曳: 原因=%s normalized_start=%s normalized_end=%s "
            "absolute_start=%s absolute_end=%s",
            reason or "下一頁",
            normalized_start,
            normalized_end,
            absolute_start,
            absolute_end,
        )
        drag_hold(
            absolute_start,
            absolute_end,
            hold_duration=self.drag_hold_duration,
            move_duration=self.drag_move_duration,
            release_pause_duration=self.drag_settle_delay,
        )
        time.sleep(max(self.action_delay, self.drag_settle_delay))

    def _close_list(self, width: int, height: int) -> None:
        self._click_normalized(self.layout.close_point, width, height)

    def _click_normalized(self, point, width: int, height: int) -> None:
        left, top, _, _ = get_window_rect()
        click_at(normalized_point(point, width, height, (left, top)))
        time.sleep(self.action_delay)

    def _click_box(self, box: tuple[int, int, int, int]) -> None:
        left, top, _, _ = get_window_rect()
        click_at((left + (box[0] + box[2]) // 2, top + (box[1] + box[3]) // 2))
        time.sleep(self.action_delay)
