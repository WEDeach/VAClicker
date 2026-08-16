"""Pure OCR-box parsing helpers for warehouse screens."""

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional

from .models import Blessing, EquipmentDetail, ItemRow, WarehouseCategory

OCRBox = tuple[str, tuple[int, int, int, int]]
_BLESSING_RE = re.compile(r"^(.+?)([+-]\s*\d+)\s*(%)?$")
_STAR_RE = re.compile(r"(?:★|☆|星)\s*(\d+)|(?<!\d)(\d+)\s*(?:星|★)")
UI_TEXT_ALIASES = {
    "blessing_marker": ("附加加護", "附加加漫", "附加加护"),
    "discard": ("丟棄", "丢棄", "丟弃", "丢弃"),
    "close": ("關閉", "关闭"),
    "select_all": ("全選", "全选"),
    "cancel_all": ("全部取消",),
    "locked": ("強化值", "强化值", "開放", "开放", "未開放", "未开放", "鎖定", "锁定"),
    "warehouse_source": ("倉庫", "倉库"),
}
_STAT_NAME_MARKERS = (
    "總持有數量",
    "總持有数量",
    "最大強化值",
    "最大强化值",
    "攻擊力",
    "防禦力",
    "行動速度",
    "法術防禦",
    "抵抗",
    "命中",
    "迴避",
    "生命",
    "強化值",
)
_UI_TEXT = {
    "道具清單",
    "倉庫",
    "全部",
    "頭",
    "武器",
    "手",
    "身體",
    "副手",
    "腳",
    "飾品",
    "消耗品",
    "收藏",
    "篩選",
    "關閉",
    "丟棄",
    "交付",
}
_LIST_NON_ITEM_TEXT = {"i", "l", "1", "丨"}
_LOCKED_MARKERS = UI_TEXT_ALIASES["locked"]


def _aliases_for(target: str) -> tuple[str, ...]:
    """Resolve an alias-group key or canonical UI label to its OCR variants."""
    if target in UI_TEXT_ALIASES:
        return UI_TEXT_ALIASES[target]
    for aliases in UI_TEXT_ALIASES.values():
        if target in aliases:
            return aliases
    return (target,)


def _contains_alias(text: str, target: str) -> bool:
    return any(alias in text for alias in _aliases_for(target))


def _is_blessing_marker(text: str) -> bool:
    return _contains_alias(text, "blessing_marker") or (
        "附加" in text and len(text) <= 6
    )


@dataclass(frozen=True)
class WarehouseLayout:
    """Normalized controls, including the centered upper detail name/stars region."""

    ocr_region: Optional[tuple[float, float, float, float]] = (0.32, 0.61, 0.02, 0.98)
    list_region: tuple[float, float, float, float] = (0.36, 0.61, 0.14, 0.70)
    detail_name_region: tuple[float, float, float, float] = (0.36, 0.61, 0.27, 0.43)
    category_points: dict[
        WarehouseCategory,
        tuple[float, float],
    ] = None  # type: ignore[assignment]
    source_region: tuple[float, float, float, float] = (0.35, 0.61, 0.65, 0.83)
    warehouse_source: tuple[float, float] = (0.39, 0.807)
    detail_safe_cursor: tuple[float, float] = (0.74, 0.50)
    close_point: tuple[float, float] = (0.50, 0.91)
    list_drag_start: tuple[float, float] = (0.50, 0.68)
    list_drag_end: tuple[float, float] = (0.50, 0.22)
    list_reset_start: tuple[float, float] = (0.50, 0.22)
    list_reset_end: tuple[float, float] = (0.50, 0.68)

    def __post_init__(self) -> None:
        if self.category_points is None:
            object.__setattr__(
                self,
                "category_points",
                {
                    WarehouseCategory.ALL: (0.369, 0.74),
                    WarehouseCategory.HEAD: (0.395, 0.74),
                    WarehouseCategory.WEAPON: (0.420, 0.74),
                    WarehouseCategory.HAND: (0.446, 0.74),
                    WarehouseCategory.BODY: (0.471, 0.74),
                    WarehouseCategory.OFF_HAND: (0.497, 0.74),
                    WarehouseCategory.FOOT: (0.522, 0.74),
                    WarehouseCategory.ACCESSORY: (0.548, 0.74),
                    WarehouseCategory.CONSUMABLE: (0.574, 0.74),
                    WarehouseCategory.COLLECTION: (0.599, 0.74),
                    WarehouseCategory.FILTER: (0.625, 0.74),
                },
            )


def normalize_text(text: str) -> str:
    """Collapse OCR whitespace without altering Traditional-Chinese characters."""
    return re.sub(r"\s+", "", text).strip()


def find_text_boxes(
    boxes: Iterable[OCRBox],
    text: str,
    region: Optional[tuple[float, float, float, float]] = None,
    width: int = 1,
    height: int = 1,
) -> list[OCRBox]:
    """Return normalized-text matches, optionally constrained to a
    normalized region.
    """
    matches = []
    for raw, box in boxes:
        normalized = normalize_text(raw)
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        if not _contains_alias(normalized, text):
            continue
        if region and not (
            region[0] * width <= center_x <= region[1] * width
            and region[2] * height <= center_y <= region[3] * height
        ):
            continue
        matches.append((normalized, box))
    return matches


def normalized_point(
    point: tuple[float, float],
    width: int,
    height: int,
    origin: tuple[int, int] = (0, 0),
) -> tuple[int, int]:
    """Convert a normalized client point into an absolute screen point."""
    return origin[0] + round(point[0] * width), origin[1] + round(point[1] * height)


def normalized_region_pixels(
    region: Optional[tuple[float, float, float, float]],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Convert an x1/x2/y1/y2 normalized region to clamped crop bounds; None
    is full screen.
    """
    if region is None:
        return 0, width, 0, height
    x1, x2, y1, y2 = region
    return (
        max(0, min(width, round(x1 * width))),
        max(0, min(width, round(x2 * width))),
        max(0, min(height, round(y1 * height))),
        max(0, min(height, round(y2 * height))),
    )


def offset_ocr_boxes(boxes: Iterable[OCRBox], offset: tuple[int, int]) -> list[OCRBox]:
    """Translate crop-local OCR boxes into full-screen client coordinates."""
    offset_x, offset_y = offset
    return [
        (text, (x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y))
        for text, (x1, y1, x2, y2) in boxes
    ]


def is_list_screen(boxes: Iterable[OCRBox]) -> bool:
    return any("道具清單" in normalize_text(text) for text, _ in boxes)


def is_detail_screen(boxes: Iterable[OCRBox]) -> bool:
    texts = [normalize_text(text) for text, _ in boxes]
    return any(_is_blessing_marker(text) for text in texts) or any(
        _contains_alias(text, "discard") for text in texts
    )


def is_filter_dialog(boxes: Iterable[OCRBox]) -> bool:
    texts = [normalize_text(text) for text, _ in boxes]
    has_select_all = any(_contains_alias(text, "select_all") for text in texts)
    has_cancel_all = any(_contains_alias(text, "cancel_all") for text in texts)
    return has_select_all and has_cancel_all


def _stars(text: str) -> Optional[int]:
    normalized = normalize_text(text)
    glyphs = re.fullmatch(r"[★☆]+", normalized)
    if glyphs:
        return len(normalized)
    match = _STAR_RE.search(normalized)
    if not match:
        return None
    return int(next(value for value in match.groups() if value is not None))


def parse_list_rows(
    boxes: Iterable[OCRBox],
    width: int,
    height: int,
    layout: Optional[WarehouseLayout] = None,
) -> list[ItemRow]:
    """Group list OCR by row and retain duplicate-name rows with
    distinct signatures.
    """
    layout = layout or WarehouseLayout()
    x1, x2, y1, y2 = layout.list_region
    groups: dict[int, list[tuple[str, tuple[int, int, int, int]]]] = defaultdict(list)
    for raw, box in boxes:
        text = normalize_text(raw)
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        if not text or not (
            x1 * width <= center_x <= x2 * width
            and y1 * height <= center_y <= y2 * height
        ):
            continue
        groups[round(center_y / max(12, height * 0.018))].append((text, box))
    rows = []
    for row_boxes in groups.values():
        raw_line = " ".join(
            text for text, _ in sorted(row_boxes, key=lambda item: item[1][0])
        )
        stars = next(
            (_stars(text) for text, _ in row_boxes if _stars(text) is not None), None
        )
        names = [
            text
            for text, _ in row_boxes
            if (
                text not in _UI_TEXT
                and text.casefold() not in _LIST_NON_ITEM_TEXT
                and _stars(text) is None
            )
        ]
        name = max(names, key=len, default="")
        if not name or re.fullmatch(r"[★☆\s\d星]+", name):
            continue
        row_y = round(
            sum((box[1] + box[3]) / 2 for _, box in row_boxes) / len(row_boxes)
        )
        normalized_name = normalize_text(name)
        star_text = stars if stars is not None else ""
        row_bucket = row_y // max(1, round(height * 0.012))
        signature = f"{normalized_name}|{star_text}|{row_bucket}"
        rows.append(
            ItemRow(
                name=name,
                stars=stars,
                row_y=row_y,
                signature=signature,
                raw_text=raw_line,
            )
        )
    return sorted(rows, key=lambda row: row.row_y)


def parse_detail(
    boxes: Iterable[OCRBox],
    category: WarehouseCategory,
    width: int,
    height: int,
    layout: Optional[WarehouseLayout] = None,
) -> EquipmentDetail:
    """Parse active additional blessings between the section marker and locked rows."""
    layout = layout or WarehouseLayout()
    ordered = sorted(
        ((normalize_text(text), box) for text, box in boxes),
        key=lambda item: (item[1][1], item[1][0]),
    )
    raw_text = tuple(text for text, _ in ordered if text)
    marker_index = next(
        (i for i, item in enumerate(ordered) if _is_blessing_marker(item[0])),
        len(ordered),
    )

    def is_name_candidate(text: str) -> bool:
        return (
            text not in _UI_TEXT
            and _stars(text) is None
            and "附加" not in text
            and not any(marker in text for marker in _STAT_NAME_MARKERS)
        )

    name_candidates = [
        (text, box) for text, box in ordered[:marker_index] if is_name_candidate(text)
    ]
    x1, x2, y1, y2 = layout.detail_name_region
    region_candidates = [
        (text, box)
        for text, box in name_candidates
        if x1 * width <= (box[0] + box[2]) / 2 <= x2 * width
        and y1 * height <= (box[1] + box[3]) / 2 <= y2 * height
    ]
    candidates = region_candidates or name_candidates
    name = min(
        candidates,
        key=lambda item: (item[1][1], item[1][0]),
        default=("", (0, 0, 0, 0)),
    )[0]
    stars = next(
        (
            _stars(text)
            for text, _ in ordered[:marker_index]
            if _stars(text) is not None
        ),
        None,
    )
    weapon_types = (
        "單手劍",
        "雙手斧",
        "投擲忍具",
        "短劍",
        "弓",
        "雙手槍",
        "雙手劍",
        "忍者刀",
        "雙手鈍器",
        "大太刀",
        "單手斧",
        "單手鈍器",
        "大劍",
        "刀",
    )
    weapon_type = next(
        (text for text, _ in ordered[:marker_index] if text in weapon_types), None
    )
    blessings = []
    for text, _ in ordered[marker_index + 1 :]:
        if any(marker in text for marker in _LOCKED_MARKERS):
            break
        if (
            text in _UI_TEXT
            or _contains_alias(text, "discard")
            or "交付" in text
            or _contains_alias(text, "close")
        ):
            continue
        match = _BLESSING_RE.match(text)
        if match:
            stat, value, percent = match.groups()
            blessings.append(
                Blessing(
                    normalize_text(stat),
                    int(value.replace(" ", "")),
                    bool(percent),
                    text,
                )
            )
    return EquipmentDetail(
        name=name,
        category=category,
        stars=stars,
        weapon_type=weapon_type,
        blessings=tuple(blessings),
        raw_text=raw_text,
        metadata={"section_found": marker_index < len(ordered)},
    )
