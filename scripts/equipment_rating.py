"""Rate active Wizardry Variants equipment blessings without discarding items."""

import argparse
import datetime
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.ai.warehouse.models import (
    DestructiveActionPolicy,
    EquipmentDetail,
    WarehouseCategory,
    WarehouseDecision,
)
from src.utils.shared import state

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

DEBUG = True
ACTION_DELAY = 0.25
DRAG_HOLD_DURATION = 0.05
DRAG_MOVE_DURATION = 1.2
DRAG_SETTLE_DELAY = 0.35
DISCARDS = DestructiveActionPolicy.DENY
_CATEGORY_TYPES = {
    "head": WarehouseCategory.HEAD,
    "weapon": WarehouseCategory.WEAPON,
    "hand": WarehouseCategory.HAND,
    "body": WarehouseCategory.BODY,
    "off_hand": WarehouseCategory.OFF_HAND,
    "offhand": WarehouseCategory.OFF_HAND,
    "foot": WarehouseCategory.FOOT,
    "accessory": WarehouseCategory.ACCESSORY,
}
GRADE_THRESHOLDS = (("SS", 90), ("S", 75), ("A", 60), ("B", 40), ("C", 0))
WEAPON_TYPE_BONUSES = {"SS": 15, "S": 10, "A": 6, "B": 2}
WEAPON_TYPE_TIERS = {
    "SS": ("單手劍", "雙手斧"),
    "S": ("投擲忍具", "短劍", "弓", "雙手槍", "雙手劍", "忍者刀"),
    "A": ("雙手鈍器", "刀", "大太刀"),
    "B": ("單手斧", "單手鈍器"),
}
WEAPON_TYPE_ALIASES = {"大劍": "雙手劍"}
CATEGORY_WEIGHTS = {
    WarehouseCategory.HEAD: {
        "防禦": 1.0,
        "法術防禦": 0.9,
        "生命": 0.8,
        "迴避": 0.8,
        "抵抗": 0.8,
        "行動速度": 0.6,
    },
    WarehouseCategory.WEAPON: {
        "攻擊": 1.2,
        "命中": 1.0,
        "會心": 1.0,
        "防禦貫穿": 1.0,
        "魔力": 1.0,
        "行動速度": 0.6,
    },
    WarehouseCategory.HAND: {
        "攻擊": 1.0,
        "命中": 1.0,
        "會心": 0.9,
        "防禦": 0.7,
        "防禦貫穿": 0.8,
    },
    WarehouseCategory.BODY: {
        "防禦": 1.2,
        "法術防禦": 1.0,
        "生命": 1.0,
        "抵抗": 0.8,
        "迴避": 0.7,
    },
    WarehouseCategory.OFF_HAND: {
        "防禦": 1.0,
        "法術防禦": 0.9,
        "迴避": 1.0,
        "抵抗": 0.8,
        "生命": 0.8,
    },
    WarehouseCategory.FOOT: {
        "迴避": 1.0,
        "防禦": 0.8,
        "行動速度": 1.0,
        "抵抗": 0.7,
        "生命": 0.7,
    },
    WarehouseCategory.ACCESSORY: {
        "攻擊": 0.8,
        "命中": 1.0,
        "會心": 1.0,
        "生命": 0.8,
        "防禦貫穿": 0.8,
        "行動速度": 0.8,
        "抵抗": 0.7,
    },
}
STAT_ALIASES = {
    "HP": "生命",
    "體力": "生命",
    "命中率": "命中",
    "攻擊力": "攻擊",
    "防禦力": "防禦",
    "物防": "防禦",
    "回避": "迴避",
    "速度": "行動速度",
    "貫穿": "防禦貫穿",
    "抗性": "抵抗",
}


@dataclass(frozen=True)
class RatingResult:
    """Transparent rating output including recognized and ignored blessing
    diagnostics.
    """

    score: int
    grade: str
    blessing_score: int
    weapon_bonus: int
    weapon_tier: Optional[str]
    recognized: tuple[tuple[str, float], ...]
    ignored: tuple[str, ...]


class EquipmentRater:
    """Deterministically rate active blessings with centralized category weights."""

    def rate(self, detail: EquipmentDetail) -> RatingResult:
        weights = CATEGORY_WEIGHTS.get(detail.category, {})
        recognized = []
        ignored = []
        total = 0.0
        for blessing in detail.blessings:
            name = STAT_ALIASES.get(blessing.name, blessing.name)
            weight = weights.get(name)
            if weight is None:
                ignored.append(blessing.raw_text)
                continue
            contribution = blessing.value / 5 * weight
            total += contribution
            recognized.append((blessing.raw_text, contribution))
        blessing_score = self._clamp(round(total * 20))
        tier = (
            self.weapon_tier(detail.weapon_type)
            if detail.category is WarehouseCategory.WEAPON
            else None
        )
        bonus = WEAPON_TYPE_BONUSES.get(tier, 0) if tier else 0
        score = self._clamp(blessing_score + bonus)
        grade = next(
            grade for grade, threshold in GRADE_THRESHOLDS if score >= threshold
        )
        return RatingResult(
            score,
            grade,
            blessing_score,
            bonus,
            tier,
            tuple(recognized),
            tuple(ignored),
        )

    @staticmethod
    def weapon_tier(weapon_type: Optional[str]) -> Optional[str]:
        if not weapon_type:
            return None
        canonical = WEAPON_TYPE_ALIASES.get(weapon_type, weapon_type)
        for tier, names in WEAPON_TYPE_TIERS.items():
            if canonical in names:
                return tier
        return None

    @staticmethod
    def _clamp(value: int) -> int:
        return max(0, min(100, value))


RATER = EquipmentRater()
GRADE_ORDER = tuple(grade for grade, _ in GRADE_THRESHOLDS)
_RECOMMEND_GRADE: Optional[str] = None
_SCANNED: list[tuple[EquipmentDetail, RatingResult]] = []


def rating_callback(detail: EquipmentDetail) -> WarehouseDecision:
    """Print one safe rating result and always request a close-only decision."""
    result = RATER.rate(detail)
    _SCANNED.append((detail, result))
    print(
        f"{detail.name}: {result.score}/100 {result.grade}; "
        f"blessings={result.blessing_score}, weapon_bonus={result.weapon_bonus}, "
        f"ignored={list(result.ignored)}"
    )
    state.logger.info(
        "裝備評級: 項目=%s 類別=%s 分數=%d 等級=%s 加護分=%d 武器階級=%s 忽略數=%d",
        detail.name,
        detail.category.value,
        result.score,
        result.grade,
        result.blessing_score,
        result.weapon_tier or "無",
        len(result.ignored),
    )
    return WarehouseDecision(close_detail=True, discard=False, discard_policy=DISCARDS)


def _recommend_threshold_and_label() -> tuple[Optional[int], str]:
    """推荐门槛: 手動指定等級時用該等級分數; 否則用本次掃描的最高等級."""
    thresholds = dict(GRADE_THRESHOLDS)
    if _RECOMMEND_GRADE:
        return thresholds[_RECOMMEND_GRADE], f"等級 {_RECOMMEND_GRADE} 以上"
    if not _SCANNED:
        return None, ""
    best_grade = min(
        (result.grade for _, result in _SCANNED), key=GRADE_ORDER.index
    )
    return thresholds[best_grade], f"本次最高等級 {best_grade} 以上"


def _detail_lines(
    detail: EquipmentDetail, result: RatingResult, recommend: bool
) -> list[str]:
    """完整詳細行, 供主控台與報告檔共用."""
    lines = [
        f"{detail.name} (類別={detail.category.value}) "
        f"星={detail.stars if detail.stars is not None else '?'}",
        f"分數: {result.score}/100 等級: {result.grade} 加護分: {result.blessing_score}",
    ]
    if result.weapon_tier:
        lines.append(f"武器階級: {result.weapon_tier} 加成: {result.weapon_bonus}")
    if detail.blessings:
        lines.append("加護:")
        lines.extend(f"  - {blessing.raw_text}" for blessing in detail.blessings)
    if result.ignored:
        lines.append(f"忽略: {'、'.join(result.ignored)}")
    lines.append(f"推薦: {'是' if recommend else '否'}")
    return lines


def _write_report(recommended) -> Optional[Path]:
    """將所有裝備詳細依評分排序寫入 TEXT 檔案並回傳路徑."""
    if not _SCANNED:
        return None
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    path = (
        results_dir
        / f"equipment_rating_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    recommended_ids = {id(detail) for detail, _ in recommended}
    lines = [
        "========== 裝備評級結果 ==========",
        f"掃描時間: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"總數: {len(_SCANNED)}",
        "",
    ]
    for rank, (detail, result) in enumerate(_SCANNED, start=1):
        detail_lines = _detail_lines(detail, result, id(detail) in recommended_ids)
        lines.append(f"[{rank}] {detail_lines[0]}")
        lines.extend(f"    {line}" for line in detail_lines[1:])
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def finish_callback(final_state) -> None:
    """Print the final scan summary with ranked recommendations."""
    _SCANNED.sort(key=lambda pair: pair[1].score, reverse=True)
    threshold, label = _recommend_threshold_and_label()
    recommended = (
        [
            (detail, result)
            for detail, result in _SCANNED
            if result.score >= threshold
        ]
        if threshold is not None
        else []
    )
    print("========== 裝備評級掃描結束 ==========")
    print(f"共掃描 {len(_SCANNED)} 件裝備")
    if recommended:
        print(f"推薦裝備 ({label}, 共 {len(recommended)} 件):")
        for rank, (detail, result) in enumerate(recommended, start=1):
            detail_lines = _detail_lines(detail, result, True)
            print(f"{rank}. {detail_lines[0]}")
            for line in detail_lines[1:]:
                print(f"   {line}")
    elif _SCANNED:
        print("無推薦裝備")
    path = _write_report(recommended)
    if path is not None:
        print(f"完整結果已寫入: {path}")
    state.logger.info(
        "裝備評級結束: 掃描=%d 推薦=%d 狀態=%s 報告=%s",
        len(_SCANNED),
        len(recommended),
        final_state,
        path,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the required category-selection parser for the rating workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--type",
        action="append",
        choices=tuple(_CATEGORY_TYPES),
        required=True,
        help="Equipment category to rate; repeat to select multiple categories.",
    )
    parser.add_argument(
        "--recommend",
        type=str.upper,
        choices=GRADE_ORDER,
        default=None,
        help="推薦門檻等級 (SS/S/A/B/C); 未指定時自動使用本次掃描的最高等級.",
    )
    return parser


def parse_recommend_grade(args: list[str]) -> Optional[str]:
    """Parse the optional recommendation grade from CLI arguments."""
    return build_argument_parser().parse_args(args).recommend


def parse_categories(args: list[str]) -> tuple[WarehouseCategory, ...]:
    """Parse ordered requested categories; no arguments print help and select
    nothing.
    """
    parser = build_argument_parser()
    if not args:
        parser.print_help()
        return ()
    namespace = parser.parse_args(args)
    return tuple(dict.fromkeys(_CATEGORY_TYPES[value] for value in namespace.type))


def entrypoint(core, args: Optional[list[str]] = None) -> None:
    """Configure OCR only after required equipment categories have been selected."""
    global _RECOMMEND_GRADE

    script_args = sys.argv[1:] if args is None else args
    selected = parse_categories(script_args)
    if not selected:
        return
    _RECOMMEND_GRADE = parse_recommend_grade(script_args)
    from src.ai.warehouse.view import WarehouseView

    state.logger.info(
        "裝備評級流程開始: 類別=%s", [category.value for category in selected]
    )
    _SCANNED.clear()
    core.setup(lang="chinese_cht", debug=DEBUG)
    core.register_ai(
        WarehouseView(
            on_item=rating_callback,
            on_finish=finish_callback,
            categories=selected,
            action_delay=ACTION_DELAY,
            drag_hold_duration=DRAG_HOLD_DURATION,
            drag_move_duration=DRAG_MOVE_DURATION,
            drag_settle_delay=DRAG_SETTLE_DELAY,
        )
    )
    state.logger.info("裝備評級丟棄策略=DENY")
    core.run()


def _run_module(args: list[str]) -> None:
    selected = parse_categories(args)
    if not selected:
        return
    from src.core import VACore

    entrypoint(VACore(), args)


if __name__ == "__main__":
    _run_module(sys.argv[1:])
