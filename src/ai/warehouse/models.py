"""Pure data contracts for warehouse OCR and workflow decisions."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class WarehouseCategory(str, Enum):
    """Warehouse tabs recognized by the reusable workflow."""

    ALL = "all"
    HEAD = "head"
    WEAPON = "weapon"
    HAND = "hand"
    BODY = "body"
    OFF_HAND = "off_hand"
    FOOT = "foot"
    ACCESSORY = "accessory"
    CONSUMABLE = "consumable"
    COLLECTION = "collection"
    FILTER = "filter"


EQUIPMENT_CATEGORIES = (
    WarehouseCategory.HEAD,
    WarehouseCategory.WEAPON,
    WarehouseCategory.HAND,
    WarehouseCategory.BODY,
    WarehouseCategory.OFF_HAND,
    WarehouseCategory.FOOT,
    WarehouseCategory.ACCESSORY,
)


class DestructiveActionPolicy(str, Enum):
    """Controls whether a callback may request a destructive discard action."""

    DENY = "deny"
    ALLOW = "allow"


@dataclass(frozen=True)
class Blessing:
    """One active additional blessing parsed from OCR."""

    name: str
    value: int
    is_percent: bool
    raw_text: str


@dataclass(frozen=True)
class ItemRow:
    """An item-list row with a page-local stable OCR signature."""

    name: str
    stars: Optional[int]
    row_y: int
    signature: str
    raw_text: str = ""


@dataclass(frozen=True)
class EquipmentDetail:
    """OCR-derived detail data for one warehouse item."""

    name: str
    category: WarehouseCategory
    stars: Optional[int] = None
    weapon_type: Optional[str] = None
    blessings: tuple[Blessing, ...] = ()
    raw_text: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WarehouseDecision:
    """Callback result; details close by default and discard requires explicit
    opt-in.
    """

    close_detail: bool = True
    discard: bool = False
    discard_policy: DestructiveActionPolicy = DestructiveActionPolicy.DENY
