"""Reusable OCR-driven warehouse workflow exports."""

from .models import (
    EQUIPMENT_CATEGORIES,
    Blessing,
    DestructiveActionPolicy,
    EquipmentDetail,
    ItemRow,
    WarehouseCategory,
    WarehouseDecision,
)


def __getattr__(name: str):
    """Lazily load platform-bound workflow code only when requested."""
    if name == "WarehouseView":
        from .view import WarehouseView

        return WarehouseView
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EQUIPMENT_CATEGORIES",
    "Blessing",
    "DestructiveActionPolicy",
    "EquipmentDetail",
    "ItemRow",
    "WarehouseCategory",
    "WarehouseDecision",
    "WarehouseView",
]
