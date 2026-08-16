import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

from src.log import logging


def load_text_mapping() -> Dict[str, Dict[str, str]]:
    path = Path(__file__).with_name("text_mapping.json")
    with path.open(encoding="utf-8-sig") as file:
        mapping = json.load(file)
    if not isinstance(mapping, dict):
        raise ValueError("Text mapping must be a JSON object")
    if not all(
        isinstance(lang, str)
        and isinstance(values, dict)
        and all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in values.items()
        )
        for lang, values in mapping.items()
    ):
        raise ValueError(
            "Text mapping must be a mapping of languages to string mappings"
        )
    return mapping


@dataclass
class SharedState:
    title: str = "WizardryVariantsDaphne"
    lang = "chinese_cht"
    logger = logging
    ocr = None
    text_mapping: Dict[str, Dict[str, str]] = field(default_factory=load_text_mapping)


state = SharedState()
