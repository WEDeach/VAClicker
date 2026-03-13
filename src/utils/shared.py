from dataclasses import dataclass, field
from typing import Dict

from src.log import logging


@dataclass
class SharedState:
    title: str = "WizardryVariantsDaphne"
    lang = "chinese_cht"
    logger = logging
    ocr = None
    text_mapping: Dict[str, Dict[str, str]] = field(default_factory=dict)


state = SharedState()

# TODO: load from file
state.text_mapping = {
    "chinese_cht": {
        "btn.chest.action.open": "打開\n\\w*?都不做",
        "btn.retry": "重試",
    }
}
