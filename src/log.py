import logging as _logging
import sys

_logging.basicConfig(
    level=_logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    stream=sys.stdout,
)

logging = _logging.getLogger("VAClicker")