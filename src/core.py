import datetime
import logging as _logging
import threading
import time
from pathlib import Path
from typing import Type

from pynput import keyboard

from . import AI, T
from .ocr import get_paddle_ocr
from .utils.shared import state
from .utils.text_map import update_text_mapping
from .utils.window import get_foreground_window, get_window_handle

logger = state.logger


class VACore:
    def __init__(self, *, signal: threading.Event = None):
        self.ais = []
        self._paused = False
        self._pause_lock = threading.Event()
        self._pause_lock.set()

    @property
    def logger(self):
        return logger

    def check_window(self):
        f = get_foreground_window()
        t = get_window_handle()
        return f == t

    def setup(self, *, lang="chinese_cht", text_mapping=None, debug=False):
        # logger
        logger.setLevel(_logging.DEBUG if debug else _logging.INFO)
        log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = (
            log_dir
            / f"vaclicker_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        _fh = _logging.FileHandler(log_file, encoding="utf-8")
        _fh.setLevel(logger.level)
        _fh.setFormatter(_logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
        logger.addHandler(_fh)

        logger.debug("正在初始化...")
        if text_mapping:
            update_text_mapping(lang, text_mapping)
        state.ocr = get_paddle_ocr(lang)
        logger.debug("初始化完成")

    def register_ai(self, ai: AI):
        ai.core = self
        self.ais.append(ai)

    def find_ai(self, ai_type: Type[T]):
        for ai in self.ais:
            if isinstance(ai, ai_type):
                return ai
        return None

    def _on_pause(self):
        if not self._paused:
            self._paused = True
            self._pause_lock.clear()
            logger.info("已暫停 (F6 恢復)")

    def _on_resume(self):
        if self._paused:
            self._paused = False
            self._pause_lock.set()
            logger.info("已恢復")

    def _on_press(self, key):
        if key == keyboard.Key.f5:
            self._on_pause()
        elif key == keyboard.Key.f6:
            self._on_resume()

    def run(self, *, interval=0.01):
        listener = keyboard.Listener(on_press=self._on_press)
        listener.start()
        while True:
            try:
                if not self._pause_lock.wait(timeout=0.5):
                    continue
                if self.check_window():
                    for ai in self.ais:
                        if callable(ai):
                            if ai():
                                state.logger.debug(f"AI Success: {ai.__name__}")
                                break
                        elif ai.check():
                            state.logger.debug(f"AI Success: {ai.__class__.__name__}")
                            break
                time.sleep(interval)
            except KeyboardInterrupt:
                break
        if state.ocr is not None:
            state.ocr.close()
            state.ocr = None
        listener.stop()
