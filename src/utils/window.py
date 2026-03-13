import ctypes
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import win32gui
import win32ui

from .shared import state

# idk why hell fking works
ctypes.windll.shcore.SetProcessDpiAwareness(2)


def switch_window(hwnd: int, *, delay: float = 0.1):
    win32gui.SetForegroundWindow(hwnd)
    time.sleep(delay)


def get_window_handle(title: Optional[str] = None):
    if title is None:
        title = state.title
    hwnd = win32gui.FindWindow(None, title)
    if hwnd:
        return hwnd
    else:
        raise RuntimeError(f"Unable to find window: {title}")


def get_foreground_window():
    hwnd = win32gui.GetForegroundWindow()
    if hwnd:
        return hwnd


def get_window_screen(hwnd: Optional[int] = None, *, color_code=cv2.COLOR_BGRA2BGR):
    if hwnd is None:
        hwnd = get_window_handle()
    cr = win32gui.GetClientRect(hwnd)
    width, height = cr[2], cr[3]
    if width == 0 or height == 0:
        raise ValueError("Window size is 0, cannot take screenshot")
    hwnd_dc = win32gui.GetDC(hwnd)
    mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
    save_dc = mfc_dc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(mfc_dc, width, height)
    save_dc.SelectObject(bmp)
    ctypes.windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 1 << 0 | 0x00000002)
    bmp_info = bmp.GetInfo()
    raw = bmp.GetBitmapBits(True)
    save_dc.DeleteDC()
    mfc_dc.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwnd_dc)
    win32gui.DeleteObject(bmp.GetHandle())
    img = np.frombuffer(raw, dtype=np.uint8).reshape(
        bmp_info["bmHeight"], bmp_info["bmWidth"], 4
    )
    return cv2.cvtColor(img, color_code)


def get_window_rect(hwnd: Optional[int] = None):
    if hwnd is None:
        hwnd = get_window_handle()
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    left, top = win32gui.ClientToScreen(hwnd, (0, 0))
    return left, top, right, bottom


def dump4log(img: np.ndarray, label: str) -> None:
    log_dir = Path(__file__).parent.parent.parent / ".logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{label}.png"
    path = log_dir / filename
    ok, buf = cv2.imencode(".png", img)
    if ok:
        path.write_bytes(buf.tobytes())
    state.logger.info("已將相關截圖保存到 %s", path)
