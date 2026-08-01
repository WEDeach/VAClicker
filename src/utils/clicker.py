import ctypes
import time
from typing import Optional, Tuple

import vgamepad as vg

from .shared import state
from .window import get_window_handle, get_window_rect

logger = state.logger

INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000
SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", _MOUSEINPUT)]


class _INPUT(ctypes.Structure):
    _anonymous_ = ("_u",)
    _fields_ = [("type", ctypes.c_ulong), ("_u", _INPUT_UNION)]


_INPUT_SIZE = ctypes.sizeof(_INPUT)


def _screen_to_absolute(x: int, y: int) -> Tuple[int, int]:
    user32 = ctypes.windll.user32
    virt_x = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
    virt_y = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
    virt_w = user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
    virt_h = user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)
    abs_x = int((x - virt_x) * 65535 / virt_w)
    abs_y = int((y - virt_y) * 65535 / virt_h)
    return abs_x, abs_y


def _send_input_mouse(flags: int, x: int, y: int) -> bool:
    try:
        win32u = ctypes.WinDLL("win32u.dll")
    except Exception as e:
        logger.warning("win32u.dll not available: %s", e)
        return False
    win32u.NtUserSendInput.restype = ctypes.c_uint
    win32u.NtUserSendInput.argtypes = [
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    abs_x, abs_y = _screen_to_absolute(x, y)
    inp = _INPUT()
    inp.type = INPUT_MOUSE
    inp.mi.dwFlags = flags
    inp.mi.dx = abs_x
    inp.mi.dy = abs_y
    inp.mi.mouseData = 0
    inp.mi.time = 0
    inp.mi.dwExtraInfo = None
    return win32u.NtUserSendInput(1, ctypes.byref(inp), _INPUT_SIZE) == 1


def calculate_click_point(
    match_loc: Tuple[int, int],
    template_shape: Tuple[int, int],
) -> Tuple[int, int]:
    left, top, _, _ = get_window_rect()
    template_width, template_height = template_shape
    click_x = left + match_loc[0] + template_width // 2
    click_y = top + match_loc[1] + template_height // 2
    return click_x, click_y


def move_cursor_to(point: Tuple[int, int]) -> None:
    """Move the mouse cursor to the given absolute screen coordinates."""
    _send_input_mouse(
        MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK,
        point[0],
        point[1],
    )


def click_at(
    point: Tuple[int, int], hwnd: Optional[int] = None, retry: int = 3
) -> None:
    if hwnd is None:
        hwnd = get_window_handle()

    x, y = point
    if not _send_input_mouse(
        MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK, x, y
    ):
        if retry > 0:
            time.sleep(5)
            logger.warning(
                "無法移動游標到指定位置，%d 次重試剩餘%d次", 3 - retry + 1, retry - 1
            )
            return click_at(point, hwnd, retry - 1)
        raise RuntimeError("無法移動游標到指定位置，請確認程式有足夠權限")
    time.sleep(0.05)

    if not _send_input_mouse(
        MOUSEEVENTF_LEFTDOWN | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK, x, y
    ):
        if retry > 0:
            time.sleep(5)
            logger.warning(
                "無法移動游標到指定位置，%d 次重試剩餘%d次", 3 - retry + 1, retry - 1
            )
            return click_at(point, hwnd, retry - 1)
        raise RuntimeError("無法移動游標到指定位置，請確認程式有足夠權限")
    time.sleep(0.05)

    if not _send_input_mouse(
        MOUSEEVENTF_LEFTUP | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK, x, y
    ):
        if retry > 0:
            time.sleep(5)
            logger.warning(
                "無法移動游標到指定位置，%d 次重試剩餘%d次", 3 - retry + 1, retry - 1
            )
            return click_at(point, hwnd, retry - 1)
        raise RuntimeError("無法移動游標到指定位置，請確認程式有足夠權限")


def _ensure_gamepad():
    if not getattr(_ensure_gamepad, "_gamepad", None):
        try:
            _ensure_gamepad._gamepad = vg.VX360Gamepad()
            logger.debug("[gamepad] 虛擬 Xbox 360 手把初始化成功")
        except Exception as e:
            logger.error(
                "[gamepad] 虛擬手把初始化失敗：%s（請確認 ViGEmBus 驅動已安裝）",
                e,
            )
            _ensure_gamepad._gamepad = None
        time.sleep(5)  # 等待驅動準備就緒
    return _ensure_gamepad._gamepad


def click_by_gamepad(btn, release_delay=0.1, retry=3):
    gp = _ensure_gamepad()
    if gp is None:
        logger.warning("[gamepad] 手把不可用")
        if retry > 0:
            click_by_gamepad(btn, release_delay, retry - 1)
        return

    gp.press_button(button=btn)
    gp.update()
    time.sleep(release_delay)
    gp.release_button(button=btn)
    gp.update()


def set_gamepad_stick(x_value=0, y_value=0, duration=0.0):
    """Set the virtual Xbox 360 left analog stick to (x_value, y_value).

    Args:
        x_value: X axis value from -32768 to 32767 (0 is neutral).
        y_value: Y axis value from -32768 to 32767 (0 is neutral).
        duration: Seconds to hold the stick before returning to neutral.
    """
    gp = _ensure_gamepad()
    if gp is None:
        logger.warning("[gamepad] 手把不可用")
        return

    gp.left_joystick(x_value, y_value)
    gp.update()
    if duration > 0:
        time.sleep(duration)
        gp.left_joystick(0, 0)
        gp.update()
