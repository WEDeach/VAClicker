import ctypes
import time
from typing import Optional, Tuple
import vgamepad as vg

from .window import get_window_handle, get_window_rect
from .shared import state

logger = state.logger


def calculate_click_point(
    match_loc: Tuple[int, int],
    template_shape: Tuple[int, int],
) -> Tuple[int, int]:
    left, top, _, _ = get_window_rect()
    template_width, template_height = template_shape
    click_x = left + match_loc[0] + template_width // 2
    click_y = top + match_loc[1] + template_height // 2
    return click_x, click_y


def click_at(
    point: Tuple[int, int], hwnd: Optional[int] = None, retry: int = 3
) -> None:
    if hwnd is None:
        hwnd = get_window_handle()

    x, y = point
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    try:
        win32u = ctypes.WinDLL("win32u.dll")
        win32u.NtUserSetCursorPos.restype = ctypes.c_bool
        win32u.NtUserSetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
        kernel32.SetLastError(0)
    except Exception as e:
        logger.warning("NtUserSetCursorPos declined: %s", e)

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

    win32u.NtUserSendInput.restype = ctypes.c_uint
    win32u.NtUserSendInput.argtypes = [
        ctypes.c_uint,  # cInputs
        ctypes.c_void_p,  # pInputs
        ctypes.c_int,  # cbSize
    ]

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
    virt_x = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
    virt_y = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
    virt_w = user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
    virt_h = user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)
    abs_x = int((x - virt_x) * 65535 / virt_w)
    abs_y = int((y - virt_y) * 65535 / virt_h)

    def _make_mouse_input(flags, dx=0, dy=0):
        inp = _INPUT()
        inp.type = INPUT_MOUSE
        inp.mi.dwFlags = flags
        inp.mi.dx = dx
        inp.mi.dy = dy
        inp.mi.mouseData = 0
        inp.mi.time = 0
        inp.mi.dwExtraInfo = None
        return inp

    sz = ctypes.sizeof(_INPUT)
    inp_move = _make_mouse_input(
        MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK,
        abs_x,
        abs_y,
    )
    ret_move = win32u.NtUserSendInput(1, ctypes.byref(inp_move), sz)
    time.sleep(0.05)

    inp_dn = _make_mouse_input(
        MOUSEEVENTF_LEFTDOWN | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK,
        abs_x,
        abs_y,
    )
    ret_dn = win32u.NtUserSendInput(1, ctypes.byref(inp_dn), sz)
    time.sleep(0.05)

    inp_up = _make_mouse_input(
        MOUSEEVENTF_LEFTUP | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK,
        abs_x,
        abs_y,
    )
    ret_up = win32u.NtUserSendInput(1, ctypes.byref(inp_up), sz)

    click_sent = ret_move == 1 and ret_dn == 1 and ret_up == 1

    if not click_sent:
        if retry > 0:
            time.sleep(5)
            logger.warning(
                "無法移動游標到指定位置，%d 次重試剩餘%d次", 3 - retry + 1, retry - 1
            )
            return click_at(point, hwnd, retry - 1)
        raise RuntimeError("無法移動游標到指定位置，請確認程式有足夠權限")


def click_by_gamepad(btn, release_delay=0.1, retry=3):
    if not getattr(click_by_gamepad, "_gamepad", None):
        try:
            click_by_gamepad._gamepad = vg.VX360Gamepad()
            logger.debug("[gamepad] 虛擬 Xbox 360 手把初始化成功")
        except Exception as e:
            logger.error(
                "[gamepad] 虛擬手把初始化失敗：%s（請確認 ViGEmBus 驅動已安裝）",
                e,
            )
            click_by_gamepad._gamepad = None
        time.sleep(5)  # 等待驅動準備就緒

    gp = click_by_gamepad._gamepad
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
