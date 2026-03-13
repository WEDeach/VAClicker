import argparse
import datetime
import os

import pyautogui
import win32gui


def switch_window(title: str):
    if not title:
        title = "WizardryVariantsDaphne"
    hwnd = win32gui.FindWindow(None, title)
    if hwnd:
        win32gui.SetForegroundWindow(hwnd)
        return hwnd
    else:
        raise RuntimeError(f"Unable to find window: {title}")


def save_screen(hwnd: int, save_path: str):
    cr = win32gui.GetClientRect(hwnd)
    left, top = win32gui.ClientToScreen(hwnd, (0, 0))
    width = cr[2]
    height = cr[3]
    if width == 0 or height == 0:
        raise ValueError("Window size is 0, cannot take screenshot")
    screenshot = pyautogui.screenshot(region=(left, top, width, height))
    if not save_path:
        save_path = os.path.join(os.getcwd(), ".dumps")
    _, ext = os.path.splitext(save_path)
    if not ext:
        os.makedirs(save_path, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_path, f"{timestamp}.png")
    screenshot.save(save_path)
    print(f"Screenshot saved to {save_path}")


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("-t", "--title", type=str, help="The target window title")
    parse.add_argument(
        "-o", "--out_path", type=str, help="The path to save the screenshot"
    )
    args = parse.parse_args()
    hwnd = switch_window(args.title)
    save_screen(hwnd, args.out_path)