import time

import vgamepad

from src.ai.battle.auto_mode import AutoMode
from src.ai.btn._return import Return
from src.ai.btn.chest_action import ChestAction
from src.ai.btn.path.chest import ChestPath
from src.ai.btn.recovery import Recovery
from src.ai.btn.retry import Retry
from src.ai.dialog.text_with_auto import TextWithAuto
from src.ai.view.town import TownView
from src.core import VACore
from src.utils.clicker import (
    calculate_click_point,
    click_at,
    click_by_gamepad,
    move_cursor_to,
)
from src.utils.image import load_template, match_template
from src.utils.shared import state
from src.utils.window import get_window_rect, get_window_screen

# Script Core Configuration

DEBUG = True
EVENT_NAME = "FFXI 深暗幻想之城"


# Script AI Configuration

LANG = "chinese_cht"  # 語言 用於OCR文字識別, 非繁體一定要修改並更改其餘文字設定
CHEST_SELECT_INDEX = 2  # 開寶箱角色順位 從0開始
WORLD_MAP_STATE = {"scrolls": 0}
FFXI_TOWN_STATE = {"event_done": False}
FFXI_IN_DUNGEON = False

CHEST_OPEN_PATTERN = "打開\n\\w*?都不做"
HEALTH_CHECK_TEXT = "意志力"
HEAL_BUTTON_TEXT = "回復"

TEXT_MAPPING = {
    "chest.action.open": CHEST_OPEN_PATTERN,
}

## FFXI地圖入口點設定
DUNGEON_SELECT_TITLE = "北穿幽靈城"
DUNGEON_SELECT_TITLE_REGION = (0.35, 0.65, 0.04, 0.16)
DUNGEON_SELECT_TARGET = "第三區"
DUNGEON_SELECT_LIST_REGION = (0.30, 0.75, 0.18, 0.75)
DUNGEON_SELECT_RETURN_TEXT = "開啟世界地圖"
DUNGEON_SELECT_RETURN_ONLY = False
DUNGEON_SELECT_WAIT = 5.0

## 世界地圖設定
WORLD_MAP_TARGET = "王都盧庫納里亞"
WORLD_MAP_UI_TEXT = ("放大", "縮小")
WORLD_MAP_UI_REGION = (0.82, 0.98, 0.70, 0.90)
WORLD_MAP_TARGET_REGION = (0.10, 0.90, 0.10, 0.85)
WORLD_MAP_SCROLL_DOWN_DURATION = 1.0
WORLD_MAP_SETTLE_DELAY = 1.0
WORLD_MAP_MAX_SCROLLS = 5
WORLD_MAP_SCROLL_BORDER_OFFSET = 5

## 旅館與EVENT入口點設定
FFXI_TOWN_CONFIRM_REGION = (0.34, 0.39, 0.37, 0.43)
FFXI_EVENT_TEMPLATE = "BTN_EVENT.png"
FFXI_EVENT_SEARCH_REGION = (0.50, 0.70, 0.58, 1.00)
FFXI_EVENT_BUTTON_THRESHOLD = 0.80
FFXI_EVENT_MENU_TITLE = "深暗幻想之城"
FFXI_EVENT_MENU_TARGET = "北穿幽靈城"
FFXI_EVENT_MENU_TITLE_REGION = (0.30, 0.70, 0.05, 0.15)
FFXI_EVENT_MENU_BUTTONS_REGION = (0.30, 0.70, 0.58, 0.75)
FFXI_EVENT_MENU_WAIT = 5.0


EYES = {
    "retry": Retry(),
    "text_with_auto": TextWithAuto(),
    "chest_action": ChestAction(select_index=CHEST_SELECT_INDEX),
    "chest_path": ChestPath(),
    "return": Return(need_ret_inn=False),
    "recovery": Recovery(),
    "auto_mode": AutoMode(
        template="ICON_auto_icon.png",
        threshold=0.8,
        region=(0.55, 0.70, 0.60, 0.75),
    ),
    "town": TownView(enable_dungeon=False),
}


def entrypoint(core: "VACore"):
    print(f"Script: {EVENT_NAME}")

    core.setup(lang=LANG, debug=DEBUG, text_mapping=TEXT_MAPPING)

    eyes = [
        EYES["retry"],
        ffxi_recovery,
        EYES["text_with_auto"],
        EYES["chest_action"],
        EYES["return"],
        EYES["recovery"],
        EYES["auto_mode"],
        ffxi_town,
        ffxi_event_menu,
        ffxi_dungeon_select,
        ffxi_world_map,
        ffxi_chest_path,
    ]

    EYES["town"].core = core

    for eye in eyes:
        core.register_ai(eye)

    core.run()


def ffxi_recovery() -> bool:
    chest_action: ChestAction = EYES["chest_action"]
    chest_path: ChestPath = EYES["chest_path"]
    if chest_action.last_unlock_time > 0:
        if chest_path.check(check_only=True):
            left, top, width, height = get_window_rect()
            center_x = int(left + width // 2)
            center_y = int(top + height * 0.95)
            state.logger.debug("嘗試恢復狀態...")
            click_at((center_x, center_y))
            time.sleep(5)

            def _check_for_chara_heath():
                _screen = get_window_screen()
                _match = match_template(
                    _screen,
                    None,
                    1,
                    False,
                    None,
                    ocr_check=[(HEALTH_CHECK_TEXT, 0)],
                    region=(0.33, 0.41, 0.5, 0.55),
                )
                return bool(_match)

            def _check_for_heal_btn(click=False):
                _screen = get_window_screen()
                _match = match_template(
                    _screen,
                    None,
                    1,
                    False,
                    None,
                    ocr_check=[(HEAL_BUTTON_TEXT, 0)],
                    region=(0.52, 0.60, 0.71, 0.77),
                )
                if _match:
                    if click:
                        _loc, _ = _match
                        point = calculate_click_point(_loc, (0, 0))
                        click_at(point)
                        time.sleep(8.0)
                    return True
                return False

            if _check_for_chara_heath():
                _x = int(left + width * 0.635)
                _y = int(top + height * 0.52)
                click_at((_x, _y))
                time.sleep(5)
                if not _check_for_heal_btn():
                    click_at((_x, _y))
                    time.sleep(5)
                _check_for_heal_btn(True)
                click_by_gamepad(vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_B)
                time.sleep(2)
                click_by_gamepad(vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_B)
                chest_action.last_unlock_time = 0
            else:
                state.logger.warning("無法進入回復界面...")
                click_by_gamepad(vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_B)
                time.sleep(3)
            return True
    return False


def ffxi_town() -> bool:
    global FFXI_TOWN_STATE

    _return: Return = EYES["return"]
    _screen = get_window_screen()
    _inn = _find_ocr_text_center(_screen, "旅店", FFXI_TOWN_CONFIRM_REGION)
    if _inn is None:
        return False

    if _return.need_ret_inn:
        FFXI_TOWN_STATE["event_done"] = False
        return False

    if not FFXI_TOWN_STATE["event_done"]:
        _temp, _mask = load_template(FFXI_EVENT_TEMPLATE, grayscale=False)
        _match = match_template(
            _screen,
            _temp,
            FFXI_EVENT_BUTTON_THRESHOLD,
            False,
            _mask,
            region=FFXI_EVENT_SEARCH_REGION,
        )
        if _match:
            _loc, _ = _match
            _point = calculate_click_point(_loc, (_temp.shape[1], _temp.shape[0]))
            state.logger.info("進入 FFXI 活動")
            click_at(_point)
            time.sleep(5)
            FFXI_TOWN_STATE["event_done"] = True
            return True

        state.logger.debug("在城鎮中，尚未找到 EVENT 按鈕")
        return False

    return True


def ffxi_event_menu() -> bool:
    _screen = get_window_screen()
    _match = match_template(
        _screen,
        None,
        0.8,
        False,
        None,
        ocr_check=[(FFXI_EVENT_MENU_TITLE, 0), (FFXI_EVENT_MENU_TARGET, 0)],
        region=FFXI_EVENT_MENU_TITLE_REGION,
    )
    if not _match:
        return False

    state.logger.info("偵測到 FFXI 活動選單")

    _point = _find_ocr_text_center(
        _screen, FFXI_EVENT_MENU_TARGET, FFXI_EVENT_MENU_BUTTONS_REGION
    )
    if _point:
        _point = calculate_click_point(_point, (0, 0))
        state.logger.info("選擇 北穿幽靈城")
        click_at(_point)
        time.sleep(FFXI_EVENT_MENU_WAIT)
        FFXI_TOWN_STATE["event_done"] = False
        return True

    state.logger.warning("找不到 北穿幽靈城 按鈕")
    return False


def _find_ocr_text_center(screen, text, region, *, prefer="first"):
    x1, x2, y1, y2 = region
    h, w = screen.shape[:2]
    x1 = int(x1 * w)
    x2 = int(x2 * w)
    y1 = int(y1 * h)
    y2 = int(y2 * h)
    crop = screen[y1:y2, x1:x2]
    results = state.ocr.predict(crop)
    if not results:
        return None
    candidates = []
    for page in results:
        if page is None:
            continue
        rec_texts = page.get("rec_texts") if hasattr(page, "get") else None
        rec_boxes = page.get("rec_boxes") if hasattr(page, "get") else None
        if not rec_texts or rec_boxes is None or len(rec_texts) == 0:
            continue
        for rec_text, box in zip(rec_texts, rec_boxes):
            if text in rec_text:
                x1b, y1b, x2b, y2b = (int(v) for v in box)
                cx = (x1b + x2b) // 2 + x1
                cy = (y1b + y2b) // 2 + y1
                candidates.append((cy, cx, cy))
    if not candidates:
        return None
    if prefer == "bottom":
        return max(candidates, key=lambda p: p[0])[1:]
    if prefer == "top":
        return min(candidates, key=lambda p: p[0])[1:]
    return candidates[0][1:]


def ffxi_dungeon_select() -> bool:
    global FFXI_IN_DUNGEON

    _screen = get_window_screen()
    _match = match_template(
        _screen,
        None,
        0.8,
        False,
        None,
        ocr_check=[(DUNGEON_SELECT_TITLE, 0), ("Dungeon Select", 0)],
        region=DUNGEON_SELECT_TITLE_REGION,
    )
    if not _match:
        return False

    state.logger.info("偵測到地下城選擇畫面")

    _return: Return = EYES["return"]
    if FFXI_IN_DUNGEON:
        _return.increment_battle()
        FFXI_IN_DUNGEON = False
        state.logger.info(
            "離開地下城，累積戰鬥次數 %d/%d",
            _return.current_battle_num,
            _return.max_battle_num,
        )

    if DUNGEON_SELECT_RETURN_ONLY or _return.need_ret_inn:
        _point = _find_ocr_text_center(
            _screen, DUNGEON_SELECT_RETURN_TEXT, DUNGEON_SELECT_LIST_REGION
        )
        if _point:
            _point = calculate_click_point(_point, (0, 0))
            state.logger.info("選擇返回城鎮")
            click_at(_point)
            time.sleep(DUNGEON_SELECT_WAIT)
            _return.need_ret_inn = False
            return True
        state.logger.warning("找不到開啟世界地圖按鈕")
        return True

    _point = _find_ocr_text_center(
        _screen, DUNGEON_SELECT_TARGET, DUNGEON_SELECT_LIST_REGION, prefer="bottom"
    )
    if _point:
        _point = calculate_click_point(_point, (0, 0))
        state.logger.info("選擇 %s", DUNGEON_SELECT_TARGET)
        click_at(_point)
        time.sleep(DUNGEON_SELECT_WAIT)
        return True

    state.logger.warning("找不到 %s", DUNGEON_SELECT_TARGET)
    return True


def ffxi_world_map() -> bool:
    global WORLD_MAP_STATE

    _screen = get_window_screen()
    _match = match_template(
        _screen,
        None,
        0.8,
        False,
        None,
        ocr_check=[(_t, 0) for _t in WORLD_MAP_UI_TEXT],
        region=WORLD_MAP_UI_REGION,
    )
    if not _match:
        WORLD_MAP_STATE["scrolls"] = 0
        return False

    state.logger.info("偵測到世界地圖")

    while WORLD_MAP_STATE["scrolls"] < WORLD_MAP_MAX_SCROLLS:
        _point = _find_ocr_text_center(
            _screen, WORLD_MAP_TARGET, WORLD_MAP_TARGET_REGION
        )
        if _point:
            _point = calculate_click_point(_point, (0, 0))
            state.logger.info("前往王都盧庫納里亞")
            click_at(_point)
            time.sleep(5)
            WORLD_MAP_STATE["scrolls"] = 0
            return True

        state.logger.info("向下捲動世界地圖...")
        _left, _top, _w, _h = get_window_rect()
        _down_x = _left + _w // 2
        _down_y = _top + _h - WORLD_MAP_SCROLL_BORDER_OFFSET
        _safe_x = _left + _w // 2
        _safe_y = _top + _h // 2

        move_cursor_to((_down_x, _down_y))
        time.sleep(WORLD_MAP_SCROLL_DOWN_DURATION)
        move_cursor_to((_safe_x, _safe_y))
        time.sleep(WORLD_MAP_SETTLE_DELAY)
        WORLD_MAP_STATE["scrolls"] += 1
        _screen = get_window_screen()

    state.logger.warning("已達最大捲動次數，仍找不到王都盧庫納里亞")
    return True


def ffxi_chest_path() -> bool:
    global FFXI_IN_DUNGEON
    if EYES["chest_path"].check():
        if not FFXI_IN_DUNGEON:
            FFXI_IN_DUNGEON = True
            state.logger.info("已設置進入地下城標記")
        return True
    return False


if __name__ == "__main__":
    entrypoint(VACore())
