"""FFXI Event.

前置條件:
 - 畫面 1920x1080 | 遊戲亮度 100%
 - 戰鬥中「自動」與「倍速」按鈕預設自動開啟
   (BattleToggle 預設 True; 可改為 False 或 CLI 覆寫)
 - 活動地圖請先完整開圖 至少到3F, 如果你只想單刷1F/5F 請自行更改 DUNGEON_SELECT_TARGET
 - 如果你想要調整開寶箱角色順位 請自行更改 CHEST_SELECT_INDEX
 - 回復的選項建議打開 "優先使用道具" 以免道具占用
 - 同上, 建議打開物資補充功能, 並使用 "替換" 並確保有足夠空間
 - 同上, 建議打開物資補充功能中的 "回旅館時補充" 選項
 - 非常重要: 請務必進入地圖內設置"標記"在出口位置, 每層樓都要設置

運作流程:
 - 腳本採用倒走方式運作, 先探寶箱 -> 使用標記至出口:
    1. (3F > 2F > 1F > 入口)
    2. (5F > 4F > 回歸)
 - 最多進入地圖3次, 之後會回旅館休息, 次數可修改 BATTLE_MAX_COUNT

指令列覆寫:
 - 支援 --變數=值 或 --變數 值 (名稱不區分大小寫, 底線/連字號等效, 唯一前綴可縮寫)
 - 例: python scripts/ffxi_event.py --CHEST_SELECT_INDEX=1 --BATTLE_MAX_COUNT=3
 - 例: python scripts/ffxi_event.py --chest-select-index 1 --allow-revive false
"""

import sys
import time
from enum import IntEnum

import vgamepad

from src.ai.battle.battle_toggle import BattleToggle
from src.ai.btn._return import Return
from src.ai.btn.battle_pause import BattlePause
from src.ai.btn.chest_action import ChestAction
from src.ai.btn.path.chest import ChestPath, PathMode
from src.ai.btn.recovery import Recovery
from src.ai.btn.retry import Retry
from src.ai.btn.server_error import ServerError
from src.ai.dialog.low_willpower import LowWillpowerDialog
from src.ai.dialog.text_with_auto import TextWithAuto
from src.ai.inventory import InventoryOrganizer
from src.ai.startup import AnnouncementScreen, LoginScreen, NoticeScreen
from src.ai.view.death import DeathView
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
WORLD_MAP_STATE = {"scrolls": 0, "warned": False}
FFXI_TOWN_STATE = {"event_done": False}
FFXI_IN_DUNGEON = False
BATTLE_MAX_COUNT = 1  # 回旅館前的最大進入地圖次數
ALLOW_REVIVE = True  # 是否允許自動復活(玩家)
CHARACTER_REVIVE_DRY_RUN = False
# 寶箱路徑模式: PathMode.CHEST_AND_MARK(0, 預設)=先點寶箱路徑, 找不到才點路標;
#               PathMode.MARK_ONLY(1)=一開始直接點擊路標路徑
CHEST_PATH_MODE = PathMode.CHEST_AND_MARK

# 背包整理(城鎮入口): 整理各角色背包「其它」欄位物品放入倉庫
INVENTORY_ORGANIZE_ENABLED = True  # 是否啟用背包整理
ALLOW_OTHER_DEPOSIT = True  # 是否允許將「其它」物品放入倉庫

# 恢復狀態: 偵測到「路徑」icon 時自動恢復, 以冷卻時間防止過度恢復
RECOVERY_COOLDOWN = 40.0  # 路徑icon觸發恢復的最小間隔(秒)
RECOVERY_STATE = {"last_time": 0.0}

CHEST_OPEN_PATTERN = "打開\n\\w*?都不做"
HEALTH_CHECK_TEXT = "意志力"
HEAL_BUTTON_TEXT = "回復"

TEXT_MAPPING = {
    "btn.chest.action.open": CHEST_OPEN_PATTERN,
}

## FFXI地圖入口點設定
DUNGEON_SELECT_TITLE = "北穿幽靈城"
DUNGEON_SELECT_TITLE_REGION = (0.35, 0.65, 0.04, 0.16)
DUNGEON_SELECT_TARGET = "第五區"
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
FFXI_EVENT_BUTTON_THRESHOLD = 0.75
FFXI_EVENT_MENU_TITLE = "深暗幻想之城"
FFXI_EVENT_MENU_TARGET = "北穿幽靈城"
FFXI_EVENT_MENU_TITLE_REGION = (0.30, 0.70, 0.05, 0.15)
FFXI_EVENT_MENU_BUTTONS_REGION = (0.30, 0.70, 0.58, 0.75)
FFXI_EVENT_MENU_WAIT = 5.0


# 常用縮寫別名(優先於前綴比對, 避免 --chest 等前綴歧義)
CLI_ALIASES = {
    "chest": "CHEST_SELECT_INDEX",
    "battle": "BATTLE_MAX_COUNT",
    "revive": "ALLOW_REVIVE",
    "dry-run": "CHARACTER_REVIVE_DRY_RUN",
    "organize": "INVENTORY_ORGANIZE_ENABLED",
    "deposit": "ALLOW_OTHER_DEPOSIT",
    "dungeon": "DUNGEON_SELECT_TARGET",
    "world-map": "WORLD_MAP_TARGET",
}


def cli_normalise_name(name: str) -> str:
    return name.replace("_", "-").lower()


def cli_coerce(value: str, current):
    # IntEnum 需先於 int 檢查(它是 int 子類), 並驗證值是否為有效成員
    if isinstance(current, IntEnum):
        try:
            return type(current)(int(value))
        except (ValueError, TypeError) as exc:
            valid = ", ".join(str(member.value) for member in type(current))
            raise ValueError(
                f"無效的 {type(current).__name__} 值: {value} (可用: {valid})"
            ) from exc
    if isinstance(current, bool):
        normalized = value.strip().lower()
        if normalized in ("1", "true", "yes", "on"):
            return True
        if normalized in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"無法解析布林值: {value}")
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if isinstance(current, tuple):
        parts = [part.strip() for part in value.split(",")]
        if len(parts) != len(current):
            raise ValueError(f"元組需要 {len(current)} 個值: {value}")
        return tuple(cli_coerce(part, element) for part, element in zip(parts, current))
    return value


def apply_cli_overrides(argv=None) -> None:
    """用指令列參數覆寫腳本設定。

    支援兩種格式:
      python scripts/ffxi_event.py --CHEST_SELECT_INDEX=1 --BATTLE_MAX_COUNT=3
      python scripts/ffxi_event.py --chest-select-index 1 --allow-revive false

    名稱不區分大小寫, 底線與連字號等效, 也接受唯一前綴縮寫(如 --chest)。
    """
    if argv is None:
        argv = sys.argv
    config_names = {
        name: value
        for name, value in globals().items()
        if name.isupper()
        and not name.startswith("_")
        and not isinstance(value, (dict, list, set))
        and name != "EYES"
    }
    normalised = {cli_normalise_name(name): name for name in config_names}

    def resolve(raw_name: str) -> str | None:
        key = cli_normalise_name(raw_name)
        if key in normalised:
            return normalised[key]
        if key in CLI_ALIASES and CLI_ALIASES[key] in config_names:
            return CLI_ALIASES[key]
        candidates = [name for name in normalised if name.startswith(key)]
        if len(candidates) == 1:
            return normalised[candidates[0]]
        return None

    consumed = []
    i = 1
    while i < len(argv):
        token = argv[i]
        if not token.startswith("--"):
            i += 1
            continue
        body = token[2:]
        if "=" in body:
            raw_name, raw_value = body.split("=", 1)
        else:
            raw_name = body
            if i + 1 >= len(argv) or argv[i + 1].startswith("--"):
                i += 1
                continue
            raw_value = argv[i + 1]
        name = resolve(raw_name)
        if name is None:
            i += 1
            continue
        current = config_names[name]
        try:
            value = cli_coerce(raw_value, current)
        except ValueError as exc:
            raise SystemExit(f"參數 --{name}={raw_value} 無效: {exc}") from exc
        globals()[name] = value
        state.logger.info("指令列覆寫 %s = %r", name, value)
        consumed.append(i)
        if "=" not in token:
            consumed.append(i + 1)
            i += 2
        else:
            i += 1
    for index in sorted(consumed, reverse=True):
        argv.pop(index)


apply_cli_overrides()


EYES = {
    "notice": NoticeScreen(delay_after_click=8.0),
    "announcement": AnnouncementScreen(delay_after_click=1.0),
    "login": LoginScreen(delay_after_click=5.0),
    "death": DeathView(
        allow_revive=ALLOW_REVIVE,
        character_revive_dry_run=CHARACTER_REVIVE_DRY_RUN,
    ),
    "retry": Retry(),
    "server_error": ServerError(delay_click=2.0),
    "battle_pause": BattlePause(delay_click=1.0),
    "low_willpower": LowWillpowerDialog(),
    "text_with_auto": TextWithAuto(),
    "chest_action": ChestAction(
        select_index=CHEST_SELECT_INDEX, select_retry_cooltime=18
    ),
    "chest_path": ChestPath(mode=CHEST_PATH_MODE),
    "return": Return(need_ret_inn=False, max_battle_num=BATTLE_MAX_COUNT),
    "recovery": Recovery(),
    "auto_mode": BattleToggle(),
    "town": TownView(enable_dungeon=False),
    "inventory": InventoryOrganizer(
        enabled=INVENTORY_ORGANIZE_ENABLED,
        allow_deposit_other=ALLOW_OTHER_DEPOSIT,
        need_organize=False,
    ),
}


def entrypoint(core: "VACore"):
    print(f"Script: {EVENT_NAME}")

    core.setup(lang=LANG, debug=DEBUG, text_mapping=TEXT_MAPPING)

    eyes = [
        EYES["retry"],
        EYES["server_error"],
        EYES["battle_pause"],
        ffxi_recovery,
        EYES["low_willpower"],
        EYES["text_with_auto"],
        EYES["chest_action"],
        ffxi_return,
        EYES["recovery"],
        EYES["auto_mode"],
        ffxi_chest_path,
        EYES["inventory"],
        ffxi_town,
        ffxi_event_menu,
        ffxi_dungeon_select,
        ffxi_world_map,
        EYES["death"],
        EYES["notice"],
        EYES["announcement"],
        EYES["login"],
    ]

    genjitsu_jyanais = [EYES["return"], EYES["town"]]
    for netsuijyou in genjitsu_jyanais:
        core.register_ai(netsuijyou, genjitsu=False)

    for eye in eyes:
        core.register_ai(eye)

    core.run()


def ffxi_recovery() -> bool:
    chest_action: ChestAction = EYES["chest_action"]
    chest_path: ChestPath = EYES["chest_path"]
    now = time.time()
    # 開寶箱後仍優先觸發恢復(既有邏輯)
    from_unlock = chest_action.last_unlock_time > 0
    if not from_unlock:
        # 非開寶箱後: 偵測到「路徑」icon 且冷卻已過才恢復
        if now - RECOVERY_STATE["last_time"] < RECOVERY_COOLDOWN:
            return False
        if not chest_path.check(check_only=True):
            return False
    elif not chest_path.check(check_only=True):
        return False
    state.logger.debug("嘗試恢復狀態...")
    ok = _perform_recovery()
    if from_unlock:
        chest_action.last_unlock_time = 0
    # 冷卻時間在恢復成功完成後才更新(以完成時間為基準)，
    # 避免恢復本身耗時 20 秒導致結束時冷卻剛好過期而立刻再觸發。
    # 恢復失敗時不更新，允許下一輪立即重試。
    if ok:
        RECOVERY_STATE["last_time"] = time.time()
    return True


def _perform_recovery() -> bool:
    left, top, width, height = get_window_rect()
    center_x = int(left + width // 2)
    center_y = int(top + height * 0.95)
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
        return True
    else:
        state.logger.warning("無法進入回復界面...")
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
        town = EYES["town"]
        if not town.check_inn():
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


def _find_ocr_text_center(screen, text, region, *, prefer="first", y_offset=0.0):
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
                cy = (y1b + y2b) // 2 + y1 + int(y_offset * h)
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

    EYES["chest_action"].reset_select_index()

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
        WORLD_MAP_STATE["warned"] = False
        return False

    state.logger.info("偵測到世界地圖")

    # 進入世界地圖先將游標置中, 避免先前操作殘留的游標位置影響地圖捲動
    _left, _top, _w, _h = get_window_rect()
    move_cursor_to((_left + _w // 2, _top + _h // 2))
    time.sleep(WORLD_MAP_SETTLE_DELAY)

    while WORLD_MAP_STATE["scrolls"] < WORLD_MAP_MAX_SCROLLS:
        _screen = get_window_screen()
        _point = _find_ocr_text_center(
            _screen, WORLD_MAP_TARGET, WORLD_MAP_TARGET_REGION, y_offset=-0.04
        )
        if _point:
            _point = calculate_click_point(_point, (0, 0))
            state.logger.info("前往王都盧庫納里亞")
            click_at(_point)
            time.sleep(5)
            WORLD_MAP_STATE["scrolls"] = 0
            WORLD_MAP_STATE["warned"] = False
            return True

        state.logger.info("向下捲動世界地圖...")
        _down_x = _left + _w // 2
        _down_y = _top + _h - WORLD_MAP_SCROLL_BORDER_OFFSET
        _safe_x = _left + _w // 2
        _safe_y = _top + _h // 2

        move_cursor_to((_down_x, _down_y))
        time.sleep(WORLD_MAP_SCROLL_DOWN_DURATION)
        move_cursor_to((_safe_x, _safe_y))
        time.sleep(WORLD_MAP_SETTLE_DELAY)
        WORLD_MAP_STATE["scrolls"] += 1

    # 已達最大捲動次數: 仍保留一次目標偵測(手動移動地圖後目標可能已可見),
    # 且警告只記錄一次直到下次重置
    _screen = get_window_screen()
    _point = _find_ocr_text_center(
        _screen, WORLD_MAP_TARGET, WORLD_MAP_TARGET_REGION, y_offset=-0.04
    )
    if _point:
        _point = calculate_click_point(_point, (0, 0))
        state.logger.info("前往王都盧庫納里亞")
        click_at(_point)
        time.sleep(5)
        WORLD_MAP_STATE["scrolls"] = 0
        WORLD_MAP_STATE["warned"] = False
        return True

    if not WORLD_MAP_STATE["warned"]:
        WORLD_MAP_STATE["warned"] = True
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


def ffxi_return() -> bool:
    global FFXI_IN_DUNGEON

    if EYES["return"].check():
        FFXI_IN_DUNGEON = False
        state.logger.info("已設置離開地下城標記")
        return True
    return False


if __name__ == "__main__":
    entrypoint(VACore())
