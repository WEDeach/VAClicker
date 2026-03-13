import time

import vgamepad

from src.ai.battle.auto_mode import AutoMode
from src.ai.btn._return import Return
from src.ai.btn.chest_action import ChestAction
from src.ai.btn.path.chest import ChestPath
from src.ai.btn.recovery import Recovery
from src.ai.btn.retry import Retry
from src.ai.dialog.text_with_auto import TextWithAuto
from src.ai.view.dungeon import DungeonView
from src.ai.view.town import TownView
from src.core import VACore
from src.utils.clicker import calculate_click_point, click_at, click_by_gamepad
from src.utils.image import load_template, match_template
from src.utils.shared import state
from src.utils.window import dump4log, get_window_rect, get_window_screen

LANG = "chinese_cht"
DEBUG = True
MAPPING = {"chest.action.open": "打開\n\\w*?都不做"}
EYES = {
    "retry": Retry(),
    "text_with_auto": TextWithAuto(),
    "chest_action": ChestAction(select_index=1),
    "chest_path": ChestPath(),
    "return": Return(need_ret_inn=True),
    "recovery": Recovery(),
    "auto_mode": AutoMode(),
    "town": TownView(),
    "dungeon": DungeonView(),
}
CORE = None
STATE = {}


def entrypoint(core: "VACore"):
    global CORE, STATE
    print("Script: 異界的冒險者")

    core.setup(lang=LANG, debug=DEBUG, text_mapping=MAPPING)
    CORE = core

    eyes = [
        EYES["retry"],
        recovery,
        #
        EYES["text_with_auto"],
        EYES["chest_action"],
        EYES["return"],
        EYES["recovery"],
        EYES["auto_mode"],
        EYES["town"],
        warppedDungeonView,
        #
        EYES["chest_path"],
    ]

    EYES["dungeon"].core = core

    for eye in eyes:
        core.register_ai(eye)

    core.run()


def recovery() -> bool:
    chest_action: ChestAction = EYES["chest_action"]
    chest_path: ChestPath = EYES["chest_path"]
    if chest_action.last_unlock_time > 0:
        if chest_path.check(check_only=True):
            left, top, width, height = get_window_rect()
            center_x = int(left + width // 2)
            center_y = int(top + height * 0.95)
            state.logger.debug(
                "嘗試恢復狀態...",
            )
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
                    ocr_check=[("意志力", 0)],
                    region=(0.33, 0.41, 0.5, 0.55),
                )
                if _match:
                    return True
                return False

            def _check_for_heal_btn(click=False):
                _screen = get_window_screen()
                _match = match_template(
                    _screen,
                    None,
                    1,
                    False,
                    None,
                    ocr_check=[("回復", 0)],
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
                    # 按鈕不是active狀態時, 需要點兩次
                    click_at((_x, _y))
                    time.sleep(5)
                _check_for_heal_btn(True)
                click_by_gamepad(vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_B)
                time.sleep(2)
                click_by_gamepad(vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_B)
                chest_action.last_unlock_time = 0
            else:
                dump4log(get_window_screen(), "回復失敗")
                state.logger.warning("無法進入回復界面...")
                click_by_gamepad(vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_B)
                time.sleep(3)
            return True
    return False


def warppedDungeonView() -> bool:
    dungeon: DungeonView = EYES["dungeon"]
    _return: Return = EYES["return"]
    if dungeon.check():
        # 回到郊外後, 可能需要回旅館休息
        if not _return.need_ret_inn:
            _screen = get_window_screen()
            _temp, _mask = load_template("郊外_魚人.png", grayscale=False)
            _match = match_template(
                _screen,
                _temp,
                0.8,
                False,
                _mask,
                region=(0.3, 0.7, 0, 1),
            )
            if _match:
                loc, score = _match
                point = calculate_click_point(loc, (_temp.shape[1], _temp.shape[0]))
                click_at(point)
                time.sleep(10)

                _temp, _mask = load_template("郊外_魚人_入口.png", grayscale=False)
                deadline = time.time() + 10.0
                while time.time() < deadline:
                    _screen = get_window_screen()
                    _match = match_template(
                        _screen,
                        _temp,
                        0.8,
                        False,
                        _mask,
                        region=(0.3, 0.7, 0, 1),
                    )
                    if _match:
                        loc, score = _match
                        point = calculate_click_point(
                            loc, (_temp.shape[1], _temp.shape[0])
                        )
                        click_at(point)
                        time.sleep(5)

                        def _check_alert():
                            _screen = get_window_screen()
                            _match = match_template(
                                _screen,
                                None,
                                0.8,
                                False,
                                None,
                                ocr_check=[("前往", 0)],  # 前往地下城
                                region=(0.52, 0.60, 0.56, 0.61),
                            )
                            if _match:
                                _loc, _ = _match
                                state.logger.warning("當前意志力較低, 仍選擇前往地下城")
                                _point = calculate_click_point(_loc, (0, 0))
                                click_at(_point)
                                time.sleep(5)

                        _check_alert()
                        return True

                    state.logger.debug("等待魚人入場入口點...")
                    time.sleep(0.5)
            else:
                state.logger.debug("已在郊外, 但無法找到魚人入場")
        return True
    return False


if __name__ == "__main__":
    entrypoint(VACore())
