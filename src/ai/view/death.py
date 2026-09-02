import time
from collections import deque

import cv2
import numpy as np

from ... import AI
from ...ocr import parse_ocr_boxes
from ...utils.clicker import calculate_click_point, click_at
from ...utils.shared import state
from ...utils.text_map import get_text_mapping
from ...utils.window import get_window_rect, get_window_screen

PLAYER_DEATH_REGION = (0.30, 0.70, 0.40, 0.65)
REVIVE_CHARGE_REGION = (0.35, 0.65, 0.00, 0.15)
REVIVE_CHARGE_HSV_LOWER = (10, 80, 120)
REVIVE_CHARGE_HSV_UPPER = (45, 255, 255)
REVIVE_CHARGE_MIN_WIDTH = 0.02
REVIVE_CHARGE_MAX_WIDTH = 0.04
REVIVE_CHARGE_MIN_HEIGHT = 0.03
REVIVE_CHARGE_MAX_HEIGHT = 0.09
REVIVE_CHARGE_MIN_AREA = 0.0004
REVIVE_CHARGE_MAX_AREA = 0.0022
REVIVE_CHARGE_PRESENT_MIN_PIXELS = 50
PLAYER_DEATH_TEXT_KEYS = (
    "view.death.player.revive",
    "view.death.player.accept",
)
CHARACTER_DEATH_WAITING = "character_death_waiting"
CHARACTER_DEATH_REVIVING = "character_death_reviving"
CHARACTER_REVIVE_SESSION_OVERLAY = "overlay"
CHARACTER_REVIVE_SESSION_NORMAL = "normal"
CHARACTER_EXIT_REGION = (0.90, 1.00, 0.82, 1.00)
CHARACTER_RING_REGION = (0.30, 0.70, 0.20, 0.80)
# 環中心空心驗證: 中心小區域的藍色像素必須極少(環形而非實心填充)。
# 死亡畫面的手部/UI 元素可能位於中心, 因此同時允許「中心藍色相對環區很少」
# 的比值條件, 以容忍小型中心元素同時排除整片藍天的城鎮畫面。
CHARACTER_RING_CENTER_REGION = (0.46, 0.54, 0.45, 0.55)
CHARACTER_RING_CENTER_MAX_PIXELS = 50
CHARACTER_RING_CENTER_MAX_RATIO = 0.35
CHARACTER_TARGET_REGION = (0.40, 0.60, 0.30, 0.70)
CHARACTER_DEATH_OVERLAY_REGION = (0.20, 0.80, 0.20, 0.80)
CHARACTER_DEATH_OVERLAY_RED_MARGIN = 15
CHARACTER_DEATH_OVERLAY_MIN_COVERAGE = 0.90
CHARACTER_EXIT_HSV_LOWER = (0, 0, 180)
CHARACTER_EXIT_HSV_UPPER = (180, 70, 255)
CHARACTER_EXIT_FALLBACK_HSV_LOWER = (0, 0, 100)
CHARACTER_EXIT_FALLBACK_HSV_UPPER = (180, 180, 255)
# 紅色 overlay 死亡畫面的退出按鈕會被染紅(飽和度升高、亮度降低),
# 白框偵測失效; 改用「低飽和亮色 + 近正方形」的按鈕形狀偵測。
# 正方形條件排除城鎮按鍵提示(扁長)與其他誤判。
# 組件必須「有色彩」(mean sat >= 下限): 排除灰色按鈕(如寶箱介面的
# 暗灰方形按鈕 sat~4, 會被誤判為退出按鈕而誤進復活追蹤)。
CHARACTER_EXIT_REDOVERLAY_HSV_LOWER = (0, 0, 100)
CHARACTER_EXIT_REDOVERLAY_HSV_UPPER = (180, 100, 255)
CHARACTER_EXIT_REDOVERLAY_MIN_ASPECT = 0.7
CHARACTER_EXIT_REDOVERLAY_MAX_ASPECT = 1.4
CHARACTER_EXIT_REDOVERLAY_MIN_SATURATION = 25.0
CHARACTER_RING_HSV_LOWER = (90, 80, 80)
CHARACTER_RING_HSV_UPPER = (130, 255, 255)
CHARACTER_RING_FALLBACK_MAX_SATURATION = 140
CHARACTER_RING_FALLBACK_MIN_VALUE = 100
CHARACTER_RING_FALLBACK_MIN_RADIUS = 0.08
CHARACTER_RING_FALLBACK_MAX_RADIUS = 0.30
CHARACTER_RING_FALLBACK_MIN_ANGULAR_COVERAGE = 0.20
CHARACTER_RING_FALLBACK_SAMPLE_COUNT = 180
# 紅色 overlay 死亡畫面的環偵測: 圓環被染紅, 用紅色圓周覆蓋偵測。
# 覆蓋閾值 0.35: 0826 死亡畫面紅色覆蓋 0.32-1.00(極少數極弱 overlay 幀
# 略低於閾值, 輪詢會在下個檢查恢復); 暫停/其他畫面的紅色弧形最高 ~0.37,
# 但那些畫面無退出按鈕, 不會被 character_death_state 判定為死亡。
CHARACTER_RING_RED_HUE_MAX = 15
CHARACTER_RING_RED_HUE_MIN = 165
CHARACTER_RING_RED_MIN_SATURATION = 40
CHARACTER_RING_RED_MIN_VALUE = 80
CHARACTER_RING_RED_MIN_ANGULAR_COVERAGE = 0.35
CHARACTER_TARGET_HSV_LOWER = (5, 50, 140)
CHARACTER_TARGET_HSV_UPPER = (45, 255, 255)
CHARACTER_WEAK_TARGET_HSV_LOWER = (0, 15, 55)
CHARACTER_WEAK_TARGET_HSV_UPPER = (55, 255, 255)
CHARACTER_NEUTRAL_TARGET_HSV_LOWER = (0, 0, 100)
CHARACTER_NEUTRAL_TARGET_HSV_UPPER = (180, 220, 255)
CHARACTER_NEUTRAL_TARGET_MIN_CONFIDENCE = 0.55
CHARACTER_NEUTRAL_TARGET_MIN_FILL_RATIO = 0.25
CHARACTER_NEUTRAL_TARGET_CONFIRMATION_TOLERANCE = 5.0
CHARACTER_NEUTRAL_TARGET_STRONG_CONFIDENCE = 0.9
CHARACTER_NEUTRAL_TARGET_PENDING_MIN_CONFIDENCE = 0.7
CHARACTER_EXIT_MIN_WIDTH = 0.02
CHARACTER_EXIT_MAX_WIDTH = 0.05
CHARACTER_EXIT_MIN_HEIGHT = 0.03
CHARACTER_EXIT_MAX_HEIGHT = 0.07
CHARACTER_RING_MIN_PIXELS = 500
# 無紅色 overlay 的死亡等待畫面: 中心黃色像素上限。
# 0820 死亡畫面 137-241px; 091517 死亡畫面變體 1754px(黃色 UI 裝飾較多);
# 復活圓環畫面的黃色 disk + 外圈裝飾為 15697px 以上, 不會誤判為死亡等待。
CHARACTER_DEATH_NO_OVERLAY_MAX_TARGET_PIXELS = 5000
CHARACTER_TARGET_MIN_PIXELS = 1000
# 復活圓環進行中(REVIVING)的黃色目標像素強門檻: 復活畫面的黃色 disk +
# 外圈裝飾達 15000px 以上; 死亡等待畫面即使有黃色 UI 裝飾(091517 為
# 1754px)也遠低於此。disk 形狀偵測是主要區分(0823 復活 450/450 幀
# disk 都存在), 此門檻是 disk 偶爾偵測失敗時的雙保險。
CHARACTER_REVIVING_STRONG_TARGET_PIXELS = 10000
CHARACTER_REVIVE_TARGET_RADIUS = 0.06
CHARACTER_REVIVE_TARGET_TOLERANCE = 0.015
CHARACTER_REVIVE_MAX_RADIUS = 0.34
CHARACTER_TARGET_MIN_RADIUS = 4.0
CHARACTER_TARGET_MAX_RADIUS = 0.12
CHARACTER_REVIVE_DIRECT_TARGET_MIN_RADIUS = 0.04
CHARACTER_REVIVE_DIRECT_TARGET_MIN_CONFIDENCE = 0.60
CHARACTER_TARGET_MIN_COMPONENT_PIXELS = 20
CHARACTER_TARGET_MIN_ANGULAR_COVERAGE = 0.45
CHARACTER_TARGET_MIN_CONFIDENCE = 0.60
CHARACTER_TARGET_CENTER_OFFSET_RATIO = 0.35
CHARACTER_TARGET_ANGULAR_BINS = 36
CHARACTER_WEAK_TARGET_MIN_RADIUS = 2.0
CHARACTER_WEAK_TARGET_MIN_COMPONENT_PIXELS = 3
CHARACTER_WEAK_TARGET_MIN_ANGULAR_COVERAGE = 0.25
CHARACTER_WEAK_TARGET_MIN_CONFIDENCE = 0.50
CHARACTER_WEAK_TARGET_CENTER_OFFSET_RATIO = 0.25
CHARACTER_WEAK_TARGET_ANGULAR_BINS = 16
CHARACTER_WEAK_TARGET_CONFIRMATION_SAMPLES = 2
CHARACTER_WEAK_TARGET_CONFIRMATION_TOLERANCE = 2.0
CHARACTER_TARGET_HISTORY_SIZE = 5
CHARACTER_TARGET_HISTORY_BRIDGE = 0.6
CHARACTER_TARGET_CACHE_TTL = 1.0
CHARACTER_TARGET_TOLERANCE_RATIO = 0.25
CHARACTER_TARGET_MIN_TOLERANCE = 3.0
CHARACTER_REVIVE_CLICK_RADIUS_MARGIN = 3.0
CHARACTER_REVIVE_RING_MIN_RADIUS = 6.0
# 復活圓環場景(fallback)的環測量下限餘裕: 死亡等待畫面的環測量常落在
# 中心偽影(r6 = 測量下限), 測到 r6+餘裕 以內一律視為非復活圓環。
CHARACTER_REVIVE_SCENE_MIN_RING_MARGIN = 14.0
CHARACTER_REVIVE_SCORE_THRESHOLD = 0.35
CHARACTER_REVIVE_FALLBACK_MAX_SATURATION = 140
CHARACTER_REVIVE_FALLBACK_MIN_VALUE = 100
CHARACTER_REVIVE_FALLBACK_SCORE_THRESHOLD = 0.35
CHARACTER_REVIVE_HISTORY_SIZE = 12
CHARACTER_REVIVE_MIN_MEASURED_FRAMES = 3
CHARACTER_REVIVE_MIN_MEASURED_CONFIDENCE = 0.5
CHARACTER_REVIVE_LOG_INTERVAL = 0.1
CHARACTER_REVIVE_DETECTION_GRACE = 0.15
CHARACTER_REVIVE_MAX_PREDICTION_GAP = 0.5
CHARACTER_REVIVE_MAX_SHRINK_SPEED = 6000.0
CHARACTER_REVIVE_MIN_SHRINK_SPEED = 25.0
# 圓環到達本輪最小半徑的容差
CHARACTER_REVIVE_MIN_RADIUS_TOLERANCE = 6.0
# 週期最小半徑需連續停留的幀數 (避免收縮途中短暫停留的誤判)
CHARACTER_REVIVE_CYCLE_MIN_STABLE_FRAMES = 2
# 週期最小半徑的停留判定容差 (嚴格 2px: 收縮中繼半徑相差 5px 不算停留)
CHARACTER_REVIVE_CYCLE_MIN_STABLE_TOLERANCE = 2.0
# 絕對最小半徑: 環收縮且本週期見過明顯外圈時, 依「到達中心的剩餘時間
# (eta)」點擊, 用於環在最小處幾乎不停留的極端畫面。
# 系統延遲(點擊送出→遊戲生效)約 50-60ms: 點擊在 eta 落於下方範圍時送出,
# 讓點擊生效時環正好到達中心。太晚(eta 過小)會讓環已擴張 → 失敗扣血。
CHARACTER_REVIVE_ABSOLUTE_MIN_ETA_MIN = 0.045
CHARACTER_REVIVE_ABSOLUTE_MIN_ETA_MAX = 0.065
# 絕對最小判定所需的外圈半徑門檻 (排除靜態 r80 盤緣/死亡等待靜態環)
CHARACTER_REVIVE_ABSOLUTE_MIN_OUTER_RADIUS = 150.0
# 昏暗黃色目標圓盤的 HSV 偵測範圍(影片畫面較暗, 放寬亮度)
CHARACTER_TARGET_DISK_HSV_LOWER = (5, 30, 80)
CHARACTER_TARGET_DISK_HSV_UPPER = (45, 255, 255)
CHARACTER_TARGET_DISK_MAX_RADIUS = 0.22
CHARACTER_TARGET_DISK_MIN_RADIUS = 20.0
CHARACTER_TARGET_DISK_MIN_AREA = 800
CHARACTER_TARGET_DISK_CENTER_SEARCH = 0.30
CHARACTER_REVIVE_CLICK_COOLDOWN = 1.0
CHARACTER_REVIVE_RETRY_DELAY = 0.75
CHARACTER_REVIVE_POST_CLICK_GRACE = 3.0
CHARACTER_REVIVE_TIMEOUT_COOLDOWN = 2.0
CHARACTER_REVIVE_SESSION_DEADLINE = 30.0
CHARACTER_REVIVE_SAMPLE_COUNT = 360
CHARACTER_DEATH_POLL_INTERVAL = 0.25
# 預測點擊: 收縮期內以最近幾幀擬合速度, 預測 ring 到達最小半徑的時刻
CHARACTER_REVIVE_PREDICT_FRAMES = 6
# 點擊窗口: eta(預測剩餘秒數) 落於 [下界, 上界] 才觸發。
# 上界 = 點擊送出→遊戲生效延遲(~55ms) + 少量餘裕: 超過則生效時環未到中心, 太早。
# 下界: eta 低於此值表示這輪已太晚(生效時環已過中心開始擴張) → 不點,
# 跳過本輪等下一輪收縮(週期僅 ~0.3s, 很快有下一次機會)。
CHARACTER_REVIVE_PREDICT_LEAD = 0.075
CHARACTER_REVIVE_PREDICT_LEAD_MIN = 0.045
CHARACTER_REVIVE_PREDICT_MIN_SAMPLES = 3
# 預測模型的有效觀察窗口: 15 秒 / 至多兩個收縮週期
CHARACTER_REVIVE_MODEL_WINDOW = 15.0
# 預測前 ring 必須從本週期外圈實質收縮的比例 (排除最小半徑附近的抖動)
CHARACTER_REVIVE_PREDICT_MIN_CONTRACTION = 0.30
# 預測擬合速度必須快於此值 (px/s), 排除停留/抖動的假收縮
CHARACTER_REVIVE_PREDICT_MIN_SPEED = 200.0
# 靜態白環(中央盤緣/UI 圓圈)偵測: 存在時 ring 測量改搜尋其外側的大環
CHARACTER_REVIVE_STATIC_RING_MIN_RADIUS = 60
CHARACTER_REVIVE_STATIC_RING_MAX_RADIUS = 120
CHARACTER_REVIVE_STATIC_RING_MIN_SCORE = 0.8
CHARACTER_REVIVE_STATIC_RING_EXCLUDE_MIN = 120
CHARACTER_REVIVE_STATIC_RING_CONFIRM_FRAMES = 8
# 圓環收縮後停止偵測: 連續 N 幀半徑變化小於容差即視為已到達中心。
# N 需夠大以排除收縮中途的短暫停頓(遊戲 ring 在真正中心會停較久)。
CHARACTER_REVIVE_SETTLE_FRAMES = 8
CHARACTER_REVIVE_SETTLE_TOLERANCE = 4.0
# 觀測階段的密集採樣預算(秒): 允許單次 check 內連續取幀直到點擊/達標
CHARACTER_REVIVE_OBSERVE_BUDGET = 10.0
# 觀測間隔: 收縮週期僅 ~0.3s, 觀測必須夠密才能抓準點擊窗口
CHARACTER_REVIVE_OBSERVE_STEP = 0.005


class DeathView(AI):
    def __init__(
        self,
        *,
        allow_revive: bool = False,
        delay_revive: float = 10.0,
        min_revive_charges: int = 1,
        player_death_region=PLAYER_DEATH_REGION,
        revive_charge_region=REVIVE_CHARGE_REGION,
        allow_character_revive: bool = True,
        allow_character_exit: bool = False,
        delay_character_action: float = 0.5,
        character_revive_dry_run: bool = True,
        character_revive_session_deadline: float = CHARACTER_REVIVE_SESSION_DEADLINE,
        character_death_poll_interval: float = CHARACTER_DEATH_POLL_INTERVAL,
        wait_if_last_revive_charge: bool = True,
        revive_charge_wait_seconds: float = 1 * 3600 + 59 * 60,
    ):
        super().__init__()
        self.allow_revive = allow_revive
        self.delay_revive = delay_revive
        self.min_revive_charges = min_revive_charges
        self.player_death_region = player_death_region
        self.revive_charge_region = revive_charge_region
        self.allow_character_revive = allow_character_revive
        self.allow_character_exit = allow_character_exit
        self.delay_character_action = delay_character_action
        self.character_revive_dry_run = character_revive_dry_run
        self.character_revive_session_deadline = character_revive_session_deadline
        self.character_death_poll_interval = character_death_poll_interval
        # 復活之火只剩最後一個時, 先等待恢復週期(每 2 小時恢復一個、
        # 非即時)再點擊復活, 避免用掉最後一個後無火可用。
        self.wait_if_last_revive_charge = wait_if_last_revive_charge
        self.revive_charge_wait_seconds = revive_charge_wait_seconds
        self._revive_charge_wait_until = None
        self._revive_charge_wait_last_shown = None
        # 本提示 episode 是否已等待過(截止後鎖存, 提示消失時清除)
        self._revive_charge_wait_done = False
        self._revive_charge_wait_missing_frames = 0
        self._last_character_death_state = None
        self._character_revive_history = deque(maxlen=CHARACTER_REVIVE_HISTORY_SIZE)
        self._character_revive_target_history = deque(
            maxlen=CHARACTER_TARGET_HISTORY_SIZE
        )
        self._character_revive_stable_target = None
        self._character_revive_stable_target_at = None
        self._character_revive_session_started = None
        self._last_character_death_seen = None
        self._character_revive_activated = False
        self._character_revive_session_mode = None
        self._character_revive_target_clicked = False
        self._character_revive_last_click = None
        self._character_revive_deadline_logged = False
        self._character_revive_session_expired = False
        self._character_revive_cycle_shrinking = False
        self._character_revive_session_min_radius = None
        self._character_revive_min_dwell = 0
        self._character_revive_last_radius = None
        self._character_revive_stable_radius_frames = 0
        self._character_revive_shrink_history = []
        self._character_revive_cycle_started_at = None
        self._character_revive_cycle_max_radius = None
        self._character_revive_cycle_min_radius = None
        self._character_revive_static_ring_radius = None
        self._character_revive_static_ring_pending = None
        self._character_revive_static_ring_frames = 0
        self._character_revive_cycle_count = 0
        self._character_revive_model_ready = False
        # 本輪已錯過預測點擊窗口(eta < 下界): 抑制同輪的 ring_at_min/
        # ring_settled 補點(太晚送出), 等下一輪收縮再點。
        self._character_revive_cycle_missed_predict = False
        self._character_revive_model_epoch = None
        self._character_revive_model_expired = False
        self._character_revive_retry_at = None
        self._character_revive_post_click_until = None
        self._character_revive_timeout_until = None

    def check(self) -> bool:
        now = time.monotonic()
        # 復活之火等待/鎖存期間, 玩家復活提示必須仍在畫面上; 消失即
        # 結束 episode(清除等待狀態)。放在所有提前 return(含冷卻分支)
        # 之前, 確保畫面轉換(角色死亡路徑/冷卻)期間也能偵測提示消失。
        # 連續 2 次確認, 避免單次 OCR 失敗誤清等待(誤清會導致重新開始
        # 兩小時等待)。
        if self._revive_charge_wait_until is not None or self._revive_charge_wait_done:
            try:
                screen_now = get_window_screen()
                present = self._find_player_death_revive_box(screen_now) is not None
            except Exception:
                # 截圖/OCR 失敗: 不當作確認 miss(避免誤取消兩小時等待),
                # 也不讓例外中斷 check。
                present = True
            if not present:
                self._revive_charge_wait_missing_frames += 1
                if self._revive_charge_wait_missing_frames >= 2:
                    self._clear_revive_charge_wait()
                    state.logger.info("玩家復活提示已消失，取消復活之火等待")
            else:
                self._revive_charge_wait_missing_frames = 0
        if self._character_revive_retry_at is not None:
            if now < self._character_revive_retry_at:
                self._poll_character_death_screen()
                return True
            self._character_revive_retry_at = None
        if self._character_revive_timeout_until is not None:
            if now < self._character_revive_timeout_until:
                return False
            self._character_revive_timeout_until = None
        if self._character_revive_post_click_until is not None:
            if now < self._character_revive_post_click_until:
                self._poll_character_death_screen()
                return True
            self._character_revive_post_click_until = None
        try:
            screen = get_window_screen()
        except Exception:
            state.logger.warning("取得畫面失敗")
            return False
        character_state = self.character_death_state(screen)
        exit_box = self._character_exit_box(screen)
        session_present = self._character_revive_activated and (
            self._has_active_character_revive_session(screen, now, exit_box)
        )
        if self._character_revive_session_expired:
            self._last_character_death_state = None
            self._reset_character_revive_observations()
        if character_state is not None:
            self._last_character_death_seen = now
            self._log_character_death_state(character_state)
            if self.allow_character_revive:
                if (
                    character_state == CHARACTER_DEATH_WAITING
                    and self._character_revive_activated
                    and self._character_revive_target_clicked
                ):
                    state.logger.warning(
                        "復活點擊後仍回到等待復活，%.2f 秒後重新啟動",
                        CHARACTER_REVIVE_RETRY_DELAY,
                    )
                    self._reset_character_revive_observations()
                    self._character_revive_retry_at = now + CHARACTER_REVIVE_RETRY_DELAY
                    return True
                if (
                    character_state == CHARACTER_DEATH_WAITING
                    and not self._character_revive_activated
                ):
                    self._character_revive_session_mode = (
                        CHARACTER_REVIVE_SESSION_OVERLAY
                        if self._has_character_death_overlay(screen)
                        else CHARACTER_REVIVE_SESSION_NORMAL
                    )
                    self._activate_character_revive()
                    return True
                if not self._character_revive_activated:
                    self._character_revive_session_mode = (
                        CHARACTER_REVIVE_SESSION_NORMAL
                    )
                    self._character_revive_activated = True
                    state.logger.info("已位於復活畫面（無需入口點擊），直接開始追蹤")
                observation = self._observe_character_revive(screen, now)
                self._poll_character_death_screen(observation)
                return not self._character_revive_session_expired
            if character_state == CHARACTER_DEATH_WAITING and self.allow_character_exit:
                if exit_box is not None:
                    exit_x, exit_y, exit_width, exit_height = exit_box
                    left, top, _, _ = get_window_rect()
                    click_at(
                        (
                            left + exit_x + exit_width // 2,
                            top + exit_y + exit_height // 2,
                        )
                    )
                    state.logger.info("已點擊角色死亡畫面關閉按鈕")
                    time.sleep(self.delay_character_action)
            return True

        if (
            self.allow_character_revive
            and not self._character_revive_activated
            and self._has_character_revive_scene(screen, exit_box)
        ):
            self._character_revive_session_mode = CHARACTER_REVIVE_SESSION_NORMAL
            self._character_revive_activated = True
            state.logger.info("已位於復活圓環畫面，直接開始追蹤（略過中心點擊）")
            observation = self._observe_character_revive(screen, now)
            self._poll_character_death_screen(observation)
            return not self._character_revive_session_expired

        if (
            self.allow_character_revive
            and self._character_revive_activated
            and session_present
        ):
            observation = self._observe_character_revive(screen, now)
            self._poll_character_death_screen(observation)
            return not self._character_revive_session_expired
        self._last_character_death_state = None
        if self._character_revive_target_clicked:
            state.logger.info("角色復活畫面已消失，最後一次目標 click 可能成功")
        self._reset_character_revive_observations()
        try:
            revive_box = self._find_player_death_revive_box(screen)
        except Exception:
            # OCR 失敗: 視為未偵測到復活提示, 不中斷 check
            state.logger.warning("玩家復活提示 OCR 失敗")
            revive_box = None
        if revive_box is not None and self.allow_revive:
            charges_present, charges = self.revive_charge_status(screen)
            state.logger.debug(
                "玩家死亡, 復活之火: %d, 圖示存在=%s",
                charges,
                charges_present,
            )
            if not charges_present:
                click_at(calculate_click_point(revive_box[:2], revive_box[2:]))
                state.logger.info("未偵測到復活之火, 執行免費復活")
                time.sleep(self.delay_revive)
            elif charges < self.min_revive_charges:
                state.logger.warning(
                    ("復活之火不足 (%d < %d)，不執行再起"),
                    charges,
                    self.min_revive_charges,
                )
            else:
                if (
                    self.wait_if_last_revive_charge
                    and charges == 1
                    and not self._revive_charge_wait_done
                ):
                    if self._revive_charge_wait_until is None:
                        self._revive_charge_wait_until = (
                            time.monotonic() + self.revive_charge_wait_seconds
                        )
                        self._revive_charge_wait_last_shown = None
                        state.logger.warning(
                            "復活之火只剩最後一個，等待 %.2f 小時 (%.0f 秒) 後再復活",
                            self.revive_charge_wait_seconds / 3600,
                            self.revive_charge_wait_seconds,
                        )
                        self._show_revive_charge_countdown()
                        return True
                    remaining = self._revive_charge_wait_until - time.monotonic()
                    if remaining > 0:
                        self._show_revive_charge_countdown(remaining)
                        return True
                    # 截止: 結束倒數, 鎖存「本提示 episode 已等待過」,
                    # 避免復活後提示仍在時立刻又進入兩小時等待。
                    self._revive_charge_wait_until = None
                    self._revive_charge_wait_last_shown = None
                    self._revive_charge_wait_done = True
                    print()
                    state.logger.info("復活之火恢復等待結束，執行復活")
                click_at(calculate_click_point(revive_box[:2], revive_box[2:]))
                state.logger.info("已使用復活之火，等待 %.1f 秒", self.delay_revive)
                time.sleep(self.delay_revive)
            return True
        # 玩家復活提示不在畫面上: 等待/鎖存狀態由 check 開頭的連續
        # 兩次確認統一清除(避免單次 OCR 失敗誤清), 這裡不需重複處理。
        return False

    def _clear_revive_charge_wait(self) -> None:
        """清除復活之火等待狀態(提示 episode 結束時呼叫)。"""
        if self._revive_charge_wait_until is not None:
            print()
        self._revive_charge_wait_until = None
        self._revive_charge_wait_last_shown = None
        self._revive_charge_wait_done = False
        self._revive_charge_wait_missing_frames = 0

    def _show_revive_charge_countdown(self, remaining: float = None) -> None:
        """在 CLI 以 \r 覆寫同一行即時顯示等待剩餘時間。

        只用 print(不走 logger), 因此不會寫入 log 檔案;
        每秒更新一次, 剩餘秒數不變時跳過。顯示用 ceil:
        剛開始的 7140 秒顯示 01:59:00(而非 floor 的 01:58:59)。
        """
        if remaining is None:
            remaining = self._revive_charge_wait_until - time.monotonic()
        shown = max(0, int(remaining))
        if remaining > shown:
            shown += 1
        if shown == self._revive_charge_wait_last_shown:
            return
        self._revive_charge_wait_last_shown = shown
        hours, rem = divmod(shown, 3600)
        minutes, seconds = divmod(rem, 60)
        print(
            f"\r等待復活之火恢復剩餘 {hours:02d}:{minutes:02d}:{seconds:02d}",
            end="",
            flush=True,
        )
        if shown <= 0:
            print()

    def _log_character_death_state(self, character_state) -> None:
        if character_state != self._last_character_death_state:
            if character_state == CHARACTER_DEATH_WAITING:
                state.logger.debug("角色死亡，等待復活")
            else:
                state.logger.debug("角色死亡，復活中")
            self._last_character_death_state = character_state

    def _reset_character_revive_observations(self) -> None:
        self._character_revive_history.clear()
        self._character_revive_target_history.clear()
        self._character_revive_stable_target = None
        self._character_revive_stable_target_at = None
        self._character_revive_session_started = None
        self._last_character_death_seen = None
        self._character_revive_activated = False
        self._character_revive_session_mode = None
        self._character_revive_target_clicked = False
        self._character_revive_last_click = None
        self._character_revive_deadline_logged = False
        self._character_revive_session_expired = False
        self._character_revive_cycle_shrinking = False
        self._character_revive_session_min_radius = None
        self._character_revive_min_dwell = 0
        self._character_revive_last_radius = None
        self._character_revive_stable_radius_frames = 0
        self._character_revive_shrink_history = []
        self._character_revive_cycle_started_at = None
        self._character_revive_cycle_max_radius = None
        self._character_revive_cycle_min_radius = None
        self._character_revive_static_ring_radius = None
        self._character_revive_static_ring_pending = None
        self._character_revive_static_ring_frames = 0
        self._character_revive_cycle_count = 0
        self._character_revive_model_ready = False
        # 本輪已錯過預測點擊窗口(eta < 下界): 抑制同輪的 ring_at_min/
        # ring_settled 補點(太晚送出), 等下一輪收縮再點。
        self._character_revive_cycle_missed_predict = False
        self._character_revive_model_epoch = None
        self._character_revive_model_expired = False
        self._character_revive_retry_at = None
        self._character_revive_post_click_until = None
        self._character_revive_timeout_until = None

    def _reset_character_revive_history(self) -> None:
        """Clear ring/target observations for a new shrink cycle while keeping
        the session (deadline, activation) intact."""
        self._character_revive_history.clear()
        self._character_revive_target_history.clear()
        self._character_revive_stable_target = None
        self._character_revive_stable_target_at = None
        # 注意: cycle_shrinking 不在此清除 — 它是「上一週期已完整收縮」的
        # 證據, 必須跨越點擊後的 history 重置, 直到下一次外圈重置才歸零
        self._character_revive_min_dwell = 0
        self._character_revive_last_radius = None
        self._character_revive_stable_radius_frames = 0
        self._character_revive_shrink_history = []
        self._character_revive_cycle_started_at = None
        self._character_revive_cycle_max_radius = None
        # 點擊後重置: 清除「錯過預測窗口」標記(下一輪重新計算)
        self._character_revive_cycle_missed_predict = False
        # 保留 cycle_min_radius: 它是「ring 真正到達過的最小半徑」,
        # 必須跨週期/跨點擊保留, 否則收縮中繼半徑會被誤當成最小值
        # 保留 last_click: 冷卻時間跨週期持續, 預測點擊後不會在同一
        # 收縮週期內再被 ring_at_min 補點一次

    def _has_active_character_revive_session(self, screen, now, exit_box=None) -> bool:
        if exit_box is None:
            exit_box = self._character_exit_box(screen)
        if self._character_revive_session_mode == CHARACTER_REVIVE_SESSION_NORMAL:
            # Bootstrap uses strict target-plus-ring evidence; an existing
            # session keeps the looser ring signal to survive animation frames
            # where the target mask is temporarily invisible.
            present = exit_box is not None and self._has_character_ring(screen)
        elif self._character_revive_session_mode == CHARACTER_REVIVE_SESSION_OVERLAY:
            present = exit_box is not None and self._has_character_death_signature(
                screen
            )
        else:
            present = False
        if present:
            self._last_character_death_seen = now
            return True
        return (
            self._last_character_death_seen is not None
            and now - self._last_character_death_seen
            <= CHARACTER_REVIVE_DETECTION_GRACE
        )

    def _activate_character_revive(self) -> None:
        self._character_revive_activated = True
        action = "would click" if self.character_revive_dry_run else "clicking"
        state.logger.info("Character revive activation %s center", action)
        if self.character_revive_dry_run:
            return
        left, top, width, height = get_window_rect()
        click_at((left + width // 2, top + height // 2))
        time.sleep(self.delay_character_action)

    def _poll_character_death_screen(self, observation=None) -> None:
        """Throttle repeated checks while the death/revive screen persists.

        Polls at a slow cadence when nothing actionable is visible (no
        measured revive target) so the same-screen loop stops flooding the
        log with repeated successful checks; tracking resumes at full speed
        once a target is measured and a click may be imminent.
        """
        if observation is None or not observation["target_measured"]:
            if self.character_death_poll_interval > 0:
                time.sleep(self.character_death_poll_interval)

    def _observe_character_revive(self, screen, now=None):
        if self._character_revive_session_expired:
            return None
        if now is None:
            now = time.monotonic()
        if self._character_revive_session_started is None:
            self._character_revive_session_started = now
        if (
            now - self._character_revive_session_started
            >= self.character_revive_session_deadline
        ):
            if not self._character_revive_deadline_logged:
                state.logger.warning("角色復活圓環追蹤逾時，未以無證據點擊")
                self._character_revive_deadline_logged = True
            self._character_revive_session_expired = True
            self._character_revive_activated = False
            self._character_revive_session_mode = None
            self._character_revive_timeout_until = (
                now + CHARACTER_REVIVE_TIMEOUT_COOLDOWN
            )
            return None
        # 密集採樣: 一個收縮週期僅約 0.3 秒，必須在同一次 check 內連續取幀，
        # 直到成功點擊、畫面確認完成或預算耗盡。
        budget_end = now + CHARACTER_REVIVE_OBSERVE_BUDGET
        last_log = 0.0
        while time.monotonic() <= budget_end:
            screen = get_window_screen()
            observation = self.character_revive_observation(screen)
            now = time.monotonic()
            if observation is None:
                if now - last_log >= CHARACTER_REVIVE_LOG_INTERVAL:
                    state.logger.debug("角色復活圓環未偵測到有效觀測")
                    last_log = now
                time.sleep(CHARACTER_REVIVE_OBSERVE_STEP)
                continue
            should_click = (
                (
                    observation["ring_settled"]
                    and self._character_revive_click_allowed(now)
                )
                or (
                    observation["ring_at_min"]
                    and observation["measured"]
                    and observation["shrink_observed"]
                    and self._character_revive_click_allowed(now)
                )
                or (
                    observation["ring_at_absolute_min"]
                    and observation["measured"]
                    and self._character_revive_click_allowed(now)
                )
                or (
                    observation["predicted_ready"]
                    and self._character_revive_click_allowed(now)
                )
            )
            if should_click:
                self._click_character_revive(now, observation)
                # 每點擊一次重新觀察與計算
                self._reset_character_revive_history()
                return observation
            if now - last_log >= CHARACTER_REVIVE_LOG_INTERVAL:
                state.logger.debug(
                    "角色復活圓環 radius=%.1f target=%.1f target_measured=%s "
                    "target_currently_measured=%s target_mode=%s "
                    "target_confidence=%.2f "
                    "target_tolerance=%.1f confidence=%.2f velocity=%s measured=%s "
                    "predicted_ready=%s predicted_eta=%s predicted_velocity=%s",
                    observation["radius"],
                    observation["target_radius"],
                    observation["target_measured"],
                    observation["target_currently_measured"],
                    observation["target_detection_mode"],
                    observation["target_confidence"],
                    observation["target_tolerance"],
                    observation["confidence"],
                    self._format_measurement(observation["velocity"]),
                    observation["measured"],
                    observation["predicted_ready"],
                    self._format_measurement(observation["predicted_eta"]),
                    self._format_measurement(observation["predicted_velocity"]),
                )
                last_log = now
            time.sleep(CHARACTER_REVIVE_OBSERVE_STEP)
        state.logger.debug("角色復活圓環觀測預算耗盡")
        return None

    def _character_revive_click_allowed(self, now) -> bool:
        return self._character_revive_last_click is None or (
            now - self._character_revive_last_click >= CHARACTER_REVIVE_CLICK_COOLDOWN
        )

    def _click_character_revive(self, now, observation) -> None:
        self._character_revive_last_click = now
        action = "would click" if self.character_revive_dry_run else "clicking"
        state.logger.info(
            "Character revive %s center radius=%.1f target=%.1f target_measured=%s "
            "target_currently_measured=%s target_mode=%s target_confidence=%.2f "
            "target_tolerance=%.1f confidence=%.2f velocity=%s measured=%s "
            "measured_frames=%s ring_at_min=%s ring_settled=%s predicted_ready=%s "
            "predicted_eta=%s predicted_velocity=%s click_margin=%.2f",
            action,
            observation.get("radius", 0.0),
            observation.get("target_radius", 0.0),
            observation.get("target_measured", False),
            observation.get("target_currently_measured", False),
            observation.get("target_detection_mode", "none"),
            observation.get("target_confidence", 0.0),
            observation.get("target_tolerance", 0.0),
            observation.get("confidence", 0.0),
            self._format_measurement(observation.get("velocity")),
            observation.get("measured", False),
            observation.get("measured_frames", 0),
            observation.get("ring_at_min", False),
            observation.get("ring_settled", False),
            observation.get("predicted_ready", False),
            self._format_measurement(observation.get("predicted_eta")),
            self._format_measurement(observation.get("predicted_velocity")),
            CHARACTER_REVIVE_CLICK_RADIUS_MARGIN,
        )
        if self.character_revive_dry_run:
            return
        left, top, width, height = get_window_rect()
        click_at((left + width // 2, top + height // 2))
        self._character_revive_target_clicked = True
        self._character_revive_post_click_until = (
            now + CHARACTER_REVIVE_POST_CLICK_GRACE
        )
        time.sleep(self.delay_character_action)

    @staticmethod
    def _format_measurement(value) -> str:
        return "n/a" if value is None else f"{value:.2f}"

    def character_revive_observation(self, screen):
        """Return the central revival ring measurement without performing input."""
        if screen is None or screen.size == 0:
            return None
        height, width = screen.shape[:2]
        now = time.monotonic()
        (
            target_radius,
            target_confidence,
            target_measured,
            target_currently_measured,
            target_detection_mode,
        ) = self._character_revive_target_measurement(screen, now)
        if target_measured:
            tolerance = max(
                CHARACTER_TARGET_MIN_TOLERANCE,
                target_radius * CHARACTER_TARGET_TOLERANCE_RATIO,
            )
        elif (
            target_detection_mode == "neutral_pending"
            and target_confidence >= CHARACTER_NEUTRAL_TARGET_PENDING_MIN_CONFIDENCE
        ):
            tolerance = max(
                CHARACTER_TARGET_MIN_TOLERANCE,
                target_radius * CHARACTER_TARGET_TOLERANCE_RATIO,
            )
        else:
            target_radius = CHARACTER_REVIVE_TARGET_RADIUS * height
            tolerance = CHARACTER_REVIVE_TARGET_TOLERANCE * height
        # 靜態白環(中央盤緣/UI 圓圈)的時間連續性確認: 同一半徑持續出現
        # 高白環分數才視為靜態。移動中的 ring 穿過 r60-120 時半徑會變化,
        # 無法連續確認, 不會誤觸發。
        self._character_revive_update_static_ring(screen, now)
        # 允許測量收縮到 target 以下的 ring，只保留物理下限
        min_radius = CHARACTER_REVIVE_RING_MIN_RADIUS
        radius, confidence = self._character_revive_ring_measurement(
            screen, now, min_radius
        )
        measured = radius is not None
        if measured:
            if confidence < CHARACTER_REVIVE_MIN_MEASURED_CONFIDENCE:
                # Low-confidence candidates are commonly the static outer
                # ring; do not let them reset or poison the shrinking history.
                radius = None
                measured = False
            else:
                consistency = self._character_revive_measurement_is_consistent(
                    radius, now
                )
                if consistency == "reset":
                    # 外圈重置：新一輪收縮開始，清空歷史重新觀察。
                    # 只有在前一週期已完整收縮過(觀察到遞減)才計入週期數,
                    # 避免漸進擴張的多幀各自被當成一次 reset 而灌水週期數。
                    state.logger.debug(
                        "角色復活圓環重置，開始新一輪收縮 radius=%.1f", radius
                    )
                    self._reset_character_revive_history()
                    self._character_revive_shrink_history.clear()
                    if self._character_revive_cycle_shrinking:
                        self._character_revive_cycle_count += 1
                        self._character_revive_model_expired = False
                        if self._character_revive_cycle_count >= 2:
                            # 至少觀察一個完整週期後, 預測模型才可用
                            self._character_revive_model_ready = True
                    # 週期結束: 以本週期實際最小半徑更新 session 最小值
                    if self._character_revive_cycle_min_radius is not None and (
                        self._character_revive_session_min_radius is None
                        or self._character_revive_cycle_min_radius
                        < self._character_revive_session_min_radius
                    ):
                        self._character_revive_session_min_radius = (
                            self._character_revive_cycle_min_radius
                        )
                    # 新一輪週期開始: 收縮證據歸零; 週期最小半徑跨週期保留
                    # (它是「ring 真正到達過的最小半徑」, 重置外圈不應把它
                    # 洗掉, 否則收縮途中的中繼半徑會立刻被當成最小值)
                    self._character_revive_cycle_shrinking = False
                    self._character_revive_cycle_started_at = now
                    self._character_revive_cycle_max_radius = radius
                    # 新一輪開始: 清除「錯過預測窗口」標記
                    self._character_revive_cycle_missed_predict = False
                    self._character_revive_history.append((now, radius, confidence))
                elif not consistency:
                    radius = None
                    measured = False
                else:
                    self._character_revive_history.append((now, radius, confidence))
        # 以「本幀更新前」的週期最小半徑判斷是否到達最小:
        # 使用雙向鄰近比較 (abs(radius - cycle_min) <= tolerance)。
        # 單向的 radius <= cycle_min + tolerance 在 ring 縮到比 cycle_min
        # 更小時仍然成立, 會讓收縮途中(例如錯誤的最小值 250 之後繼續
        # 縮到 200)誤判為「已到達最小」而提前點擊。
        prev_radius = self._character_revive_last_radius
        at_session_min = (
            measured
            and self._character_revive_cycle_min_radius is not None
            and abs(radius - self._character_revive_cycle_min_radius)
            <= CHARACTER_REVIVE_MIN_RADIUS_TOLERANCE
        )
        # 仍在持續收縮(比前一幀明顯更小)時, 即使落在最小半徑容差帶內,
        # 也不算「停留於最小」: 例如 250→247→244 會逐幀累積 dwell,
        # 造成 ring 還在縮小時就提前點擊。
        still_shrinking = (
            prev_radius is not None
            and radius is not None
            and radius < prev_radius - CHARACTER_REVIVE_CYCLE_MIN_STABLE_TOLERANCE
        )
        if measured and radius is not None:
            # 週期最小半徑: 只有在 ring「停留」於該半徑時才更新(跨週期保留)。
            # 停留判定使用嚴格容差(2px): 收縮途中的中繼半徑(如 367→362,
            # 相差 5px)不會被當成停留; 只有 ring 真正到達並停留在本輪
            # 最低點(如 120/55)才更新, 避免 ring_at_min 提前點擊。
            if (
                self._character_revive_last_radius is not None
                and abs(radius - self._character_revive_last_radius)
                <= CHARACTER_REVIVE_CYCLE_MIN_STABLE_TOLERANCE
            ):
                self._character_revive_stable_radius_frames += 1
            else:
                self._character_revive_stable_radius_frames = 0
            self._character_revive_last_radius = radius
            if self._character_revive_stable_radius_frames >= (
                CHARACTER_REVIVE_CYCLE_MIN_STABLE_FRAMES - 1
            ):
                if (
                    self._character_revive_cycle_min_radius is None
                    or radius < self._character_revive_cycle_min_radius
                ):
                    self._character_revive_cycle_min_radius = radius
        if not measured:
            prediction = self._character_revive_prediction(now, height)
            if prediction is None:
                return None
            radius, velocity, confidence = prediction
        else:
            velocity = self._character_revive_velocity()
        measured_frames = len(self._character_revive_history)
        has_evidence = (
            measured_frames >= CHARACTER_REVIVE_MIN_MEASURED_FRAMES
            and velocity is not None
            and -CHARACTER_REVIVE_MAX_SHRINK_SPEED
            <= velocity
            <= -CHARACTER_REVIVE_MIN_SHRINK_SPEED
            and min(entry[2] for entry in self._character_revive_history)
            >= CHARACTER_REVIVE_MIN_MEASURED_CONFIDENCE
        )
        # 本週期內只要觀察到一次有效收縮，就算有收縮證據。
        # 一旦 ring 到達 target 後靜止(velocity 變 None)仍可點擊；
        # 但啟動時若 ring 已在窗口內(從未觀察到收縮)則不點擊。
        if has_evidence:
            self._character_revive_cycle_shrinking = True
        shrink_observed = self._character_revive_cycle_shrinking
        # 收縮後停止偵測: 連續多幀半徑幾乎不變 => ring 已收縮到最小(到達中心)。
        # 當 target 無法測量時(黃色中心不可見)此訊號是唯一可靠的點擊依據。
        ring_settled = self._character_revive_ring_settled()
        if at_session_min and not still_shrinking:
            self._character_revive_min_dwell += 1
        else:
            self._character_revive_min_dwell = 0
        ring_at_min = at_session_min and self._character_revive_min_dwell >= 2
        predicted_eta, predicted_velocity = self._character_revive_predicted_arrival(
            radius, now
        )
        if (
            predicted_eta is not None
            and predicted_eta < CHARACTER_REVIVE_PREDICT_LEAD_MIN
        ):
            # 錯過本輪預測點擊窗口(eta 已低於下界, 環即將/已到達中心):
            # 點擊此刻送出會在生效時錯過 min, 抑制本輪其餘補點,
            # 等下一輪收縮(週期僅 ~0.3s)再點。
            self._character_revive_cycle_missed_predict = True
        elif (
            predicted_eta is None
            and self._character_revive_model_ready
            and radius is not None
            and self._character_revive_session_min_radius is not None
            and radius
            <= self._character_revive_session_min_radius
            + CHARACTER_REVIVE_MIN_RADIUS_TOLERANCE
        ):
            # 模型就緒但環已到達/超過最小半徑(觀測間隔跳過預測窗口,
            # predicted_arrival 回傳 None): 本輪已錯過窗口, 抑制補點。
            self._character_revive_cycle_missed_predict = True
        # 絕對最小半徑: 某些極端畫面的環收縮到非常接近中心(radius 很小)
        # 且幾乎不停留(下一個觀測就重置), cycle_min/停留機制無法建立。
        # 在 predicted_arrival 之後計算, 讓 shrink_history 已含當前幀。
        # 關鍵: 點擊必須在「環到達中心」的時刻生效。系統延遲(點擊送出到
        # 遊戲生效)約 50-60ms, 因此點擊要在環離中心還有 CLICK_LATENCY 的
        # 時間時送出 — 即 eta = radius / -velocity 落在 [ETA_MIN, ETA_MAX]。
        # 太晚送出(eta 過小): 點擊生效時環已擴張 → 失敗(扣血);
        # 太早送出(eta 過大): 點擊生效時環還沒到中心 → 失敗。
        # 其餘守衛: 收縮軌跡 >= 2 樣本、外圈 cycle_max >= 150(排除靜態
        # r80 盤緣/死亡等待靜態環)。
        at_absolute_min = (
            measured
            and velocity is not None
            and velocity < 0
            and len(self._character_revive_shrink_history) >= 2
            and self._character_revive_cycle_max_radius is not None
            and self._character_revive_cycle_max_radius
            >= CHARACTER_REVIVE_ABSOLUTE_MIN_OUTER_RADIUS
            and CHARACTER_REVIVE_ABSOLUTE_MIN_ETA_MIN
            <= radius / -velocity
            <= CHARACTER_REVIVE_ABSOLUTE_MIN_ETA_MAX
        )
        # 記錄 absolute_min 為獨立訊號: 它本身已含「本週期/歷史見過更大環」
        # 的收縮證據, 點擊條件不需再額外要求 shrink_observed
        ring_at_absolute_min = at_absolute_min
        if at_absolute_min:
            ring_at_min = True
        if self._character_revive_cycle_missed_predict:
            # 本輪預測窗口已錯過: 太晚的 at_min/settled 補點會失效,
            # 一律抑制, 等待下一輪收縮。
            ring_at_min = False
            ring_settled = False
            ring_at_absolute_min = False
        predicted_ready = (
            measured
            and predicted_eta is not None
            and CHARACTER_REVIVE_PREDICT_LEAD_MIN
            <= predicted_eta
            <= CHARACTER_REVIVE_PREDICT_LEAD
            and radius
            > (
                self._character_revive_session_min_radius
                if self._character_revive_session_min_radius is not None
                else 0.0
            )
        )
        return {
            "radius": radius,
            "target_radius": target_radius,
            "target_measured": target_measured,
            "target_currently_measured": target_currently_measured,
            "target_detection_mode": target_detection_mode,
            "target_confidence": target_confidence,
            "target_tolerance": tolerance,
            "click_margin": CHARACTER_REVIVE_CLICK_RADIUS_MARGIN,
            "confidence": confidence,
            "velocity": velocity,
            "measured": measured,
            "measured_frames": measured_frames,
            "ring_settled": ring_settled and shrink_observed,
            "ring_at_min": ring_at_min,
            "ring_at_absolute_min": ring_at_absolute_min,
            "shrink_observed": shrink_observed,
            "predicted_eta": predicted_eta,
            "predicted_velocity": predicted_velocity,
            "predicted_ready": predicted_ready,
        }

    def _character_revive_ring_settled(self) -> bool:
        """True when the ring radius has been nearly constant for the last few
        frames, i.e. the shrink cycle has reached its minimum.

        Guards against mid-shrink pauses: the ring must have first been seen
        to shrink from a substantially larger radius before a stable tail
        counts as settled.
        """
        history = self._character_revive_history
        if len(history) < CHARACTER_REVIVE_SETTLE_FRAMES:
            return False
        recent = list(history)[-CHARACTER_REVIVE_SETTLE_FRAMES:]
        radii = [entry[1] for entry in recent]
        if max(radii) - min(radii) > CHARACTER_REVIVE_SETTLE_TOLERANCE:
            return False
        tail = radii[0]
        # 收縮距離驗證: 歷史中必須出現過明顯大於穩定尾端的半徑,
        # 否則可能是收縮中途的短暫停頓而非真正到達中心。
        max_seen = max(entry[1] for entry in history)
        return max_seen >= tail * 2.0 + CHARACTER_REVIVE_RING_MIN_RADIUS * 2

    def _character_revive_target_measurement(self, screen, now):
        radius, confidence, detection_mode = self._character_revive_target_candidate(
            screen
        )
        height = screen.shape[0]
        max_radius = (
            CHARACTER_TARGET_DISK_MAX_RADIUS * height
            if detection_mode == "disk"
            else CHARACTER_TARGET_MAX_RADIUS * height
        )
        if (
            radius is not None
            and confidence >= CHARACTER_TARGET_MIN_CONFIDENCE
            and CHARACTER_REVIVE_DIRECT_TARGET_MIN_RADIUS * height
            <= radius
            <= max_radius
        ):
            if (
                self._character_revive_stable_target is None
                or self._character_revive_stable_target_at is None
                or now - self._character_revive_stable_target_at
                > CHARACTER_TARGET_CACHE_TTL
            ):
                stable_radius = radius
            else:
                previous_radius = self._character_revive_stable_target[0]
                stable_radius = 0.75 * previous_radius + 0.25 * radius
            self._character_revive_stable_target = (
                float(stable_radius),
                float(confidence),
            )
            self._character_revive_stable_target_at = now
        history = self._character_revive_target_history
        if history and now - history[-1][0] > CHARACTER_TARGET_HISTORY_BRIDGE:
            history.clear()
        if detection_mode is not None and history and history[-1][3] != detection_mode:
            history.clear()
        if detection_mode is not None:
            history.append((now, radius, confidence, detection_mode))
        weak_confirmed = self._weak_target_confirmed(history)
        neutral_confirmed = self._target_mode_confirmed(history, "neutral")
        current_reliable = (
            detection_mode in ("strong", "disk")
            or (detection_mode == "weak" and weak_confirmed)
            or (
                detection_mode == "neutral"
                and (
                    neutral_confirmed
                    or confidence >= CHARACTER_NEUTRAL_TARGET_STRONG_CONFIDENCE
                )
            )
        )
        if current_reliable:
            return (
                float(np.median([entry[1] for entry in history])),
                float(np.median([entry[2] for entry in history])),
                True,
                True,
                detection_mode,
            )
        if history and now - history[-1][0] <= CHARACTER_TARGET_HISTORY_BRIDGE:
            if (
                history[-1][3] in ("strong", "disk")
                or weak_confirmed
                or neutral_confirmed
            ):
                return (
                    float(np.median([entry[1] for entry in history])),
                    float(np.median([entry[2] for entry in history])),
                    True,
                    False,
                    "history",
                )
        if (
            self._character_revive_stable_target is not None
            and self._character_revive_stable_target_at is not None
            and now - self._character_revive_stable_target_at
            <= CHARACTER_TARGET_CACHE_TTL
        ):
            return (
                self._character_revive_stable_target[0],
                self._character_revive_stable_target[1],
                True,
                False,
                "history",
            )
        if detection_mode == "weak":
            return None, confidence, False, False, "weak_pending"
        if detection_mode == "neutral":
            return radius, confidence, False, False, "neutral_pending"
        return None, 0.0, False, False, "none"

    def _weak_target_confirmed(self, history) -> bool:
        return self._target_mode_confirmed(history, "weak")

    def _target_mode_confirmed(self, history, mode) -> bool:
        if len(history) < CHARACTER_WEAK_TARGET_CONFIRMATION_SAMPLES:
            return False
        samples = list(history)[-CHARACTER_WEAK_TARGET_CONFIRMATION_SAMPLES:]
        if any(entry[3] != mode for entry in samples):
            return False
        radii = [entry[1] for entry in samples]
        if mode in ("neutral", "disk"):
            tolerance = CHARACTER_NEUTRAL_TARGET_CONFIRMATION_TOLERANCE
        else:
            tolerance = max(
                CHARACTER_WEAK_TARGET_CONFIRMATION_TOLERANCE,
                max(radii) * CHARACTER_WEAK_TARGET_CENTER_OFFSET_RATIO,
            )
        return max(radii) - min(radii) <= tolerance

    def _measure_character_revive_target_radius(self, screen):
        """Return a centered target radius and confidence for existing callers."""
        radius, confidence, _ = self._character_revive_target_candidate(screen)
        return radius, confidence

    def _character_revive_target_candidate(self, screen):
        """Prefer the yellow center disk; fall back to constrained geometry.

        The yellow target circle is dim in video captures and may be partially
        occluded on the right side, so the relaxed disk detector is tried
        first and the legacy strong/weak/neutral pass is kept as fallback.
        """
        height, width = screen.shape[:2]
        disk = self._character_revive_target_disk(screen)
        if disk[0] is not None:
            return disk[0], disk[1], "disk"
        crop, _, _ = self._crop_normalized(screen, CHARACTER_TARGET_REGION)
        if crop.size == 0:
            return None, 0.0, None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        crop_x = int(CHARACTER_TARGET_REGION[0] * width)
        crop_y = int(CHARACTER_TARGET_REGION[2] * height)
        center = np.array([width / 2 - crop_x, height / 2 - crop_y])
        max_radius = min(
            CHARACTER_TARGET_MAX_RADIUS * height,
            width / 2,
            height / 2,
        )
        strong = self._target_candidate_from_mask(
            cv2.inRange(hsv, CHARACTER_TARGET_HSV_LOWER, CHARACTER_TARGET_HSV_UPPER),
            center,
            CHARACTER_TARGET_MIN_RADIUS,
            max_radius,
            CHARACTER_TARGET_MIN_COMPONENT_PIXELS,
            CHARACTER_TARGET_MIN_ANGULAR_COVERAGE,
            CHARACTER_TARGET_MIN_CONFIDENCE,
            CHARACTER_TARGET_CENTER_OFFSET_RATIO,
            CHARACTER_TARGET_ANGULAR_BINS,
        )
        if strong[0] is not None:
            return strong[0], strong[1], "strong"
        weak = self._target_candidate_from_mask(
            cv2.inRange(
                hsv,
                CHARACTER_WEAK_TARGET_HSV_LOWER,
                CHARACTER_WEAK_TARGET_HSV_UPPER,
            ),
            center,
            CHARACTER_WEAK_TARGET_MIN_RADIUS,
            max_radius,
            CHARACTER_WEAK_TARGET_MIN_COMPONENT_PIXELS,
            CHARACTER_WEAK_TARGET_MIN_ANGULAR_COVERAGE,
            CHARACTER_WEAK_TARGET_MIN_CONFIDENCE,
            CHARACTER_WEAK_TARGET_CENTER_OFFSET_RATIO,
            CHARACTER_WEAK_TARGET_ANGULAR_BINS,
        )
        if weak[0] is not None:
            return weak[0], weak[1], "weak"
        neutral = self._target_candidate_from_mask(
            cv2.inRange(
                hsv,
                CHARACTER_NEUTRAL_TARGET_HSV_LOWER,
                CHARACTER_NEUTRAL_TARGET_HSV_UPPER,
            ),
            center,
            CHARACTER_TARGET_MIN_RADIUS,
            max_radius,
            CHARACTER_TARGET_MIN_COMPONENT_PIXELS,
            CHARACTER_TARGET_MIN_ANGULAR_COVERAGE,
            CHARACTER_NEUTRAL_TARGET_MIN_CONFIDENCE,
            CHARACTER_TARGET_CENTER_OFFSET_RATIO,
            CHARACTER_TARGET_ANGULAR_BINS,
            CHARACTER_NEUTRAL_TARGET_MIN_FILL_RATIO,
        )
        if neutral[0] is not None:
            return neutral[0], neutral[1], "neutral"
        return None, 0.0, None

    def _character_revive_target_disk(self, screen):
        """Detect the yellow center disk with relaxed thresholds.

        The target is a filled circle whose radius varies between occasions
        and whose right half may be occluded, so only the largest yellow
        component near the screen center is used.
        """
        height, width = screen.shape[:2]
        crop, _, _ = self._crop_normalized(
            screen,
            (
                0.5 - CHARACTER_TARGET_DISK_CENTER_SEARCH / 2,
                0.5 + CHARACTER_TARGET_DISK_CENTER_SEARCH / 2,
                0.5 - CHARACTER_TARGET_DISK_CENTER_SEARCH / 2,
                0.5 + CHARACTER_TARGET_DISK_CENTER_SEARCH / 2,
            ),
        )
        if crop.size == 0:
            return None, 0.0
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv, CHARACTER_TARGET_DISK_HSV_LOWER, CHARACTER_TARGET_DISK_HSV_UPPER
        )
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        max_radius = CHARACTER_TARGET_DISK_MAX_RADIUS * height
        best = (None, 0.0)
        for label in range(1, count):
            area = stats[label, cv2.CC_STAT_AREA]
            if area < CHARACTER_TARGET_DISK_MIN_AREA:
                continue
            component_width = stats[label, cv2.CC_STAT_WIDTH]
            component_height = stats[label, cv2.CC_STAT_HEIGHT]
            radius = max(component_width, component_height) / 2
            if not (CHARACTER_TARGET_DISK_MIN_RADIUS <= radius <= max_radius):
                continue
            ccx, ccy = centroids[label]
            offset = np.hypot(crop.shape[1] / 2 - ccx, crop.shape[0] / 2 - ccy)
            if offset > radius * 0.8:
                continue
            fill_ratio = area / (np.pi * radius * radius)
            confidence = 0.6 * min(1.0, fill_ratio) + 0.4 * max(
                0.0, 1.0 - offset / max(radius, 1.0)
            )
            if confidence > best[1]:
                best = (float(radius), float(confidence))
        return best

    @staticmethod
    def _target_candidate_from_mask(
        mask,
        center,
        min_radius,
        max_radius,
        min_component_pixels,
        min_angular_coverage,
        min_confidence,
        center_offset_ratio,
        angular_bins,
        min_fill_ratio=0.0,
    ):
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        best = (None, 0.0)
        for label in range(1, count):
            area = stats[label, cv2.CC_STAT_AREA]
            if area < min_component_pixels:
                continue
            ys, xs = np.nonzero(labels == label)
            distances = np.hypot(xs - center[0], ys - center[1])
            radius = float(np.percentile(distances, 98.0))
            if not min_radius <= radius <= max_radius:
                continue
            if area / (np.pi * radius**2) < min_fill_ratio:
                continue
            centroid_distance = float(np.hypot(*(centroids[label] - center)))
            allowed_offset = max(min_radius, radius * center_offset_ratio)
            if centroid_distance > allowed_offset:
                continue
            angles = np.arctan2(ys - center[1], xs - center[0])
            bins = np.floor((angles + np.pi) * angular_bins / (2 * np.pi)).astype(int)
            coverage = np.unique(np.clip(bins, 0, angular_bins - 1)).size / angular_bins
            if coverage < min_angular_coverage:
                continue
            center_score = max(0.0, 1.0 - centroid_distance / allowed_offset)
            pixel_score = min(1.0, area / (2 * np.pi * radius))
            candidate_confidence = (
                0.50 * coverage + 0.35 * center_score + 0.15 * pixel_score
            )
            if (
                candidate_confidence >= min_confidence
                and candidate_confidence > best[1]
            ):
                best = (radius, float(candidate_confidence))
        return best

    def _character_revive_measurement_is_consistent(self, radius, now):
        """Classify a new ring measurement against the shrinking history.

        Returns:
          True     - measurement accepted as part of the current shrink cycle
          False    - measurement rejected (keep waiting)
          "reset"  - the ring jumped back to the outer band: a new shrink
                     cycle started, callers should clear history and accept
        """
        history = self._character_revive_history
        if not history:
            return True
        previous_time, previous_radius, _ = history[-1]
        gap = max(now - previous_time, 0.001)
        shrink = previous_radius - radius
        if shrink >= 0:
            # Shrinking: accept within the physical speed bound.
            return shrink <= CHARACTER_REVIVE_MAX_SHRINK_SPEED * gap
        # Growing. The ring expands back outward before the next shrink cycle;
        # any growth beyond measurement noise starts a new cycle. Rejecting
        # growth would discard otherwise valid observations.
        if radius - previous_radius > CHARACTER_REVIVE_MIN_RADIUS_TOLERANCE:
            return "reset"
        # 最小半徑附近的微小抖動(1~2px)是測量雜訊, 視為可接受
        return True

    def _character_revive_update_static_ring(self, screen, now) -> None:
        """以時間連續性確認中央靜態白環。

        某些復活畫面在 r~60-120 有恆定的白環(黃色圓盤邊緣/UI 圓圈),
        會蓋住真正收縮的大白環。此處追蹤每一幀的白環候選半徑, 只有
        同一半徑連續多幀出現(>=STATIC_RING_CONFIRM_FRAMES)才確認,
        避免把收縮途中穿過此帶的移動 ring 誤判為靜態。
        """
        height, width = screen.shape[:2]
        center_x, center_y = width // 2, height // 2
        hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
        angles = np.linspace(
            0, 2 * np.pi, CHARACTER_REVIVE_SAMPLE_COUNT, endpoint=False
        )
        best_radius, best_score = None, 0.0
        for radius in range(
            CHARACTER_REVIVE_STATIC_RING_MIN_RADIUS,
            CHARACTER_REVIVE_STATIC_RING_MAX_RADIUS + 1,
            2,
        ):
            xs = np.rint(center_x + radius * np.cos(angles)).astype(int)
            ys = np.rint(center_y + radius * np.sin(angles)).astype(int)
            valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
            saturation = hsv[ys.clip(0, height - 1), xs.clip(0, width - 1), 1]
            value = hsv[ys.clip(0, height - 1), xs.clip(0, width - 1), 2]
            score = float((valid & (saturation <= 70) & (value >= 180)).mean())
            if score >= CHARACTER_REVIVE_STATIC_RING_MIN_SCORE and score > best_score:
                best_radius, best_score = radius, score
        # 分離「候選」與「已確認」: 同一半徑連續多幀出現才確認靜態,
        # 未確認前不切換 ring 測量模式(避免移動 ring 穿過此帶時誤判)。
        if (
            best_radius is not None
            and best_radius == self._character_revive_static_ring_pending
        ):
            self._character_revive_static_ring_frames += 1
            if (
                self._character_revive_static_ring_frames
                >= CHARACTER_REVIVE_STATIC_RING_CONFIRM_FRAMES
            ):
                if self._character_revive_static_ring_radius is None:
                    # 確認的瞬間, 清除確認前由靜態盤緣污染的量測狀態:
                    # 確認前的 ring 半徑都是盤緣(r~80), 不是真正的大環,
                    # 若不清除, 週期最小值會被設成 80, ring_at_min 永遠不觸發
                    self._character_revive_cycle_min_radius = None
                    self._character_revive_last_radius = None
                    self._character_revive_stable_radius_frames = 0
                    self._character_revive_min_dwell = 0
                self._character_revive_static_ring_radius = best_radius
        else:
            self._character_revive_static_ring_pending = best_radius
            self._character_revive_static_ring_frames = 1

    def _character_revive_ring_measurement(self, screen, now, min_radius):
        height, width = screen.shape[:2]
        center_x, center_y = width // 2, height // 2
        max_radius = min(int(CHARACTER_REVIVE_MAX_RADIUS * height), width // 2)
        min_radius = max(
            int(np.floor(min_radius)), int(CHARACTER_REVIVE_RING_MIN_RADIUS)
        )
        if min_radius > max_radius:
            return None, 0.0
        angles = np.linspace(
            0, 2 * np.pi, CHARACTER_REVIVE_SAMPLE_COUNT, endpoint=False
        )
        hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
        sat_ch = hsv[:, :, 1]
        val_ch = hsv[:, :, 2]

        # 某些復活畫面中央有「靜態白環」(黃色圓盤邊緣/UI 圓圈, r~60-120,
        # 白環分數恆定 ~1.0), 會蓋住真正持續收縮的大白環, 使 ring 半徑
        # 永遠測到靜態環而無法觸發點擊。觀測流程會以時間連續性確認靜態
        # 白環存在, 並傳入 static_radius; 此時只搜尋其外側 r120+ 的亮帶
        # (真正收縮的大環)。舊畫面無靜態白環時維持原邏輯。
        static_radius = self._character_revive_static_ring_radius
        if static_radius is not None:
            search_min = max(min_radius, CHARACTER_REVIVE_STATIC_RING_EXCLUDE_MIN)
        else:
            search_min = min_radius
        if search_min > max_radius:
            return None, 0.0
        radii = np.arange(search_min, max_radius + 1)
        # 向量化: 一次取樣所有半徑 x 角度 的像素 (原逐半徑迴圈 ~100ms,
        # 收縮週期僅 ~0.3s, 觀測必須夠快才能抓準點擊時機)
        xs = np.rint(center_x + radii[:, None] * np.cos(angles)).astype(np.int32)
        ys = np.rint(center_y + radii[:, None] * np.sin(angles)).astype(np.int32)
        valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        xs_c = xs.clip(0, width - 1)
        ys_c = ys.clip(0, height - 1)
        sample_sat = sat_ch[ys_c, xs_c]
        sample_val = val_ch[ys_c, xs_c]
        if static_radius is not None:
            # 有靜態盤緣: 大環是低飽和亮帶(白/亮灰), 用較寬鬆門檻
            mask_ok = (
                valid
                & (sample_sat <= CHARACTER_REVIVE_FALLBACK_MAX_SATURATION)
                & (sample_val >= 90)
            )
            scores = mask_ok.mean(axis=1)
            candidates = np.flatnonzero(scores >= CHARACTER_REVIVE_SCORE_THRESHOLD)
        else:
            mask_ok = valid & (sample_sat <= 70) & (sample_val >= 180)
            scores = mask_ok.mean(axis=1)
            candidates = np.flatnonzero(scores >= CHARACTER_REVIVE_SCORE_THRESHOLD)
            if candidates.size == 0:
                mask_ok = (
                    valid
                    & (sample_sat <= CHARACTER_REVIVE_FALLBACK_MAX_SATURATION)
                    & (sample_val >= CHARACTER_REVIVE_FALLBACK_MIN_VALUE)
                )
                scores = mask_ok.mean(axis=1)
                candidates = np.flatnonzero(
                    scores >= CHARACTER_REVIVE_FALLBACK_SCORE_THRESHOLD
                )
        if candidates.size == 0:
            return None, 0.0
        # 回歸最強候選: 一致性/重置分類交由 _character_revive_measurement_is_consistent
        # 處理, 這裡不再用前一幀半徑過濾, 以免把合法的外圈/重置觀測丟棄。
        index = int(candidates[np.argmax(scores[candidates])])
        return float(radii[index]), float(scores[index])

    def _character_revive_prediction(self, now, height):
        velocity = self._character_revive_velocity()
        if velocity is None or velocity >= 0:
            return None
        last_time, last_radius, _ = self._character_revive_history[-1]
        gap = now - last_time
        if gap > CHARACTER_REVIVE_MAX_PREDICTION_GAP:
            return None
        radius = last_radius + velocity * gap
        if not 0 <= radius <= CHARACTER_REVIVE_MAX_RADIUS * height:
            return None
        confidence = min(entry[2] for entry in self._character_revive_history)
        return radius, velocity, confidence

    def _character_revive_velocity(self):
        history = self._character_revive_history
        if len(history) < CHARACTER_REVIVE_MIN_MEASURED_FRAMES:
            return None
        times = np.array([entry[0] for entry in history])
        radii = np.array([entry[1] for entry in history])
        deltas = np.diff(radii)
        if not np.all(deltas < 0) or times[-1] == times[0]:
            return None
        velocity = float(np.polyfit(times - times[0], radii, 1)[0])
        if velocity < -CHARACTER_REVIVE_MAX_SHRINK_SPEED:
            return None
        return velocity

    def _character_revive_predicted_arrival(self, radius, now):
        """預測 ring 到達最小半徑(中心)的剩餘秒數。

        收縮期內每幀半徑線性遞減, 取收縮緩衝區擬合速度即可預測到達時刻;
        預測模型限於最近 15 秒 / 至多兩個週期的觀察窗口。
        回傳 (eta, velocity), 無法預測時回傳 (None, None)。
        """
        if radius is None:
            return None, None
        # 模型觀察窗口 (epoch): 獨立於單一週期, 15 秒過期即重設模型重新學習
        if self._character_revive_model_epoch is None:
            self._character_revive_model_epoch = now
        elif now - self._character_revive_model_epoch > CHARACTER_REVIVE_MODEL_WINDOW:
            self._character_revive_shrink_history.clear()
            self._character_revive_cycle_started_at = None
            self._character_revive_cycle_max_radius = None
            # 注意: cycle_min_radius 是 session 級的最小半徑, 過期重學不清除
            self._character_revive_cycle_count = 0
            was_ready = self._character_revive_model_ready
            self._character_revive_model_ready = False
            self._character_revive_model_epoch = now
            # 只有已就緒(學過完整週期)的模型過期後才允許重建立即恢復;
            # 初次啟動從未就緒時, 過期後仍須重新通過週期門檻
            self._character_revive_model_expired = was_ready
            return None, None
        if self._character_revive_shrink_history:
            last_time, last_radius = self._character_revive_shrink_history[-1]
            if radius > last_radius + CHARACTER_REVIVE_MIN_RADIUS_TOLERANCE:
                # 明顯增長: 清空趨勢, 由 reset 分支重新開始新週期
                self._character_revive_shrink_history.clear()
                return None, None
            if radius == last_radius:
                # 重複取幀(取樣快於遊戲幀率)或最小半徑停留: 略過, 不破壞趨勢
                return None, None
            if radius > last_radius:
                # 微幅增長(測量雜訊): 趨勢反轉, 清空避免舊趨勢續用
                self._character_revive_shrink_history.clear()
                return None, None
        # 週期必須已開始才能預測; 若週期資訊因窗口過期遺失,
        # 以本次收縮起點重建週期, 不依賴外圈重置事件。
        # 僅在「曾建立完整週期模型後過期」時直接恢復就緒;
        # 初次啟動/一般情況仍須經 reset 分支的兩週期門檻。
        if self._character_revive_cycle_started_at is None:
            self._character_revive_cycle_started_at = now
            self._character_revive_cycle_max_radius = radius
            if self._character_revive_model_expired:
                self._character_revive_model_ready = True
                self._character_revive_model_expired = False
            self._character_revive_shrink_history.append((now, radius))
            return None, None
        self._character_revive_shrink_history.append((now, radius))
        if len(self._character_revive_shrink_history) > CHARACTER_REVIVE_PREDICT_FRAMES:
            self._character_revive_shrink_history.pop(0)
        if (
            self._character_revive_cycle_max_radius is None
            or radius > self._character_revive_cycle_max_radius
        ):
            self._character_revive_cycle_max_radius = radius
        recent = self._character_revive_shrink_history
        if (
            len(recent) < CHARACTER_REVIVE_PREDICT_MIN_SAMPLES
            or not self._character_revive_model_ready
        ):
            return None, None
        # 預測目標: session 最小半徑; 未建立時(極端畫面環不停留)以中心(0)為目標
        target_radius = self._character_revive_session_min_radius
        if target_radius is None:
            target_radius = 0.0
        # 排除最小半徑附近的抖動: ring 必須已從本週期外圈實質收縮
        if self._character_revive_cycle_max_radius is not None:
            span = self._character_revive_cycle_max_radius - target_radius
            if (
                span > 0
                and (self._character_revive_cycle_max_radius - radius)
                < span * CHARACTER_REVIVE_PREDICT_MIN_CONTRACTION
            ):
                return None, None
        times = np.array([entry[0] for entry in recent])
        radii = np.array([entry[1] for entry in recent])
        if times[-1] == times[0] or np.any(np.diff(radii) >= 0):
            return None, None
        velocity = float(np.polyfit(times - times[0], radii, 1)[0])
        if (
            velocity >= 0
            or velocity < -CHARACTER_REVIVE_MAX_SHRINK_SPEED
            or -velocity < CHARACTER_REVIVE_PREDICT_MIN_SPEED
        ):
            return None, None
        remaining = radius - target_radius
        if remaining <= 0:
            # 已到達最小半徑(中心): 無需預測, 由 ring_at_min / ring_settled 處理
            return None, None
        return remaining / -velocity, velocity

    def _find_player_death_revive_box(self, screen):
        player_death_keywords = tuple(
            get_text_mapping(key) for key in PLAYER_DEATH_TEXT_KEYS
        )
        boxes = self._ocr_boxes(screen, self.player_death_region)
        revive_box = None
        found_keywords = set()
        for text, box in boxes:
            for keyword in player_death_keywords:
                if keyword in text:
                    found_keywords.add(keyword)
            if player_death_keywords[0] in text:
                revive_box = box
        if all(keyword in found_keywords for keyword in player_death_keywords):
            return revive_box
        return None

    def _has_character_death_overlay(self, screen) -> bool:
        crop, _, _ = self._crop_normalized(screen, CHARACTER_DEATH_OVERLAY_REGION)
        if crop.size == 0:
            return False
        blue = crop[:, :, 0].astype(np.int16)
        green = crop[:, :, 1].astype(np.int16)
        red = crop[:, :, 2].astype(np.int16)
        red_dominant = (red > green + CHARACTER_DEATH_OVERLAY_RED_MARGIN) & (
            red > blue + CHARACTER_DEATH_OVERLAY_RED_MARGIN
        )
        return bool(red_dominant.mean() >= CHARACTER_DEATH_OVERLAY_MIN_COVERAGE)

    def _has_character_death_signature(self, screen) -> bool:
        return self._has_character_death_overlay(screen) or (
            self._mask_pixel_count(
                screen,
                CHARACTER_TARGET_REGION,
                CHARACTER_TARGET_HSV_LOWER,
                CHARACTER_TARGET_HSV_UPPER,
            )
            < CHARACTER_DEATH_NO_OVERLAY_MAX_TARGET_PIXELS
        )

    def character_death_state(self, screen):
        if screen is None or screen.size == 0:
            return None
        if (
            self._character_exit_box(screen) is None
            or not self._has_character_ring(screen)
            or not self._has_character_death_signature(screen)
        ):
            return None
        if self._mask_pixel_count(
            screen,
            CHARACTER_TARGET_REGION,
            CHARACTER_TARGET_HSV_LOWER,
            CHARACTER_TARGET_HSV_UPPER,
        ) > CHARACTER_TARGET_MIN_PIXELS and (
            self._character_revive_target_disk(screen)[0] is not None
            or self._mask_pixel_count(
                screen,
                CHARACTER_TARGET_REGION,
                CHARACTER_TARGET_HSV_LOWER,
                CHARACTER_TARGET_HSV_UPPER,
            )
            > CHARACTER_REVIVING_STRONG_TARGET_PIXELS
        ):
            return CHARACTER_DEATH_REVIVING
        return CHARACTER_DEATH_WAITING

    def is_character_death(self, screen=None) -> bool:
        if screen is None:
            screen = get_window_screen()
        return self.character_death_state(screen) is not None

    def _character_exit_box(self, screen):
        crop, width, height = self._crop_normalized(screen, CHARACTER_EXIT_REGION)
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        crop_x = int(CHARACTER_EXIT_REGION[0] * width)
        crop_y = int(CHARACTER_EXIT_REGION[2] * height)
        # 只用高亮度白偵測退出按鈕。fallback(灰白)會誤判城鎮畫面的
        # 按鍵提示 UI(Shift/Tab 等)為退出按鈕。
        mask = cv2.inRange(hsv, CHARACTER_EXIT_HSV_LOWER, CHARACTER_EXIT_HSV_UPPER)
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        for component_x, component_y, component_width, component_height, _ in stats[
            1:count
        ]:
            if (
                CHARACTER_EXIT_MIN_WIDTH * width
                <= component_width
                <= CHARACTER_EXIT_MAX_WIDTH * width
                and CHARACTER_EXIT_MIN_HEIGHT * height
                <= component_height
                <= CHARACTER_EXIT_MAX_HEIGHT * height
            ):
                return (
                    crop_x + component_x,
                    crop_y + component_y,
                    component_width,
                    component_height,
                )
        # 紅色 overlay 死亡畫面: 退出按鈕被染紅, 白框偵測失效。
        # 以「低飽和亮色 + 近正方形按鈕形狀」偵測(排除扁長 UI 誤判)。
        overlay_mask = cv2.inRange(
            hsv,
            CHARACTER_EXIT_REDOVERLAY_HSV_LOWER,
            CHARACTER_EXIT_REDOVERLAY_HSV_UPPER,
        )
        count, _, stats, _ = cv2.connectedComponentsWithStats(overlay_mask)
        best = None
        best_area = 0
        for (
            component_x,
            component_y,
            component_width,
            component_height,
            component_area,
        ) in stats[1:count]:
            if not (
                CHARACTER_EXIT_MIN_WIDTH * width
                <= component_width
                <= CHARACTER_EXIT_MAX_WIDTH * width
                and CHARACTER_EXIT_MIN_HEIGHT * height
                <= component_height
                <= CHARACTER_EXIT_MAX_HEIGHT * height
            ):
                continue
            aspect = component_width / max(component_height, 1)
            if not (
                CHARACTER_EXIT_REDOVERLAY_MIN_ASPECT
                <= aspect
                <= CHARACTER_EXIT_REDOVERLAY_MAX_ASPECT
            ):
                continue
            # 組件必須有色彩: 灰階按鈕(如寶箱介面)不是死亡畫面的退出按鈕
            component_mask = overlay_mask[
                component_y : component_y + component_height,
                component_x : component_x + component_width,
            ]
            component_hsv = hsv[
                component_y : component_y + component_height,
                component_x : component_x + component_width,
            ]
            if component_mask.sum() > 0:
                mean_sat = float(component_hsv[component_mask > 0, 1].mean())
                if mean_sat < CHARACTER_EXIT_REDOVERLAY_MIN_SATURATION:
                    continue
            if component_area > best_area:
                best_area = component_area
                best = (
                    crop_x + component_x,
                    crop_y + component_y,
                    component_width,
                    component_height,
                )
        return best

    def _has_character_revive_scene(self, screen, exit_box=None) -> bool:
        """Detect the character revive ring scene even without the waiting
        entry, e.g. when the script restarts mid-minigame."""
        if exit_box is None:
            exit_box = self._character_exit_box(screen)
        if exit_box is None:
            return False
        height = screen.shape[0]
        target_radius, target_confidence, _ = self._character_revive_target_candidate(
            screen
        )
        if (
            target_radius is not None
            and target_confidence >= CHARACTER_REVIVE_DIRECT_TARGET_MIN_CONFIDENCE
            and CHARACTER_REVIVE_DIRECT_TARGET_MIN_RADIUS * height
            <= target_radius
            <= CHARACTER_TARGET_MAX_RADIUS * height
        ):
            target_tolerance = max(
                CHARACTER_TARGET_MIN_TOLERANCE,
                target_radius * CHARACTER_TARGET_TOLERANCE_RATIO,
            )
            ring_radius, ring_confidence = self._character_revive_ring_measurement(
                screen,
                time.monotonic(),
                target_radius + target_tolerance,
            )
            return (
                ring_radius is not None
                and ring_confidence >= CHARACTER_REVIVE_SCORE_THRESHOLD
            )
        # Target is not measurable in some ring variants (the yellow center is
        # too faint/dark). Fall back to exit-box + ring presence: the observe
        # loop itself guards clicks with shrink/settled evidence, so starting
        # the session here is safe.
        if not self._has_character_ring(screen):
            return False
        ring_radius, ring_confidence = self._character_revive_ring_measurement(
            screen, time.monotonic(), CHARACTER_REVIVE_RING_MIN_RADIUS
        )
        if ring_radius is None:
            return False
        # 死亡等待畫面的環測量常落在中心偽影(r6 = 測量下限): 那是靜態
        # 角色光效, 不是收縮中的復活圓環。要求測到實質半徑(> 下限 + 餘裕)
        # 才視為復活圓環; 復活圓環收縮中段半徑遠大於此, 不影響。
        if (
            ring_radius
            <= CHARACTER_REVIVE_RING_MIN_RADIUS + CHARACTER_REVIVE_SCENE_MIN_RING_MARGIN
        ):
            return False
        return ring_confidence >= CHARACTER_REVIVE_SCORE_THRESHOLD

    def _has_character_ring(self, screen) -> bool:
        # 復活圓環是「空心環」: 藍色像素集中在圓周，中心區域幾乎無藍色。
        # 城鎮畫面的藍色天空/圓形 UI 會讓中心也大量藍色，故中心空心是
        # 藍色捷徑與 fallback 圓周檢查的共同前置條件。
        center_pixels = self._mask_pixel_count(
            screen,
            CHARACTER_RING_CENTER_REGION,
            CHARACTER_RING_HSV_LOWER,
            CHARACTER_RING_HSV_UPPER,
        )
        ring_pixels = self._mask_pixel_count(
            screen,
            CHARACTER_RING_REGION,
            CHARACTER_RING_HSV_LOWER,
            CHARACTER_RING_HSV_UPPER,
        )
        center_is_hollow = center_pixels <= CHARACTER_RING_CENTER_MAX_PIXELS or (
            ring_pixels > 0
            and center_pixels / ring_pixels <= CHARACTER_RING_CENTER_MAX_RATIO
        )
        if not center_is_hollow:
            return False
        if (
            self._mask_pixel_count(
                screen,
                CHARACTER_RING_REGION,
                CHARACTER_RING_HSV_LOWER,
                CHARACTER_RING_HSV_UPPER,
            )
            >= CHARACTER_RING_MIN_PIXELS
        ):
            return True
        crop, _, height = self._crop_normalized(screen, CHARACTER_RING_REGION)
        if crop.size == 0:
            return False
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        min_radius = int(CHARACTER_RING_FALLBACK_MIN_RADIUS * height)
        max_radius = int(CHARACTER_RING_FALLBACK_MAX_RADIUS * height)
        max_radius = min(
            max_radius,
            (crop.shape[0] - 1) // 2,
            (crop.shape[1] - 1) // 2,
        )
        if min_radius > max_radius:
            return False
        radii = np.arange(min_radius, max_radius + 1)
        angles = np.linspace(
            0, 2 * np.pi, CHARACTER_RING_FALLBACK_SAMPLE_COUNT, endpoint=False
        )
        xs = np.rint(crop.shape[1] / 2 + radii[:, None] * np.cos(angles)).astype(int)
        ys = np.rint(crop.shape[0] / 2 + radii[:, None] * np.sin(angles)).astype(int)
        saturation = hsv[ys, xs, 1]
        value = hsv[ys, xs, 2]
        coverage = (
            (saturation <= CHARACTER_RING_FALLBACK_MAX_SATURATION)
            & (value >= CHARACTER_RING_FALLBACK_MIN_VALUE)
        ).mean(axis=1)
        if np.any(coverage >= CHARACTER_RING_FALLBACK_MIN_ANGULAR_COVERAGE):
            return True
        # 紅色 overlay 死亡畫面: 圓環被紅色特效染紅, 白/低飽和環偵測失效。
        # 用紅色圓周覆蓋偵測。誤判風險由 character_death_state 的
        # exit_box 與死亡特徵(紅色 overlay / 低 target 像素)把關:
        # 非死亡畫面(城鎮/暫停等)通常無退出按鈕, 不會被判定為死亡。
        hue = hsv[ys, xs, 0]
        red_coverage = (
            ((hue < CHARACTER_RING_RED_HUE_MAX) | (hue >= CHARACTER_RING_RED_HUE_MIN))
            & (saturation >= CHARACTER_RING_RED_MIN_SATURATION)
            & (value >= CHARACTER_RING_RED_MIN_VALUE)
        ).mean(axis=1)
        return bool(np.any(red_coverage >= CHARACTER_RING_RED_MIN_ANGULAR_COVERAGE))

    def _mask_pixel_count(self, screen, region, lower, upper) -> int:
        crop, _, _ = self._crop_normalized(screen, region)
        if crop.size == 0:
            return 0
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        return cv2.countNonZero(cv2.inRange(hsv, lower, upper))

    def _crop_normalized(self, screen, region):
        height, width = screen.shape[:2]
        x1f, x2f, y1f, y2f = region
        x1, x2 = int(x1f * width), int(x2f * width)
        y1, y2 = int(y1f * height), int(y2f * height)
        return screen[y1:y2, x1:x2], width, height

    def _ocr_boxes(self, screen, region):
        height, width = screen.shape[:2]
        x1f, x2f, y1f, y2f = region
        x1, x2 = int(x1f * width), int(x2f * width)
        y1, y2 = int(y1f * height), int(y2f * height)
        crop = screen[y1:y2, x1:x2]
        if state.ocr is None or crop.size == 0:
            return []
        return [
            (text, (x1 + bx1, y1 + by1, bx2 - bx1, by2 - by1))
            for text, (bx1, by1, bx2, by2) in parse_ocr_boxes(state.ocr.predict(crop))
        ]

    def revive_charge_status(self, screen) -> tuple[bool, int]:
        height, width = screen.shape[:2]
        x1f, x2f, y1f, y2f = self.revive_charge_region
        x1, x2 = int(x1f * width), int(x2f * width)
        y1, y2 = int(y1f * height), int(y2f * height)
        crop = screen[y1:y2, x1:x2]
        if crop.size == 0:
            return False, 0
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, REVIVE_CHARGE_HSV_LOWER, REVIVE_CHARGE_HSV_UPPER)
        present = cv2.countNonZero(mask) >= REVIVE_CHARGE_PRESENT_MIN_PIXELS
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        charges = sum(
            REVIVE_CHARGE_MIN_WIDTH * width
            <= component_width
            <= REVIVE_CHARGE_MAX_WIDTH * width
            and REVIVE_CHARGE_MIN_HEIGHT * height
            <= component_height
            <= REVIVE_CHARGE_MAX_HEIGHT * height
            and REVIVE_CHARGE_MIN_AREA * width * height
            <= area
            <= REVIVE_CHARGE_MAX_AREA * width * height
            for _, _, component_width, component_height, area in stats[1:count]
        )
        return present, charges

    def revive_charge_count(self, screen) -> int:
        """Return the count while preserving the count-only API."""
        _, charges = self.revive_charge_status(screen)
        return charges
