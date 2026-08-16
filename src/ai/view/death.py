import time
from collections import deque

import cv2
import numpy as np

from ... import AI
from ...ocr import parse_ocr_boxes
from ...utils.clicker import calculate_click_point, click_at
from ...utils.shared import state
from ...utils.text_map import get_text_mapping
from ...utils.window import dump4log, get_window_rect, get_window_screen

PLAYER_DEATH_REGION = (0.30, 0.70, 0.40, 0.65)
REVIVE_CHARGE_REGION = (0.35, 0.65, 0.00, 0.15)
REVIVE_CHARGE_HSV_LOWER = (10, 80, 120)
REVIVE_CHARGE_HSV_UPPER = (45, 255, 255)
REVIVE_CHARGE_MIN_WIDTH = 0.02
REVIVE_CHARGE_MAX_WIDTH = 0.04
REVIVE_CHARGE_MIN_HEIGHT = 0.03
REVIVE_CHARGE_MAX_HEIGHT = 0.07
REVIVE_CHARGE_MIN_AREA = 0.0004
REVIVE_CHARGE_MAX_AREA = 0.0012
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
CHARACTER_TARGET_REGION = (0.40, 0.60, 0.30, 0.70)
CHARACTER_DEATH_OVERLAY_REGION = (0.20, 0.80, 0.20, 0.80)
CHARACTER_DEATH_OVERLAY_RED_MARGIN = 15
CHARACTER_DEATH_OVERLAY_MIN_COVERAGE = 0.90
CHARACTER_EXIT_HSV_LOWER = (0, 0, 180)
CHARACTER_EXIT_HSV_UPPER = (180, 70, 255)
CHARACTER_EXIT_FALLBACK_HSV_LOWER = (0, 0, 100)
CHARACTER_EXIT_FALLBACK_HSV_UPPER = (180, 180, 255)
CHARACTER_RING_HSV_LOWER = (90, 80, 80)
CHARACTER_RING_HSV_UPPER = (130, 255, 255)
CHARACTER_RING_FALLBACK_MAX_SATURATION = 140
CHARACTER_RING_FALLBACK_MIN_VALUE = 100
CHARACTER_RING_FALLBACK_MIN_RADIUS = 0.08
CHARACTER_RING_FALLBACK_MAX_RADIUS = 0.30
CHARACTER_RING_FALLBACK_MIN_ANGULAR_COVERAGE = 0.20
CHARACTER_RING_FALLBACK_SAMPLE_COUNT = 180
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
CHARACTER_DEATH_NO_OVERLAY_MAX_TARGET_PIXELS = 500
CHARACTER_TARGET_MIN_PIXELS = 1000
CHARACTER_REVIVE_TARGET_RADIUS = 0.06
CHARACTER_REVIVE_TARGET_TOLERANCE = 0.015
CHARACTER_REVIVE_MIN_RADIUS = (
    CHARACTER_REVIVE_TARGET_RADIUS + CHARACTER_REVIVE_TARGET_TOLERANCE
)
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
CHARACTER_REVIVE_SCORE_THRESHOLD = 0.35
CHARACTER_REVIVE_FALLBACK_MAX_SATURATION = 140
CHARACTER_REVIVE_FALLBACK_MIN_VALUE = 100
CHARACTER_REVIVE_FALLBACK_SCORE_THRESHOLD = 0.35
CHARACTER_REVIVE_HISTORY_SIZE = 4
CHARACTER_REVIVE_MIN_MEASURED_FRAMES = 3
CHARACTER_REVIVE_MIN_MEASURED_CONFIDENCE = 0.5
CHARACTER_REVIVE_LOG_INTERVAL = 0.1
CHARACTER_REVIVE_DETECTION_GRACE = 0.15
CHARACTER_REVIVE_MAX_PREDICTION_GAP = 0.5
CHARACTER_REVIVE_MAX_SHRINK_SPEED = 2200.0
CHARACTER_REVIVE_MIN_SHRINK_SPEED = 25.0
CHARACTER_REVIVE_OUTER_RESET_RATIO = 0.85
CHARACTER_REVIVE_BOUNDARY_BAND_GROWTH_LIMIT = 0.08
CHARACTER_REVIVE_BOUNDARY_RESET_FRAMES = 3
CHARACTER_REVIVE_CLICK_COOLDOWN = 1.0
CHARACTER_REVIVE_RETRY_DELAY = 0.75
CHARACTER_REVIVE_POST_CLICK_GRACE = 3.0
CHARACTER_REVIVE_TIMEOUT_COOLDOWN = 2.0
CHARACTER_REVIVE_SESSION_DEADLINE = 30.0
CHARACTER_REVIVE_SAMPLE_COUNT = 360
CHARACTER_DEATH_POLL_INTERVAL = 0.25


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
        self._last_character_revive_log = 0.0
        self._character_revive_band_rejections = 0
        self._character_revive_cycle_reached_target = False
        self._character_revive_retry_at = None
        self._character_revive_post_click_until = None
        self._character_revive_timeout_until = None

    def check(self) -> bool:
        now = time.monotonic()
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
        screen = get_window_screen()
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
                    self._character_revive_retry_at = (
                        now + CHARACTER_REVIVE_RETRY_DELAY
                    )
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
        revive_box = self._find_player_death_revive_box(screen)
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
                click_at(calculate_click_point(revive_box[:2], revive_box[2:]))
                state.logger.info("已使用復活之火，等待 %.1f 秒", self.delay_revive)
                time.sleep(self.delay_revive)
            return True
        return False

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
        self._last_character_revive_log = 0.0
        self._character_revive_band_rejections = 0
        self._character_revive_cycle_reached_target = False
        self._character_revive_retry_at = None
        self._character_revive_post_click_until = None
        self._character_revive_timeout_until = None

    def _has_active_character_revive_session(self, screen, now, exit_box=None) -> bool:
        if exit_box is None:
            exit_box = self._character_exit_box(screen)
        if self._character_revive_session_mode == CHARACTER_REVIVE_SESSION_NORMAL:
            # Bootstrap uses strict target-plus-ring evidence; an existing
            # session keeps the looser ring signal to survive animation frames
            # where the target mask is temporarily invisible.
            present = exit_box is not None and self._has_character_ring(screen)
        elif self._character_revive_session_mode == CHARACTER_REVIVE_SESSION_OVERLAY:
            present = (
                exit_box is not None and self._has_character_death_signature(screen)
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
                try:
                    dump4log(screen, "角色復活圓環逾時")
                except Exception:
                    pass
            self._character_revive_session_expired = True
            self._character_revive_activated = False
            self._character_revive_session_mode = None
            self._character_revive_timeout_until = (
                now + CHARACTER_REVIVE_TIMEOUT_COOLDOWN
            )
            return None
        observation = self.character_revive_observation(screen)
        if observation is None:
            if now - self._last_character_revive_log >= CHARACTER_REVIVE_LOG_INTERVAL:
                state.logger.debug("角色復活圓環未偵測到有效觀測")
                self._last_character_revive_log = now
            return None
        should_click = observation["click_ready"] and observation["in_target_window"]
        if should_click and self._character_revive_click_allowed(now):
            self._click_character_revive(now, observation)
        if now - self._last_character_revive_log >= CHARACTER_REVIVE_LOG_INTERVAL:
            state.logger.debug(
                "角色復活圓環 radius=%.1f target=%.1f target_measured=%s "
                "target_currently_measured=%s target_mode=%s target_confidence=%.2f "
                "target_tolerance=%.1f confidence=%.2f velocity=%s eta=%s measured=%s",
                observation["radius"],
                observation["target_radius"],
                observation["target_measured"],
                observation["target_currently_measured"],
                observation["target_detection_mode"],
                observation["target_confidence"],
                observation["target_tolerance"],
                observation["confidence"],
                self._format_measurement(observation["velocity"]),
                self._format_measurement(observation["eta"]),
                observation["measured"],
            )
            self._last_character_revive_log = now
        return observation

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
            "target_tolerance=%.1f confidence=%.2f velocity=%s eta=%s measured=%s "
            "measured_frames=%s in_target_window=%s click_margin=%.2f",
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
            self._format_measurement(observation.get("eta")),
            observation.get("measured", False),
            observation.get("measured_frames", 0),
            observation.get("in_target_window", False),
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
        min_radius = max(
            CHARACTER_REVIVE_RING_MIN_RADIUS,
            target_radius + tolerance,
        )
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
            elif not self._character_revive_measurement_is_consistent(
                radius, now, height, target_radius
            ):
                radius = None
                measured = False
            else:
                self._character_revive_history.append((now, radius, confidence))
        if not measured:
            prediction = self._character_revive_prediction(now, height)
            if prediction is None:
                return None
            radius, velocity, confidence = prediction
        else:
            velocity = self._character_revive_velocity()
        if target_measured and radius <= (
            target_radius + CHARACTER_REVIVE_CLICK_RADIUS_MARGIN
        ):
            self._character_revive_cycle_reached_target = True
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
        eta = None
        if velocity is not None and velocity < 0 and radius > target_radius:
            eta = (radius - target_radius) / -velocity
        in_target_window = (
            target_measured
            and radius <= target_radius + CHARACTER_REVIVE_CLICK_RADIUS_MARGIN
            and has_evidence
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
            "eta": eta,
            "in_target_window": in_target_window,
            "measured": measured,
            "measured_frames": measured_frames,
            "click_ready": has_evidence and target_measured,
        }

    def _character_revive_target_measurement(self, screen, now):
        radius, confidence, detection_mode = self._character_revive_target_candidate(
            screen
        )
        height = screen.shape[0]
        if (
            radius is not None
            and confidence >= CHARACTER_TARGET_MIN_CONFIDENCE
            and CHARACTER_REVIVE_DIRECT_TARGET_MIN_RADIUS * height
            <= radius
            <= CHARACTER_TARGET_MAX_RADIUS * height
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
            detection_mode == "strong"
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
            if history[-1][3] == "strong" or weak_confirmed or neutral_confirmed:
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
        if mode == "neutral":
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
        """Prefer a strong target; use constrained faint-target geometry as fallback."""
        height, width = screen.shape[:2]
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

    def _character_revive_measurement_is_consistent(
        self, radius, now, height, target_radius
    ) -> bool:
        history = self._character_revive_history
        if not history:
            return True
        previous_time, previous_radius, _ = history[-1]
        gap = max(now - previous_time, 0.001)
        shrink = previous_radius - radius
        if shrink >= 0:
            self._character_revive_band_rejections = 0
            return shrink <= CHARACTER_REVIVE_MAX_SHRINK_SPEED * gap
        if (
            not self._character_revive_cycle_reached_target
            and previous_radius
            <= target_radius + CHARACTER_REVIVE_CLICK_RADIUS_MARGIN
        ):
            self._character_revive_cycle_reached_target = True
        if not self._character_revive_cycle_reached_target:
            # The ring only resets after reaching the yellow center. Any
            # upward jump before that point is a false outer-ring candidate.
            self._character_revive_band_rejections = 0
            return False
        band_min = (
            CHARACTER_REVIVE_MAX_RADIUS * height * CHARACTER_REVIVE_OUTER_RESET_RATIO
        )
        if radius >= band_min:
            if previous_radius >= band_min and (
                radius - previous_radius
            ) <= CHARACTER_REVIVE_BOUNDARY_BAND_GROWTH_LIMIT * height:
                self._character_revive_band_rejections += 1
                if (
                    self._character_revive_band_rejections
                    >= CHARACTER_REVIVE_BOUNDARY_RESET_FRAMES
                ):
                    history.clear()
                    self._character_revive_band_rejections = 0
                    self._character_revive_cycle_reached_target = False
                    return True
                return False
            history.clear()
            self._character_revive_cycle_reached_target = False
            return True
        self._character_revive_band_rejections = 0
        return False

    def _character_revive_ring_measurement(self, screen, now, min_radius):
        height, width = screen.shape[:2]
        center_x, center_y = width // 2, height // 2
        max_radius = min(int(CHARACTER_REVIVE_MAX_RADIUS * height), width // 2)
        min_radius = max(
            int(np.floor(min_radius)), int(CHARACTER_REVIVE_RING_MIN_RADIUS)
        )
        if min_radius > max_radius:
            return None, 0.0
        radii = np.arange(min_radius, max_radius + 1)
        angles = np.linspace(
            0, 2 * np.pi, CHARACTER_REVIVE_SAMPLE_COUNT, endpoint=False
        )
        xs = np.rint(center_x + radii[:, None] * np.cos(angles)).astype(int)
        ys = np.rint(center_y + radii[:, None] * np.sin(angles)).astype(int)
        valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
        saturation = hsv[ys.clip(0, height - 1), xs.clip(0, width - 1), 1]
        value = hsv[ys.clip(0, height - 1), xs.clip(0, width - 1), 2]
        bright_white = valid & (saturation <= 70) & (value >= 180)
        scores = bright_white.mean(axis=1)
        candidates = np.flatnonzero(scores >= CHARACTER_REVIVE_SCORE_THRESHOLD)
        if candidates.size == 0:
            red_overlay = (
                valid
                & (saturation <= CHARACTER_REVIVE_FALLBACK_MAX_SATURATION)
                & (value >= CHARACTER_REVIVE_FALLBACK_MIN_VALUE)
            )
            scores = red_overlay.mean(axis=1)
            candidates = np.flatnonzero(
                scores >= CHARACTER_REVIVE_FALLBACK_SCORE_THRESHOLD
            )
        if candidates.size == 0:
            return None, float(scores.max())
        if self._character_revive_history:
            previous_time, previous_radius, _ = self._character_revive_history[-1]
            gap = max(now - previous_time, 0.001)
            max_shrink = CHARACTER_REVIVE_MAX_SHRINK_SPEED * gap
            plausible = candidates[
                (radii[candidates] <= previous_radius)
                & (radii[candidates] >= previous_radius - max_shrink)
            ]
            if plausible.size:
                index = int(plausible[np.argmax(scores[plausible])])
                return float(radii[index]), float(scores[index])
            outer = candidates[
                radii[candidates]
                >= CHARACTER_REVIVE_MAX_RADIUS
                * height
                * CHARACTER_REVIVE_OUTER_RESET_RATIO
            ]
            if outer.size:
                index = int(outer[np.argmin(np.abs(radii[outer] - previous_radius))])
                return float(radii[index]), float(scores[index])
            return None, float(scores[candidates].max())
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
        if (
            self._mask_pixel_count(
                screen,
                CHARACTER_TARGET_REGION,
                CHARACTER_TARGET_HSV_LOWER,
                CHARACTER_TARGET_HSV_UPPER,
            )
            > CHARACTER_TARGET_MIN_PIXELS
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
        for lower, upper in (
            (CHARACTER_EXIT_HSV_LOWER, CHARACTER_EXIT_HSV_UPPER),
            (CHARACTER_EXIT_FALLBACK_HSV_LOWER, CHARACTER_EXIT_FALLBACK_HSV_UPPER),
        ):
            mask = cv2.inRange(hsv, lower, upper)
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
        return None

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
            target_radius is None
            or target_confidence < CHARACTER_REVIVE_DIRECT_TARGET_MIN_CONFIDENCE
            or not (
                CHARACTER_REVIVE_DIRECT_TARGET_MIN_RADIUS * height
                <= target_radius
                <= CHARACTER_TARGET_MAX_RADIUS * height
            )
        ):
            return False
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

    def _has_character_ring(self, screen) -> bool:
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
        return bool(np.any(coverage >= CHARACTER_RING_FALLBACK_MIN_ANGULAR_COVERAGE))

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
