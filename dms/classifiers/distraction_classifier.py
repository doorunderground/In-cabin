"""
distraction_classifier.py
Head Pose 데이터를 기반으로 시선 분산 레벨 판정.

레벨 정의
─────────
  NORMAL     (0) : 전방 주시 정상
  GLANCE     (1) : 잠깐 시선 이탈 (< 2 초)
  DISTRACTED (2) : 시선 분산 경고  (2 ~ 5 초)
  CRITICAL   (3) : 즉각 조치 필요  (> 5 초 또는 극단 각도)
"""

import time
from enum import IntEnum

from dms.utils.config import (
    YAW_CRITICAL_MAX,
    PITCH_CRITICAL_MAX,
    HEAD_DISTRACTION_MIN_DURATION_SEC,
    HEAD_DISTRACTION_ALERT_RATIO,
    ALERT_COOLDOWN_SEC,
)

DISTRACTION_CRITICAL_SEC = HEAD_DISTRACTION_MIN_DURATION_SEC + 3.0   # 5 초


class DistractionLevel(IntEnum):
    NORMAL     = 0
    GLANCE     = 1
    DISTRACTED = 2
    CRITICAL   = 3


LEVEL_LABELS = {
    DistractionLevel.NORMAL:     "NORMAL",
    DistractionLevel.GLANCE:     "GLANCE AWAY",
    DistractionLevel.DISTRACTED: "DISTRACTED",
    DistractionLevel.CRITICAL:   "CRITICAL",
}

LEVEL_KO = {
    DistractionLevel.NORMAL:     "전방 주시",
    DistractionLevel.GLANCE:     "잠깐 시선 이탈",
    DistractionLevel.DISTRACTED: "시선 분산 경고",
    DistractionLevel.CRITICAL:   "위험 - 전방 주시",
}


class DistractionClassifier:
    """
    규칙 기반 시선 분산 상태 머신.

    판정 우선순위
    ─────────────
    1. 극단 각도 (|Yaw| > 60° or |Pitch| > 60°) → CRITICAL
    2. 분산 지속시간 > 5초 → CRITICAL
    3. 슬라이딩 윈도우 비율 ≥ 60% → CRITICAL
    4. 분산 지속시간 2 ~ 5초 → DISTRACTED
    5. 현재 분산 중 → GLANCE
    """

    def __init__(self):
        self.level: DistractionLevel = DistractionLevel.NORMAL
        self._last_alert_at: float   = 0.0

    def update(self, head_data: dict) -> dict:
        dur   = head_data["distraction_duration"]
        ratio = head_data["distraction_ratio"]
        yaw   = abs(head_data["yaw"])
        pitch = abs(head_data["pitch"])
        is_d  = head_data["is_distracted"]

        # ── 레벨 산정 ──────────────────────────────────
        if yaw > YAW_CRITICAL_MAX or pitch > PITCH_CRITICAL_MAX:
            level = DistractionLevel.CRITICAL
        elif dur >= DISTRACTION_CRITICAL_SEC:
            level = DistractionLevel.CRITICAL
        elif ratio >= HEAD_DISTRACTION_ALERT_RATIO:
            level = DistractionLevel.CRITICAL
        elif dur >= HEAD_DISTRACTION_MIN_DURATION_SEC:
            level = DistractionLevel.DISTRACTED
        elif is_d:
            level = DistractionLevel.GLANCE
        else:
            level = DistractionLevel.NORMAL

        self.level = level

        # ── 경보 트리거 ────────────────────────────────
        now = time.perf_counter()
        should_alert = (
            level >= DistractionLevel.DISTRACTED
            and (now - self._last_alert_at) > ALERT_COOLDOWN_SEC
        )
        if should_alert:
            self._last_alert_at = now

        return {
            "level":        level,
            "label":        LEVEL_LABELS[level],
            "label_ko":     LEVEL_KO[level],
            "should_alert": should_alert,
        }

    def reset(self):
        self.level          = DistractionLevel.NORMAL
        self._last_alert_at = 0.0
