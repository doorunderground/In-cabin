"""
drowsiness_classifier.py
EAR/PERCLOS/하품 데이터를 조합해 졸음 레벨을 판정하는 상태 머신.

판정 레벨
─────────
  NORMAL   (0) : 정상
  EARLY    (1) : 초기 졸음 징후 (주의)
  WARNING  (2) : 졸음 경고
  CRITICAL (3) : 즉각 대응 필요
"""

import time
from enum import IntEnum

from dms.utils.config import (
    DROWSY_EARLY_SEC,
    DROWSY_WARNING_SEC,
    DROWSY_CRITICAL_SEC,
    PERCLOS_WARN_RATIO,
    PERCLOS_ALERT_RATIO,
    ALERT_COOLDOWN_SEC,
)


class DrowsinessLevel(IntEnum):
    NORMAL   = 0
    EARLY    = 1
    WARNING  = 2
    CRITICAL = 3


LEVEL_LABELS = {
    DrowsinessLevel.NORMAL:   "NORMAL",
    DrowsinessLevel.EARLY:    "EARLY DROWSINESS",
    DrowsinessLevel.WARNING:  "DROWSY - WARNING",
    DrowsinessLevel.CRITICAL: "DROWSY - CRITICAL",
}

LEVEL_KO = {
    DrowsinessLevel.NORMAL:   "정상",
    DrowsinessLevel.EARLY:    "졸음 초기",
    DrowsinessLevel.WARNING:  "졸음 경고",
    DrowsinessLevel.CRITICAL: "위험 - 즉시 조치",
}


class DrowsinessClassifier:
    """
    규칙 기반 졸음 상태 머신.

    판정 우선순위 (높은 쪽 우선)
    ─────────────────────────────
    1. 눈 감김 지속시간 (closed_duration)
    2. PERCLOS 비율
    3. 하품 횟수
    """

    def __init__(self):
        self.level: DrowsinessLevel = DrowsinessLevel.NORMAL
        self._last_alert_at: float = 0.0

    # ── 메인 업데이트 ─────────────────────────────────
    def update(self, eye_data: dict, mouth_data: dict) -> dict:
        dur     = eye_data["closed_duration"]
        perclos = eye_data["perclos"]
        yawns   = mouth_data["yawn_count"]
        yawn_now = mouth_data["yawn_detected"]

        # ── 레벨 산정 ──────────────────────────────────
        if dur >= DROWSY_CRITICAL_SEC or perclos >= PERCLOS_ALERT_RATIO:
            level = DrowsinessLevel.CRITICAL

        elif dur >= DROWSY_WARNING_SEC or perclos >= PERCLOS_WARN_RATIO:
            level = DrowsinessLevel.WARNING

        elif dur >= DROWSY_EARLY_SEC or yawns >= 2:
            level = DrowsinessLevel.EARLY

        else:
            level = DrowsinessLevel.NORMAL

        # 하품 감지 시 최소 EARLY 보장
        if yawn_now and level < DrowsinessLevel.EARLY:
            level = DrowsinessLevel.EARLY

        self.level = level

        # ── 경보 트리거 판정 ───────────────────────────
        now = time.perf_counter()
        should_alert = (
            level >= DrowsinessLevel.WARNING
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
        self.level         = DrowsinessLevel.NORMAL
        self._last_alert_at = 0.0
