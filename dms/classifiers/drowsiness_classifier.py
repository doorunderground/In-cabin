# drowsiness_classifier.py
# 눈/하품 데이터를 보고 졸음 레벨을 판정합니다.
#
# 판정 레벨:
#   NORMAL   (0) : 정상
#   EARLY    (1) : 초기 졸음 징후
#   WARNING  (2) : 졸음 경고
#   CRITICAL (3) : 즉각 대응 필요

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


# 레벨별 영문/한글 라벨
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
    규칙 기반으로 졸음 레벨을 판정합니다.

    판정 우선순위 (높은 쪽 우선):
    1. 눈 감김 지속 시간
    2. PERCLOS 비율
    3. 하품 횟수
    """

    def __init__(self):
        self.level: DrowsinessLevel = DrowsinessLevel.NORMAL
        self._last_alert_at: float  = 0.0

    def update(self, eye_data: dict, mouth_data: dict,
               now: float | None = None) -> dict:
        """눈/하품 데이터를 받아 졸음 레벨을 계산하고 결과를 반환합니다.

        now: 영상 파일일 때 cap.get(CAP_PROP_POS_MSEC)/1000.0 을 전달.
             None이면 벽시계(perf_counter) 사용 (카메라 모드).
        """
        dur      = eye_data["closed_duration"]   # 눈 감긴 지속 시간
        perclos  = eye_data["perclos"]            # 최근 5초 눈 감긴 비율
        yawns    = mouth_data["yawn_count"]       # 누적 하품 횟수
        yawn_now = mouth_data["yawn_detected"]    # 방금 하품 감지됐는지

        # 레벨 판정 (높은 것이 우선)
        if dur >= DROWSY_CRITICAL_SEC or perclos >= PERCLOS_ALERT_RATIO:
            level = DrowsinessLevel.CRITICAL

        elif dur >= DROWSY_WARNING_SEC or perclos >= PERCLOS_WARN_RATIO:
            level = DrowsinessLevel.WARNING

        elif dur >= DROWSY_EARLY_SEC or yawns >= 2:
            level = DrowsinessLevel.EARLY

        else:
            level = DrowsinessLevel.NORMAL

        # 방금 하품이 감지됐으면 최소 EARLY 이상으로 보장
        if yawn_now and level < DrowsinessLevel.EARLY:
            level = DrowsinessLevel.EARLY

        self.level = level

        # 경보 트리거 (쿨다운 적용)
        if now is None:
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
        """상태를 초기화합니다."""
        self.level          = DrowsinessLevel.NORMAL
        self._last_alert_at = 0.0
