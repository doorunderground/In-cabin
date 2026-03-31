# distraction_classifier.py
# 시선 데이터를 보고 운전자가 도로에서 얼마나 시선이 분산됐는지 판정합니다.
#
# 판정 레벨:
#   NORMAL     (0) : 전방 주시 정상
#   GLANCE     (1) : 잠깐 시선 이탈 (2초 미만)
#   DISTRACTED (2) : 시선 분산 경고  (2~5초)
#   CRITICAL   (3) : 즉각 조치 필요  (5초 이상 또는 극단 각도)

import time
from enum import IntEnum

from dms.utils.config import (
    YAW_CRITICAL_MAX,
    PITCH_CRITICAL_MAX,
    HEAD_DISTRACTION_MIN_DURATION_SEC,
    ALERT_COOLDOWN_SEC,
)

# CRITICAL 판정 기준 시간 (2초 + 3초 = 5초)
DISTRACTION_CRITICAL_SEC = HEAD_DISTRACTION_MIN_DURATION_SEC + 3.0


class DistractionLevel(IntEnum):
    NORMAL     = 0
    GLANCE     = 1
    DISTRACTED = 2
    CRITICAL   = 3


# 레벨별 영문/한글 라벨
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
    규칙 기반으로 시선 분산 레벨을 판정합니다.

    판정 우선순위:
    1. 극단 각도 (|yaw| > 60° 또는 |pitch| > 55°) → CRITICAL
    2. 5초 이상 시선 이탈 → CRITICAL
    3. 2~5초 시선 이탈 → DISTRACTED
    4. 현재 시선이 도로 아님 → GLANCE
    5. 도로 주시 → NORMAL
    """

    def __init__(self):
        self.level: DistractionLevel    = DistractionLevel.NORMAL
        self._last_alert_at:  float     = 0.0
        self._off_road_since: float | None = None   # 도로 이탈 시작 시간

    def update(self, head_data: dict, now: float | None = None) -> dict:
        """head_data를 받아 시선 분산 레벨을 계산하고 결과를 반환합니다.

        now: 영상 파일일 때 cap.get(CAP_PROP_POS_MSEC)/1000.0 을 전달.
             None이면 벽시계(perf_counter) 사용 (카메라 모드).
        """
        zone = head_data.get("gaze_zone", "ROAD")

        # 캘리브레이션 완료 시 보정 각도, 미완료 시 원시 각도 사용
        yaw_abs   = abs(head_data.get("yaw_corr",   head_data.get("yaw",   0.0)))
        pitch_abs = abs(head_data.get("pitch_corr", head_data.get("pitch", 0.0)))

        is_off_road = (zone != "ROAD")
        if now is None:
            now = time.perf_counter()

        # 도로 이탈 연속 시간 계산
        if is_off_road:
            if self._off_road_since is None:
                self._off_road_since = now   # 이탈 시작 시점 기록
            off_dur = now - self._off_road_since
        else:
            self._off_road_since = None
            off_dur = 0.0

        # 레벨 판정
        if yaw_abs > YAW_CRITICAL_MAX or pitch_abs > PITCH_CRITICAL_MAX:
            level = DistractionLevel.CRITICAL   # 극단 각도는 즉시 위험
        elif off_dur >= DISTRACTION_CRITICAL_SEC:
            level = DistractionLevel.CRITICAL   # 5초 이상 이탈
        elif off_dur >= HEAD_DISTRACTION_MIN_DURATION_SEC:
            level = DistractionLevel.DISTRACTED  # 2~5초 이탈
        elif is_off_road:
            level = DistractionLevel.GLANCE      # 잠깐 이탈
        else:
            level = DistractionLevel.NORMAL      # 정상

        self.level = level

        # 경보 트리거 (같은 경보가 너무 자주 울리지 않도록 쿨다운 적용)
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
        """상태를 초기화합니다."""
        self.level           = DistractionLevel.NORMAL
        self._last_alert_at  = 0.0
        self._off_road_since = None
