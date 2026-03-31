# eye_analyzer.py
# 눈 열린 정도(EAR)를 계산하고, 눈이 얼마나 오래 감겼는지 측정합니다.
#
# EAR(Eye Aspect Ratio) 계산 방법:
#         p2  p3
#    p1           p4
#         p6  p5
#
#  EAR = ( ||p2-p6|| + ||p3-p5|| ) / ( 2 * ||p1-p4|| )
#  → 눈이 떠 있으면 세로 거리가 크고, 감으면 작아짐
#
# MediaPipe 랜드마크 인덱스:
#   오른쪽 눈: [33, 160, 158, 133, 153, 144]
#   왼쪽  눈: [263, 387, 385, 362, 380, 373]

import time
from collections import deque

import numpy as np

from dms.utils.config import (
    EAR_CLOSE_THRESHOLD,
    PERCLOS_WINDOW_SEC,
    PERCLOS_WARN_RATIO,
    PERCLOS_ALERT_RATIO,
)

# 눈 랜드마크 인덱스 (MediaPipe FaceMesh 기준)
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]   # 오른쪽 눈 6개 포인트
LEFT_EYE_IDX  = [263, 387, 385, 362, 380, 373]   # 왼쪽 눈 6개 포인트


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    # 두 점 사이의 거리 계산
    return float(np.linalg.norm(a - b))


def calc_ear(landmarks: np.ndarray, idx: list[int]) -> float:
    """6개 랜드마크 인덱스로 EAR(눈 열린 정도)를 계산합니다."""
    p = landmarks[idx]   # 6개 포인트 추출, shape (6, 2)
    vertical   = _dist(p[1], p[5]) + _dist(p[2], p[4])   # 눈 세로 길이 합
    horizontal = _dist(p[0], p[3])                         # 눈 가로 길이
    if horizontal < 1e-6:
        return 0.0
    return vertical / (2.0 * horizontal)


class EyeAnalyzer:
    """
    매 프레임 update()를 호출하면 EAR, 눈 감김 지속시간, PERCLOS를 계산합니다.
    """

    def __init__(self):
        # PERCLOS 계산용 슬라이딩 윈도우: (시간, 눈감김여부) 쌍을 저장
        self._window: deque = deque()
        self._closed_since: float | None = None   # 눈 감기 시작 시간

        # 최신 계산 결과
        self.ear: float           = 0.0
        self.left_ear: float      = 0.0
        self.right_ear: float     = 0.0
        self.eyes_closed: bool    = False
        self.closed_duration: float = 0.0   # 눈 감긴 지속 시간 (초)
        self.perclos: float       = 0.0     # 최근 5초 동안 눈 감긴 비율

    def update(self, landmarks: np.ndarray, now: float | None = None) -> dict:
        """랜드마크를 받아 눈 상태를 분석하고 결과를 딕셔너리로 반환합니다.

        now: 영상 파일일 때 cap.get(CAP_PROP_POS_MSEC)/1000.0 을 전달.
             None이면 벽시계(perf_counter) 사용 (카메라 모드).
        """
        if now is None:
            now = time.perf_counter()

        # 양쪽 눈 EAR 계산 후 평균
        self.right_ear = calc_ear(landmarks, RIGHT_EYE_IDX)
        self.left_ear  = calc_ear(landmarks, LEFT_EYE_IDX)
        self.ear       = (self.right_ear + self.left_ear) / 2.0

        # 기준값보다 작으면 눈이 감긴 것으로 판정
        self.eyes_closed = self.ear < EAR_CLOSE_THRESHOLD

        # 눈이 얼마나 오래 감겼는지 측정
        if self.eyes_closed:
            if self._closed_since is None:
                self._closed_since = now   # 처음 감긴 시점 기록
            self.closed_duration = now - self._closed_since
        else:
            self._closed_since   = None   # 눈 뜨면 초기화
            self.closed_duration = 0.0

        # PERCLOS 계산: 최근 N초 동안 눈이 감긴 프레임 비율
        self._window.append((now, self.eyes_closed))
        cutoff = now - PERCLOS_WINDOW_SEC
        # 오래된 데이터 제거
        while self._window and self._window[0][0] < cutoff:
            self._window.popleft()

        if self._window:
            closed_cnt = sum(1 for _, c in self._window if c)
            self.perclos = closed_cnt / len(self._window)
        else:
            self.perclos = 0.0

        return {
            "ear":             self.ear,
            "left_ear":        self.left_ear,
            "right_ear":       self.right_ear,
            "eyes_closed":     self.eyes_closed,
            "closed_duration": self.closed_duration,
            "perclos":         self.perclos,
        }

    def reset(self):
        """상태를 초기화합니다."""
        self._window.clear()
        self._closed_since   = None
        self.closed_duration = 0.0
        self.perclos         = 0.0
