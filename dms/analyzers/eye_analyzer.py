"""
eye_analyzer.py
EAR(Eye Aspect Ratio) 계산 및 PERCLOS 누적.

MediaPipe FaceMesh 6-point EAR 인덱스
──────────────────────────────────────
         p2  p3
    p1           p4
         p6  p5

  오른쪽 눈 (카메라 기준): [33, 160, 158, 133, 153, 144]
  왼쪽  눈 (카메라 기준): [263, 387, 385, 362, 380, 373]

EAR = ( ||p2-p6|| + ||p3-p5|| ) / ( 2 * ||p1-p4|| )
"""

import time
from collections import deque

import numpy as np

from dms.utils.config import (
    EAR_CLOSE_THRESHOLD,
    PERCLOS_WINDOW_SEC,
    PERCLOS_WARN_RATIO,
    PERCLOS_ALERT_RATIO,
)

# ── 랜드마크 인덱스 ────────────────────────────────────
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]
LEFT_EYE_IDX  = [263, 387, 385, 362, 380, 373]


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def calc_ear(landmarks: np.ndarray, idx: list[int]) -> float:
    """6개 랜드마크 인덱스로 EAR 계산."""
    p = landmarks[idx]          # shape (6, 2)
    vertical = _dist(p[1], p[5]) + _dist(p[2], p[4])
    horizontal = _dist(p[0], p[3])
    if horizontal < 1e-6:
        return 0.0
    return vertical / (2.0 * horizontal)


class EyeAnalyzer:
    """
    매 프레임 update() 호출 → EAR, 눈 감김 지속시간, PERCLOS 반환.
    """

    def __init__(self):
        # 슬라이딩 윈도우: deque of (timestamp, is_closed)
        self._window: deque = deque()
        self._closed_since: float | None = None

        # 최신 상태 (외부 참조용)
        self.ear: float = 0.0
        self.left_ear: float = 0.0
        self.right_ear: float = 0.0
        self.eyes_closed: bool = False
        self.closed_duration: float = 0.0
        self.perclos: float = 0.0

    def update(self, landmarks: np.ndarray) -> dict:
        now = time.perf_counter()

        # ── EAR 계산 ──────────────────────────────────
        self.right_ear = calc_ear(landmarks, RIGHT_EYE_IDX)
        self.left_ear  = calc_ear(landmarks, LEFT_EYE_IDX)
        self.ear       = (self.right_ear + self.left_ear) / 2.0

        # ── 눈 감김 판정 ───────────────────────────────
        self.eyes_closed = self.ear < EAR_CLOSE_THRESHOLD

        # ── 감김 지속시간 ──────────────────────────────
        if self.eyes_closed:
            if self._closed_since is None:
                self._closed_since = now
            self.closed_duration = now - self._closed_since
        else:
            self._closed_since   = None
            self.closed_duration = 0.0

        # ── PERCLOS 슬라이딩 윈도우 ───────────────────
        self._window.append((now, self.eyes_closed))
        cutoff = now - PERCLOS_WINDOW_SEC
        while self._window and self._window[0][0] < cutoff:
            self._window.popleft()

        if self._window:
            closed_cnt = sum(1 for _, c in self._window if c)
            self.perclos = closed_cnt / len(self._window)
        else:
            self.perclos = 0.0

        return {
            "ear":              self.ear,
            "left_ear":         self.left_ear,
            "right_ear":        self.right_ear,
            "eyes_closed":      self.eyes_closed,
            "closed_duration":  self.closed_duration,
            "perclos":          self.perclos,
            "perclos_level":    self._perclos_level(),
        }

    def reset(self):
        self._window.clear()
        self._closed_since   = None
        self.closed_duration = 0.0
        self.perclos         = 0.0

    def _perclos_level(self) -> int:
        """0=정상 / 1=주의 / 2=위험"""
        if self.perclos >= PERCLOS_ALERT_RATIO:
            return 2
        if self.perclos >= PERCLOS_WARN_RATIO:
            return 1
        return 0
