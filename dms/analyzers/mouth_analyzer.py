"""
mouth_analyzer.py
MAR(Mouth Aspect Ratio) 계산 및 하품 감지.

MediaPipe FaceMesh 6-point MAR 인덱스
──────────────────────────────────────
        p2        p3
   p1                  p4
        p6        p5

  p1 = 61   (왼쪽 입꼬리)
  p2 = 82   (윗입술 좌측)
  p3 = 312  (윗입술 우측)
  p4 = 291  (오른쪽 입꼬리)
  p5 = 317  (아랫입술 우측)
  p6 = 87   (아랫입술 좌측)

MAR = ( ||p2-p6|| + ||p3-p5|| ) / ( 2 * ||p1-p4|| )
"""

import time
import numpy as np

from dms.utils.config import (
    MAR_OPEN_THRESHOLD,
    YAWN_MIN_DURATION_SEC,
    YAWN_COOLDOWN_SEC,
)

# ── 랜드마크 인덱스 ────────────────────────────────────
MOUTH_IDX = [61, 82, 312, 291, 317, 87]


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def calc_mar(landmarks: np.ndarray) -> float:
    """6개 입 랜드마크로 MAR 계산."""
    p = landmarks[MOUTH_IDX]    # shape (6, 2)
    vertical   = _dist(p[1], p[5]) + _dist(p[2], p[4])
    horizontal = _dist(p[0], p[3])
    if horizontal < 1e-6:
        return 0.0
    return vertical / (2.0 * horizontal)


class MouthAnalyzer:
    """
    매 프레임 update() 호출 → MAR, 하품 여부, 누적 하품 횟수 반환.
    """

    def __init__(self):
        self._open_since: float | None = None
        self._last_yawn_at: float = 0.0

        self.mar: float          = 0.0
        self.mouth_open: bool    = False
        self.yawn_detected: bool = False
        self.yawn_count: int     = 0
        self.open_duration: float = 0.0

    def update(self, landmarks: np.ndarray) -> dict:
        now = time.perf_counter()

        # ── MAR 계산 ───────────────────────────────────
        self.mar        = calc_mar(landmarks)
        self.mouth_open = self.mar > MAR_OPEN_THRESHOLD

        # ── 열림 지속시간 ──────────────────────────────
        if self.mouth_open:
            if self._open_since is None:
                self._open_since = now
            self.open_duration = now - self._open_since
        else:
            self._open_since   = None
            self.open_duration = 0.0

        # ── 하품 확정 ──────────────────────────────────
        self.yawn_detected = False
        if (
            self.open_duration >= YAWN_MIN_DURATION_SEC
            and (now - self._last_yawn_at) > YAWN_COOLDOWN_SEC
        ):
            self.yawn_detected  = True
            self._last_yawn_at  = now
            self.yawn_count    += 1

        return {
            "mar":           self.mar,
            "mouth_open":    self.mouth_open,
            "open_duration": self.open_duration,
            "yawn_detected": self.yawn_detected,
            "yawn_count":    self.yawn_count,
        }

    def reset(self):
        self._open_since   = None
        self.open_duration = 0.0
        self.yawn_count    = 0
