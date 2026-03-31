# mouth_analyzer.py
# 입 열린 정도(MAR)를 계산하고 하품을 감지합니다.
#
# MAR(Mouth Aspect Ratio) 계산 방법:
#        p2        p3
#   p1                  p4
#        p6        p5
#
#   MAR = ( ||p2-p6|| + ||p3-p5|| ) / ( 2 * ||p1-p4|| )
#   → 입이 열릴수록 세로 거리가 커져 MAR 값이 증가
#
# MediaPipe 랜드마크 인덱스:
#   p1=61(왼쪽 입꼬리), p2=82(윗입술 좌), p3=312(윗입술 우),
#   p4=291(오른쪽 입꼬리), p5=317(아랫입술 우), p6=87(아랫입술 좌)

import time
import numpy as np

from dms.utils.config import (
    MAR_OPEN_THRESHOLD,
    YAWN_MIN_DURATION_SEC,
    YAWN_COOLDOWN_SEC,
)

# 입 랜드마크 인덱스
MOUTH_IDX = [61, 82, 312, 291, 317, 87]


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    # 두 점 사이의 거리 계산
    return float(np.linalg.norm(a - b))


def calc_mar(landmarks: np.ndarray) -> float:
    """6개 입 랜드마크로 MAR(입 열린 정도)를 계산합니다."""
    p = landmarks[MOUTH_IDX]   # 6개 포인트 추출
    vertical   = _dist(p[1], p[5]) + _dist(p[2], p[4])   # 입 세로 길이 합
    horizontal = _dist(p[0], p[3])                         # 입 가로 길이
    if horizontal < 1e-6:
        return 0.0
    return vertical / (2.0 * horizontal)


class MouthAnalyzer:
    """
    매 프레임 update()를 호출하면 MAR, 하품 여부, 누적 하품 횟수를 계산합니다.
    """

    def __init__(self):
        self._open_since: float | None = None   # 입이 열리기 시작한 시간
        self._last_yawn_at: float = 0.0         # 마지막 하품 인식 시간

        self.mar: float           = 0.0
        self.mouth_open: bool     = False
        self.yawn_detected: bool  = False
        self.yawn_count: int      = 0
        self.open_duration: float = 0.0   # 입이 열린 지속 시간 (초)

    def update(self, landmarks: np.ndarray, now: float | None = None) -> dict:
        """랜드마크를 받아 입 상태를 분석하고 결과를 딕셔너리로 반환합니다.

        now: 영상 파일일 때 cap.get(CAP_PROP_POS_MSEC)/1000.0 을 전달.
             None이면 벽시계(perf_counter) 사용 (카메라 모드).
        """
        if now is None:
            now = time.perf_counter()

        # MAR 계산 및 입 열림 판정
        self.mar        = calc_mar(landmarks)
        self.mouth_open = self.mar > MAR_OPEN_THRESHOLD

        # 입 열린 지속 시간 측정
        if self.mouth_open:
            if self._open_since is None:
                self._open_since = now   # 처음 열린 시점 기록
            self.open_duration = now - self._open_since
        else:
            self._open_since   = None   # 입 닫히면 초기화
            self.open_duration = 0.0

        # 하품 확정: 충분히 오래 열려 있고, 쿨다운이 지났으면 하품으로 인정
        self.yawn_detected = False
        if (
            self.open_duration >= YAWN_MIN_DURATION_SEC
            and (now - self._last_yawn_at) > YAWN_COOLDOWN_SEC
        ):
            self.yawn_detected = True
            self._last_yawn_at = now
            self.yawn_count   += 1

        return {
            "mar":           self.mar,
            "mouth_open":    self.mouth_open,
            "open_duration": self.open_duration,
            "yawn_detected": self.yawn_detected,
            "yawn_count":    self.yawn_count,
        }

    def reset(self):
        """상태를 초기화합니다."""
        self._open_since   = None
        self.open_duration = 0.0
        self.yawn_count    = 0
