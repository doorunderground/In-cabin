# fps_counter.py
# 최근 N 프레임의 시간을 기록해서 현재 FPS를 계산합니다.

import time
from collections import deque


class FPSCounter:
    """
    매 프레임 tick()을 호출하면 최근 N 프레임의 평균 FPS를 계산합니다.
    """

    def __init__(self, window: int = 30):
        # 최근 30 프레임의 타임스탬프를 저장 (오래된 것은 자동 삭제)
        self._timestamps: deque = deque(maxlen=window)

    def tick(self) -> None:
        """프레임마다 호출합니다. 현재 시간을 기록합니다."""
        self._timestamps.append(time.perf_counter())

    @property
    def fps(self) -> float:
        """현재 FPS를 반환합니다."""
        if len(self._timestamps) < 2:
            return 0.0   # 데이터가 부족하면 0 반환
        elapsed = self._timestamps[-1] - self._timestamps[0]   # 가장 오래된 것 ~ 최신
        if elapsed < 1e-9:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed
