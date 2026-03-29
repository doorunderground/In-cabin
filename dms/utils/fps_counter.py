import time
from collections import deque


class FPSCounter:
    """슬라이딩 윈도우 방식의 FPS 측정기."""

    def __init__(self, window: int = 30):
        self._timestamps: deque = deque(maxlen=window)

    def tick(self) -> None:
        """매 프레임마다 호출."""
        self._timestamps.append(time.perf_counter())

    @property
    def fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed < 1e-9:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed
