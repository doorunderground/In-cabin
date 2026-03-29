"""
head_pose_analyzer.py
OpenCV solvePnP 기반 헤드 포즈 추정 + 시선 분산 누적.

추정 각도
─────────
  Pitch : 고개 끄덕임 (+ = 아래, - = 위)
  Yaw   : 좌우 회전   (+ = 오른쪽, - = 왼쪽)
  Roll  : 옆으로 기울 (+ = 오른쪽 기울, - = 왼쪽 기울)

시선 화살표 계산
────────────────
  nose_2d      : 코끝 랜드마크 픽셀 좌표 (화살표 시작점)
  gaze_end_2d  : Yaw/Pitch 각도로 계산한 시선 방향 끝점

MediaPipe 6-point 랜드마크 인덱스
──────────────────────────────────
  1   : 코끝 (Nose tip)
  152 : 턱 끝 (Chin)
  263 : 왼쪽 눈 외곽 (Left eye outer)
  33  : 오른쪽 눈 외곽 (Right eye outer)
  61  : 왼쪽 입꼬리 (Left mouth corner)
  291 : 오른쪽 입꼬리 (Right mouth corner)
"""

import time
from collections import deque

import cv2
import numpy as np

from dms.utils.config import (
    YAW_ALERT_MAX,
    PITCH_ALERT_MAX,
    HEAD_DISTRACTION_WINDOW_SEC,
    HEAD_DISTRACTION_WARN_RATIO,
    HEAD_DISTRACTION_ALERT_RATIO,
)

# ── 6-point 얼굴 3D 모델 좌표 (mm, 코끝 원점) ─────────────
FACE_3D = np.array([
    [0.0,    0.0,    0.0  ],   # 코끝 (1)
    [0.0,   -330.0, -65.0 ],   # 턱 끝 (152)
    [-225.0, 170.0, -135.0],   # 왼쪽 눈 외곽 (263)
    [225.0,  170.0, -135.0],   # 오른쪽 눈 외곽 (33)
    [-150.0,-150.0, -125.0],   # 왼쪽 입꼬리 (61)
    [150.0, -150.0, -125.0],   # 오른쪽 입꼬리 (291)
], dtype=np.float64)

POSE_IDX = [1, 152, 263, 33, 61, 291]

# 시선 화살표 길이 (픽셀)
ARROW_LEN = 120


class HeadPoseAnalyzer:
    """
    매 프레임 update(landmarks) 호출 → Pitch/Yaw/Roll + 시선 분산 판정.

    Parameters
    ----------
    frame_w, frame_h : int
        캡처 해상도 (카메라 행렬 초기화용).
    """

    def __init__(self, frame_w: int = 1280, frame_h: int = 720):
        self._cam_mat   = self._build_camera_matrix(frame_w, frame_h)
        self._dist_zero = np.zeros((4, 1), dtype=np.float64)

        # 슬라이딩 윈도우: (timestamp, is_distracted)
        self._window: deque = deque()
        self._distracted_since: float | None = None

        # 외부 참조용 최신값
        self.pitch: float = 0.0
        self.yaw:   float = 0.0
        self.roll:  float = 0.0
        self.nose_2d:     tuple = (0, 0)
        self.gaze_end_2d: tuple = (0, 0)
        self.gaze_dir_2d: tuple = (0.0, 0.0)   # 정규화된 시선 방향 단위벡터 (image 좌표계)
        self.is_distracted:       bool  = False
        self.distraction_duration: float = 0.0
        self.distraction_ratio:   float = 0.0

    # ── 메인 업데이트 ────────────────────────────────────
    def update(self, landmarks: np.ndarray) -> dict:
        now = time.perf_counter()

        # 2D 이미지 좌표 (float64)
        face_2d = landmarks[POSE_IDX].astype(np.float64)

        # solvePnP
        ok, rvec, tvec = cv2.solvePnP(
            FACE_3D, face_2d,
            self._cam_mat, self._dist_zero,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return self._default()

        # ── 오일러 각도 추출 ──────────────────────────────
        self.pitch, self.yaw, self.roll = _rvec_to_euler(rvec)

        # ── 시선 방향 단위벡터 (Yaw/Pitch → image 좌표계) ──
        #   Yaw   + = person's right = image LEFT  → x 음수
        #   Pitch + = 아래 (downward)              → y 양수
        nose = landmarks[1]
        self.nose_2d = (int(nose[0]), int(nose[1]))

        gdx = -np.sin(np.radians(self.yaw))
        gdy =  np.sin(np.radians(self.pitch))
        mag = np.hypot(gdx, gdy)
        self.gaze_dir_2d = (gdx / mag, gdy / mag) if mag > 1e-3 else (0.0, 0.0)

        self.gaze_end_2d = (
            int(nose[0] + gdx * ARROW_LEN),
            int(nose[1] + gdy * ARROW_LEN),
        )

        # ── 시선 분산 판정 ────────────────────────────────
        self.is_distracted = (
            abs(self.yaw)   > YAW_ALERT_MAX or
            abs(self.pitch) > PITCH_ALERT_MAX
        )

        if self.is_distracted:
            if self._distracted_since is None:
                self._distracted_since = now
            self.distraction_duration = now - self._distracted_since
        else:
            self._distracted_since    = None
            self.distraction_duration = 0.0

        # ── 슬라이딩 윈도우 PERCLOS 유사 비율 ────────────
        self._window.append((now, self.is_distracted))
        cutoff = now - HEAD_DISTRACTION_WINDOW_SEC
        while self._window and self._window[0][0] < cutoff:
            self._window.popleft()

        if self._window:
            n_distracted = sum(1 for _, d in self._window if d)
            self.distraction_ratio = n_distracted / len(self._window)
        else:
            self.distraction_ratio = 0.0

        return self._pack()

    def reset(self):
        self._window.clear()
        self._distracted_since    = None
        self.pitch = self.yaw = self.roll = 0.0
        self.gaze_dir_2d          = (0.0, 0.0)
        self.is_distracted        = False
        self.distraction_duration = 0.0
        self.distraction_ratio    = 0.0

    # ── 내부 헬퍼 ────────────────────────────────────────
    def _pack(self) -> dict:
        return {
            "pitch":                self.pitch,
            "yaw":                  self.yaw,
            "roll":                 self.roll,
            "nose_2d":              self.nose_2d,
            "gaze_end_2d":          self.gaze_end_2d,
            "gaze_dir_2d":          self.gaze_dir_2d,
            "is_distracted":        self.is_distracted,
            "distraction_duration": self.distraction_duration,
            "distraction_ratio":    self.distraction_ratio,
        }

    def _default(self) -> dict:
        return {
            "pitch": 0.0, "yaw": 0.0, "roll": 0.0,
            "nose_2d": (0, 0), "gaze_end_2d": (0, 0),
            "gaze_dir_2d": (0.0, 0.0),
            "is_distracted": False,
            "distraction_duration": 0.0,
            "distraction_ratio": 0.0,
        }

    @staticmethod
    def _build_camera_matrix(w: int, h: int) -> np.ndarray:
        """초점거리 = 이미지 너비 근사값 카메라 행렬."""
        f = float(w)
        return np.array([
            [f,   0.0, w / 2.0],
            [0.0, f,   h / 2.0],
            [0.0, 0.0, 1.0   ],
        ], dtype=np.float64)


# ════════════════════════════════════════════════════
#  회전 벡터 → 오일러 각도 (ZYX 분해)
# ════════════════════════════════════════════════════

def _rvec_to_euler(rvec: np.ndarray) -> tuple[float, float, float]:
    """
    Returns
    -------
    pitch_deg, yaw_deg, roll_deg : float  (단위: 도)
    """
    rmat, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)

    if sy > 1e-6:                          # 일반 케이스
        pitch = np.arctan2( rmat[2, 1], rmat[2, 2])
        yaw   = np.arctan2(-rmat[2, 0], sy)
        roll  = np.arctan2( rmat[1, 0], rmat[0, 0])
    else:                                  # Gimbal lock
        pitch = np.arctan2(-rmat[1, 2], rmat[1, 1])
        yaw   = np.arctan2(-rmat[2, 0], sy)
        roll  = 0.0

    to_deg = 180.0 / np.pi
    return pitch * to_deg, yaw * to_deg, roll * to_deg
