# head_pose_analyzer.py
# 카메라 영상에서 머리 방향(Pitch/Yaw/Roll)과 시선 방향을 계산합니다.
#
# 각도 설명:
#   Pitch : 고개 앞뒤 끄덕임  (+ = 아래, - = 위)
#   Yaw   : 고개 좌우 돌림    (+ = 오른쪽, - = 왼쪽)
#   Roll  : 고개 옆으로 기울임
#
# 시선 퓨전:
#   머리 방향(70%) + 눈동자 방향(30%)을 합쳐서 최종 시선 방향을 만듭니다.
#
# 캘리브레이션:
#   처음 N 프레임 동안 정면을 바라보면, 그것을 "기준"으로 설정합니다.
#   이후 각도는 그 기준에서 얼마나 벗어났는지로 계산됩니다.

import time
from collections import deque

import cv2
import numpy as np

from dms.utils.config import (
    HEAD_DISTRACTION_WINDOW_SEC,
    CALIB_FRAMES,
    REPROJ_ERR_MAX,
    GAZE_HEAD_WEIGHT,
    GAZE_EYE_WEIGHT,
    GAZE_EYE_SCALE,
    GAZE_ZONE_LR_TH,
    GAZE_ZONE_DOWN_TH,
)
from dms.analyzers.eye_analyzer import RIGHT_EYE_IDX, LEFT_EYE_IDX

# 얼굴의 실제 3D 좌표 (단위: mm, 코끝 원점 기준)
# solvePnP가 2D 이미지 좌표와 이 3D 좌표를 매칭해 머리 방향을 계산함
FACE_3D = np.array([
    [0.0,    0.0,    0.0  ],   # 코끝 (1번 랜드마크)
    [0.0,   -330.0, -65.0 ],   # 턱 끝 (152번)
    [-225.0, 170.0, -135.0],   # 왼쪽 눈 바깥 (263번)
    [225.0,  170.0, -135.0],   # 오른쪽 눈 바깥 (33번)
    [-150.0,-150.0, -125.0],   # 왼쪽 입꼬리 (61번)
    [150.0, -150.0, -125.0],   # 오른쪽 입꼬리 (291번)
], dtype=np.float64)

# 위 3D 좌표에 대응하는 MediaPipe 랜드마크 번호
POSE_IDX = [1, 152, 263, 33, 61, 291]

# 홍채(눈동자) 랜드마크 인덱스
RIGHT_IRIS = 468
LEFT_IRIS  = 473

# 눈 가로 폭 계산용 (홍채 위치 정규화에 사용)
_R_OUTER = RIGHT_EYE_IDX[0]   # 33  (오른쪽 눈 바깥 끝)
_R_INNER = RIGHT_EYE_IDX[3]   # 133 (오른쪽 눈 안쪽 끝)
_L_OUTER = LEFT_EYE_IDX[0]    # 263 (왼쪽 눈 바깥 끝)
_L_INNER = LEFT_EYE_IDX[3]    # 362 (왼쪽 눈 안쪽 끝)


class HeadPoseAnalyzer:
    """
    매 프레임 update(landmarks)를 호출하면
    머리 각도(Pitch/Yaw/Roll), 시선 방향, 시선 존을 계산합니다.
    """

    def __init__(self, frame_w: int = 1280, frame_h: int = 720):
        # 카메라 내부 행렬 초기화 (화면 크기 기반으로 초점거리 근사)
        self._cam_mat   = self._build_camera_matrix(frame_w, frame_h)
        self._dist_zero = np.zeros((4, 1), dtype=np.float64)  # 렌즈 왜곡 없다고 가정

        # 시선 이탈 판정용 슬라이딩 윈도우: (시간, 이탈여부) 쌍 저장
        self._window: deque = deque()
        self._distracted_since: float | None = None   # 이탈 시작 시간

        # 캘리브레이션 버퍼: 처음 N 프레임의 각도/시선을 모아 기준값 계산
        self._calib_buf: list = []
        self.calibrated:     bool  = False
        self.yaw_baseline:   float = 0.0
        self.pitch_baseline: float = 0.0
        self._gaze_base_x:   float = 0.0
        self._gaze_base_y:   float = 0.0

        # 최신 계산 결과 (외부에서 읽을 수 있음)
        self.pitch: float = 0.0           # 고개 앞뒤 기울기 (도)
        self.yaw:   float = 0.0           # 고개 좌우 회전 (도)
        self.roll:  float = 0.0           # 고개 옆으로 기울기 (도)
        self.yaw_corr:   float = 0.0      # 보정된 Yaw (기준값 뺀 값)
        self.pitch_corr: float = 0.0      # 보정된 Pitch (기준값 뺀 값)
        self.fused_gaze_2d: tuple = (0.0, 0.0)   # 퓨전된 시선 방향벡터
        self.gaze_zone:     str   = "ROAD"        # 시선 존 (ROAD/LEFT/RIGHT/DOWN)
        self.is_distracted:        bool  = False
        self.distraction_duration: float = 0.0    # 시선 이탈 지속 시간 (초)
        self.distraction_ratio:    float = 0.0    # 최근 N초 동안 이탈 비율

    def update(self, landmarks: np.ndarray, now: float | None = None) -> dict:
        """랜드마크를 받아 머리 각도와 시선 정보를 계산하고 딕셔너리로 반환합니다.

        now: 영상 파일일 때 cap.get(CAP_PROP_POS_MSEC)/1000.0 을 전달.
             None이면 벽시계(perf_counter) 사용 (카메라 모드).
        """
        if now is None:
            now = time.perf_counter()

        # 얼굴 6개 포인트의 2D 이미지 좌표 추출
        face_2d = landmarks[POSE_IDX].astype(np.float64)

        # solvePnP: 2D ↔ 3D 매칭으로 머리 회전 벡터와 이동 벡터를 계산
        ok, rvec, tvec = cv2.solvePnP(
            FACE_3D, face_2d,
            self._cam_mat, self._dist_zero,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return self._default()   # 계산 실패 시 기본값 반환

        # 재투영 오차 확인 (solvePnP 결과가 얼마나 정확한지 검증)
        proj_pts, _ = cv2.projectPoints(
            FACE_3D, rvec, tvec, self._cam_mat, self._dist_zero
        )
        reproj_err = float(np.mean(
            np.linalg.norm(proj_pts.reshape(-1, 2) - face_2d, axis=1)
        ))
        if reproj_err > REPROJ_ERR_MAX:
            return self._default()   # 오차가 너무 크면 신뢰할 수 없으므로 무시

        # 회전 벡터 → 회전 행렬 변환
        rmat, _ = cv2.Rodrigues(rvec)
        self.pitch, self.yaw, self.roll = _rmat_to_euler(rmat)

        # 머리 방향 벡터: 회전 행렬 3열에서 추출
        # rmat[0,2]에 부호 반전 = 미러 웹캠(좌우 반전) 보정
        hx = -float(rmat[0, 2])
        hy =  float(rmat[1, 2])

        # 홍채 방향 벡터: 눈동자가 중심에서 얼마나 치우쳤는지 계산
        ex, ey = _calc_iris_dir(landmarks)

        # 머리 방향(70%)과 눈동자 방향(30%)을 합쳐 최종 시선 방향 계산
        fused_x = GAZE_HEAD_WEIGHT * hx + GAZE_EYE_WEIGHT * ex
        fused_y = GAZE_HEAD_WEIGHT * hy + GAZE_EYE_WEIGHT * ey
        self.fused_gaze_2d = (fused_x, fused_y)

        # ── 캘리브레이션 ──────────────────────────────────
        if len(self._calib_buf) < CALIB_FRAMES:
            # 아직 기준값 수집 중
            self._calib_buf.append((self.yaw, self.pitch, fused_x, fused_y))
            self.yaw_corr   = 0.0
            self.pitch_corr = 0.0
            fx_c, fy_c = fused_x, fused_y
        else:
            # 수집 완료 → 기준값 계산 (최초 1회만)
            if not self.calibrated:
                self.yaw_baseline   = float(np.median([v[0] for v in self._calib_buf]))
                self.pitch_baseline = float(np.median([v[1] for v in self._calib_buf]))
                self._gaze_base_x   = float(np.median([v[2] for v in self._calib_buf]))
                self._gaze_base_y   = float(np.median([v[3] for v in self._calib_buf]))
                self.calibrated     = True

            # 기준값을 빼서 "정면 기준 보정값" 계산
            self.yaw_corr   = self.yaw   - self.yaw_baseline
            self.pitch_corr = self.pitch - self.pitch_baseline
            fx_c = fused_x - self._gaze_base_x
            fy_c = fused_y - self._gaze_base_y

        # 시선 존 분류 (보정된 시선 방향으로 ROAD/LEFT/RIGHT/DOWN 판단)
        self.gaze_zone = _classify_zone(fx_c, fy_c)

        # 도로에서 시선이 벗어났는지 판정
        self.is_distracted = (self.gaze_zone != "ROAD")

        # 이탈 지속 시간 측정
        if self.is_distracted:
            if self._distracted_since is None:
                self._distracted_since = now   # 이탈 시작 시점 기록
            self.distraction_duration = now - self._distracted_since
        else:
            self._distracted_since    = None
            self.distraction_duration = 0.0

        # 슬라이딩 윈도우로 최근 N초 동안 이탈 비율 계산
        self._window.append((now, self.is_distracted))
        cutoff = now - HEAD_DISTRACTION_WINDOW_SEC
        while self._window and self._window[0][0] < cutoff:
            self._window.popleft()   # 오래된 데이터 제거

        if self._window:
            n_distracted = sum(1 for _, d in self._window if d)
            self.distraction_ratio = n_distracted / len(self._window)
        else:
            self.distraction_ratio = 0.0

        return self._pack()

    def reset(self):
        """슬라이딩 윈도우와 타이머를 초기화합니다. (캘리브레이션은 유지)"""
        self._window.clear()
        self._distracted_since    = None
        self.pitch = self.yaw = self.roll = 0.0
        self.yaw_corr = self.pitch_corr   = 0.0
        self.fused_gaze_2d    = (0.0, 0.0)
        self.gaze_zone        = "ROAD"
        self.is_distracted    = False
        self.distraction_duration = 0.0
        self.distraction_ratio    = 0.0

    def _pack(self) -> dict:
        """현재 상태를 딕셔너리로 묶어 반환합니다."""
        return {
            "pitch":                self.pitch,
            "yaw":                  self.yaw,
            "roll":                 self.roll,
            "yaw_corr":             self.yaw_corr,
            "pitch_corr":           self.pitch_corr,
            "calibrated":           self.calibrated,
            "calib_progress":       min(len(self._calib_buf), CALIB_FRAMES),
            "fused_gaze_2d":        self.fused_gaze_2d,
            "gaze_zone":            self.gaze_zone,
            "gaze_valid":           True,
            "is_distracted":        self.is_distracted,
            "distraction_duration": self.distraction_duration,
            "distraction_ratio":    self.distraction_ratio,
        }

    def _default(self) -> dict:
        """solvePnP 실패 시 반환할 기본값 딕셔너리."""
        return {
            "pitch": 0.0, "yaw": 0.0, "roll": 0.0,
            "yaw_corr": 0.0, "pitch_corr": 0.0,
            "calibrated":     self.calibrated,
            "calib_progress": min(len(self._calib_buf), CALIB_FRAMES),
            "fused_gaze_2d":  (0.0, 0.0),
            "gaze_zone":      self.gaze_zone,
            "gaze_valid":     False,
            "is_distracted":  False,
            "distraction_duration": 0.0,
            "distraction_ratio":    0.0,
        }

    @staticmethod
    def _build_camera_matrix(w: int, h: int) -> np.ndarray:
        """카메라 내부 행렬을 생성합니다. 초점거리 = 이미지 가로 크기로 근사."""
        f = float(w)
        return np.array([
            [f,   0.0, w / 2.0],
            [0.0, f,   h / 2.0],
            [0.0, 0.0, 1.0   ],
        ], dtype=np.float64)


def _calc_iris_dir(landmarks: np.ndarray) -> tuple[float, float]:
    """
    양쪽 눈동자(홍채) 위치를 이용해 시선 방향벡터를 계산합니다.
    홍채가 눈 중심에서 얼마나 치우쳤는지를 측정해 방향으로 변환합니다.
    두 눈의 평균값을 반환합니다.
    """
    if len(landmarks) <= max(RIGHT_IRIS, LEFT_IRIS):
        return (0.0, 0.0)

    # 오른쪽 눈: 눈 안팎 끝점의 중심 대비 홍채 위치
    r_center = (landmarks[_R_OUTER] + landmarks[_R_INNER]) / 2.0
    r_width  = float(np.linalg.norm(landmarks[_R_OUTER] - landmarks[_R_INNER])) + 1e-6
    r_offset = (landmarks[RIGHT_IRIS] - r_center) / r_width   # 정규화된 오프셋

    # 왼쪽 눈
    l_center = (landmarks[_L_OUTER] + landmarks[_L_INNER]) / 2.0
    l_width  = float(np.linalg.norm(landmarks[_L_OUTER] - landmarks[_L_INNER])) + 1e-6
    l_offset = (landmarks[LEFT_IRIS] - l_center) / l_width

    # 두 눈 평균 → 스케일 적용
    avg = (r_offset + l_offset) / 2.0
    return (float(avg[0]) * GAZE_EYE_SCALE, float(avg[1]) * GAZE_EYE_SCALE)


def _classify_zone(fx: float, fy: float) -> str:
    """보정된 시선 방향값으로 시선 존을 판별합니다."""
    if fx < -GAZE_ZONE_LR_TH:
        return "LEFT"
    if fx > GAZE_ZONE_LR_TH:
        return "RIGHT"
    if fy > GAZE_ZONE_DOWN_TH:
        return "DOWN"
    return "ROAD"


def _rmat_to_euler(rmat: np.ndarray) -> tuple[float, float, float]:
    """회전 행렬을 Pitch/Yaw/Roll 각도(도 단위)로 변환합니다."""
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)

    if sy > 1e-6:
        # 일반적인 경우
        pitch = np.arctan2( rmat[2, 1], rmat[2, 2])
        yaw   = np.arctan2(-rmat[2, 0], sy)
        roll  = np.arctan2( rmat[1, 0], rmat[0, 0])
    else:
        # 짐벌 락 (특수 케이스, 거의 발생하지 않음)
        pitch = np.arctan2(-rmat[1, 2], rmat[1, 1])
        yaw   = np.arctan2(-rmat[2, 0], sy)
        roll  = 0.0

    to_deg = 180.0 / np.pi
    return pitch * to_deg, yaw * to_deg, roll * to_deg
