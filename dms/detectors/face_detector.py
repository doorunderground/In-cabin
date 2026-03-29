"""
face_detector.py
MediaPipe Tasks API 기반 얼굴 랜드마크 추출기 (mediapipe >= 0.10.x).

반환 형식: numpy array (478, 2) – 픽셀 좌표 (x, y)
  - 인덱스 0~467 : 표준 FaceMesh 468개
  - 인덱스 468~477: 홍채 10개

변경 이력
─────────
  mediapipe 0.10.14+ 에서 mp.solutions API 제거됨.
  → mp.tasks.vision.FaceLandmarker (Tasks API) 로 재작성.
  → 모델 파일(face_landmarker.task)을 최초 실행 시 자동 다운로드.
"""

import os
import time
import urllib.request
import numpy as np
import mediapipe as mp

from dms.utils.config import FACE_DETECTION_CONFIDENCE, FACE_TRACKING_CONFIDENCE

# ── 모델 다운로드 설정 ────────────────────────────────
_MODEL_FILENAME = "face_landmarker.task"
_MODEL_DIR      = os.path.join(os.path.dirname(__file__), "..", "..", "models")
_MODEL_PATH     = os.path.abspath(os.path.join(_MODEL_DIR, _MODEL_FILENAME))
_MODEL_URL      = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


def _ensure_model() -> str:
    """모델 파일이 없으면 자동 다운로드 후 경로 반환."""
    os.makedirs(_MODEL_DIR, exist_ok=True)
    if not os.path.exists(_MODEL_PATH):
        print(f"[FaceDetector] 모델 파일 다운로드 중...")
        print(f"  URL  : {_MODEL_URL}")
        print(f"  저장 : {_MODEL_PATH}")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("[FaceDetector] 다운로드 완료.")
    return _MODEL_PATH


class FaceDetector:
    """
    매 프레임 detect(frame_rgb) 호출 → 랜드마크 배열 반환.

    VIDEO 모드를 사용해 프레임 간 추적을 활성화한다.
    detect() 호출 시 단조 증가하는 타임스탬프(ms)를 자동 부여.
    """

    def __init__(self):
        model_path = _ensure_model()

        BaseOptions          = mp.tasks.BaseOptions
        FaceLandmarker       = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode    = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,          # 프레임 간 추적
            num_faces=1,
            min_face_detection_confidence=FACE_DETECTION_CONFIDENCE,
            min_face_presence_confidence=FACE_DETECTION_CONFIDENCE,
            min_tracking_confidence=FACE_TRACKING_CONFIDENCE,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._start_ms   = int(time.perf_counter() * 1000)

    def detect(self, frame_rgb: np.ndarray) -> np.ndarray | None:
        """
        RGB 프레임을 받아 랜드마크 배열을 반환.

        Parameters
        ----------
        frame_rgb : np.ndarray
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 로 변환한 이미지.

        Returns
        -------
        np.ndarray  shape (478, 2)  픽셀 좌표 (x, y)
        None        얼굴 미감지 시
        """
        h, w = frame_rgb.shape[:2]

        # Tasks API 는 mp.Image 객체를 입력으로 받음
        mp_img     = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp  = int(time.perf_counter() * 1000) - self._start_ms

        result = self._landmarker.detect_for_video(mp_img, timestamp)

        if not result.face_landmarks:
            return None

        lm = result.face_landmarks[0]   # NormalizedLandmark 리스트
        points = np.array(
            [[p.x * w, p.y * h] for p in lm],
            dtype=np.float32,
        )
        return points

    def close(self):
        self._landmarker.close()

    def __del__(self):
        try:
            self._landmarker.close()
        except Exception:
            pass
