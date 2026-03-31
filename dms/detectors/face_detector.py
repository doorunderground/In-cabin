# face_detector.py
# MediaPipe를 이용해 얼굴 랜드마크(478개 포인트)를 추출합니다.
# 처음 실행 시 모델 파일을 자동으로 다운로드합니다.
#
# 반환값: numpy array (478, 2) – 픽셀 좌표 (x, y)
#   인덱스 0~467 : 얼굴 468개 랜드마크
#   인덱스 468~477: 홍채 10개 랜드마크

import os
import time
import urllib.request
import numpy as np
import mediapipe as mp

from dms.utils.config import FACE_DETECTION_CONFIDENCE, FACE_TRACKING_CONFIDENCE

# 모델 파일 경로 설정
_MODEL_FILENAME = "face_landmarker.task"
_MODEL_DIR      = os.path.join(os.path.dirname(__file__), "..", "..", "models")
_MODEL_PATH     = os.path.abspath(os.path.join(_MODEL_DIR, _MODEL_FILENAME))
_MODEL_URL      = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


def _ensure_model() -> str:
    """모델 파일이 없으면 자동으로 다운로드하고 경로를 반환합니다."""
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
    매 프레임 detect(frame_rgb)를 호출하면 얼굴 랜드마크를 반환합니다.
    VIDEO 모드를 사용해 프레임 간 추적을 활성화합니다.
    """

    def __init__(self):
        model_path = _ensure_model()

        # MediaPipe Tasks API 설정
        BaseOptions           = mp.tasks.BaseOptions
        FaceLandmarker        = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode     = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,   # 비디오 모드: 프레임 간 추적
            num_faces=1,                             # 한 명만 감지
            min_face_detection_confidence=FACE_DETECTION_CONFIDENCE,
            min_face_presence_confidence=FACE_DETECTION_CONFIDENCE,
            min_tracking_confidence=FACE_TRACKING_CONFIDENCE,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._start_ms   = int(time.perf_counter() * 1000)

    def detect(self, frame_rgb: np.ndarray,
               timestamp_ms: int | None = None) -> np.ndarray | None:
        """
        RGB 이미지를 받아 얼굴 랜드마크 배열을 반환합니다.
        얼굴이 없으면 None을 반환합니다.

        timestamp_ms: 영상 파일일 때 cap.get(CAP_PROP_POS_MSEC)의 정수값을 전달.
                      None이면 벽시계(perf_counter) 기준으로 자동 계산 (카메라 모드).
        """
        h, w = frame_rgb.shape[:2]

        # MediaPipe가 요구하는 이미지 형식으로 변환
        mp_img    = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp = (timestamp_ms if timestamp_ms is not None
                     else int(time.perf_counter() * 1000) - self._start_ms)

        result = self._landmarker.detect_for_video(mp_img, timestamp)

        if not result.face_landmarks:
            return None

        # 정규화된 좌표(0~1)를 픽셀 좌표로 변환
        lm = result.face_landmarks[0]
        points = np.array(
            [[p.x * w, p.y * h] for p in lm],
            dtype=np.float32,
        )
        return points

    def close(self):
        """리소스를 해제합니다."""
        self._landmarker.close()

    def __del__(self):
        try:
            self._landmarker.close()
        except Exception:
            pass
