"""
main.py  –  DMS Phase 1 + 2 실행 진입점
──────────────────────────────────────────────
실행 방법:
  python main.py                                      # 카메라
  python main.py --source test_video/sleep_test1.mp4  # 영상 파일
  python main.py --source test_video/sleep_test1.mp4 --loop --speed 0.5

키 조작:
  Q / ESC   : 종료
  SPACE     : 일시정지 / 재개  (영상 모드)
  R         : 분석기 초기화
  ← / →     : 5초 이동         (영상 모드)
"""

import argparse
import os
import sys

import cv2

from dms.detectors.face_detector             import FaceDetector
from dms.analyzers.eye_analyzer              import EyeAnalyzer
from dms.analyzers.mouth_analyzer            import MouthAnalyzer
from dms.analyzers.head_pose_analyzer        import HeadPoseAnalyzer
from dms.classifiers.drowsiness_classifier   import DrowsinessClassifier, DrowsinessLevel
from dms.classifiers.distraction_classifier  import DistractionClassifier, DistractionLevel
from dms.alert.visual_overlay                import (
    draw_overlay, draw_landmarks, draw_gaze_arrow,
)
from dms.utils.fps_counter import FPSCounter
from dms.utils.config      import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT

# ── 기본 데이터 (얼굴 미감지 시) ──────────────────────
_DEF_EYE = {
    "ear": 0.0, "left_ear": 0.0, "right_ear": 0.0,
    "eyes_closed": False, "closed_duration": 0.0,
    "perclos": 0.0, "perclos_level": 0,
}
_DEF_MOUTH = {
    "mar": 0.0, "mouth_open": False,
    "open_duration": 0.0, "yawn_detected": False, "yawn_count": 0,
}
_DEF_HEAD = {
    "pitch": 0.0, "yaw": 0.0, "roll": 0.0,
    "nose_2d": (0, 0), "gaze_end_2d": (0, 0),
    "gaze_dir_2d": (0.0, 0.0),
    "is_distracted": False,
    "distraction_duration": 0.0, "distraction_ratio": 0.0,
}
_DEF_DROWSY = {
    "level": DrowsinessLevel.NORMAL,
    "label": "NORMAL", "label_ko": "정상", "should_alert": False,
}
_DEF_DISTRACT = {
    "level": DistractionLevel.NORMAL,
    "label": "NORMAL", "label_ko": "전방 주시", "should_alert": False,
}


# ════════════════════════════════════════════════════
#  인자 파싱
# ════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(description="DMS Phase 1+2")
    p.add_argument("--source", default=str(CAMERA_INDEX),
                   help="카메라 인덱스(0,1,...) 또는 영상 파일 경로")
    p.add_argument("--loop",  action="store_true",
                   help="영상 파일 반복 재생")
    p.add_argument("--speed", type=float, default=1.0,
                   help="영상 재생 속도 배율 (기본: 1.0)")
    return p.parse_args()


# ════════════════════════════════════════════════════
#  메인
# ════════════════════════════════════════════════════

def main():
    args = _parse_args()

    # ── 소스 판별 ─────────────────────────────────
    try:
        source   = int(args.source)
        is_video = False
        src_name = f"Camera {source}"
    except ValueError:
        source   = args.source
        is_video = True
        if not os.path.exists(source):
            print(f"[ERROR] 파일 없음: {source}")
            sys.exit(1)
        src_name = os.path.basename(source)

    print(f"[DMS] 소스  : {src_name}")
    print(f"[DMS] 모드  : {'영상 파일' if is_video else '실시간 카메라'}")

    # ── VideoCapture ──────────────────────────────
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] 소스를 열 수 없습니다: {source}")
        sys.exit(1)

    if not is_video:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)

    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_video else 0
    total_sec    = total_frames / video_fps if is_video else 0.0
    frame_delay  = max(1, int(1000 / video_fps / max(args.speed, 0.1))) if is_video else 1

    # 실제 해상도 읽기 (HeadPoseAnalyzer 카메라 행렬에 사용)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if is_video:
        print(f"[DMS] 해상도: {actual_w}x{actual_h}  |  FPS: {video_fps:.1f}"
              f"  |  총 길이: {total_sec:.1f}s  |  재생속도: x{args.speed}")

    # ── 모듈 초기화 ───────────────────────────────
    print("[DMS] AI 모듈 초기화 중...")
    face_det    = FaceDetector()
    eye_ana     = EyeAnalyzer()
    mouth_ana   = MouthAnalyzer()
    head_ana    = HeadPoseAnalyzer(frame_w=actual_w, frame_h=actual_h)
    drowsy_cls  = DrowsinessClassifier()
    distract_cls = DistractionClassifier()
    fps_ctr     = FPSCounter(window=30)

    eye_data     = _DEF_EYE.copy()
    mouth_data   = _DEF_MOUTH.copy()
    head_data    = _DEF_HEAD.copy()
    drowsy_res   = _DEF_DROWSY.copy()
    distract_res = _DEF_DISTRACT.copy()

    paused       = False
    cached_frame = None

    print("[DMS] 실행 중.  Q/ESC: 종료 | SPACE: 정지/재개 | R: 리셋")
    WIN = f"DMS Phase 1+2  |  {src_name}"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, max(actual_w, FRAME_WIDTH), max(actual_h, FRAME_HEIGHT))

    while True:
        # ── 일시정지 ──────────────────────────────
        if paused and cached_frame is not None:
            display = cached_frame.copy()
            cv2.putText(display, "|| PAUSED",
                        (actual_w // 2 - 60, actual_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                        (200, 200, 200), 3, cv2.LINE_AA)
            cv2.imshow(WIN, display)
            key = cv2.waitKey(50) & 0xFF
            if key in (ord('q'), 27):    break
            elif key == ord(' '):        paused = False
            elif is_video and key == 81: _seek(cap, -5)
            elif is_video and key == 83: _seek(cap, +5)
            continue

        # ── 프레임 읽기 ───────────────────────────
        ret, frame = cap.read()
        if not ret:
            if is_video and args.loop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                _reset_all(eye_ana, mouth_ana, head_ana, drowsy_cls, distract_cls)
                continue
            elif is_video:
                print("[DMS] 영상 재생 완료.")
                break
            else:
                continue

        fps_ctr.tick()

        # ── 영상 진행 정보 ────────────────────────
        pos_sec   = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if is_video else 0.0
        frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) if is_video else 0

        video_info = {
            "is_video": is_video, "src_name": src_name,
            "pos_sec": pos_sec,   "total_sec": total_sec,
            "frame_pos": frame_pos, "total_frames": total_frames,
            "speed": args.speed,  "is_paused": paused,
        }

        # ── 얼굴 랜드마크 추출 ────────────────────
        frame_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks  = face_det.detect(frame_rgb)
        face_found = landmarks is not None

        if face_found:
            eye_data     = eye_ana.update(landmarks)
            mouth_data   = mouth_ana.update(landmarks)
            head_data    = head_ana.update(landmarks)
            drowsy_res   = drowsy_cls.update(eye_data, mouth_data)
            distract_res = distract_cls.update(head_data)

            # 랜드마크 + 시선 화살표 렌더링
            draw_landmarks(frame, landmarks,
                           drowsy_res["level"], distract_res["level"])
            draw_gaze_arrow(frame, head_data, distract_res["level"], landmarks)
        else:
            _reset_all(eye_ana, mouth_ana, head_ana, drowsy_cls, distract_cls)
            eye_data     = _DEF_EYE.copy()
            mouth_data   = _DEF_MOUTH.copy()
            head_data    = _DEF_HEAD.copy()
            drowsy_res   = _DEF_DROWSY.copy()
            distract_res = _DEF_DISTRACT.copy()

        # ── 대시보드 렌더링 ───────────────────────
        draw_overlay(
            frame, eye_data, mouth_data,
            drowsy_res, head_data, distract_res,
            fps_ctr.fps, face_found, video_info,
        )

        cached_frame = frame.copy()
        cv2.imshow(WIN, frame)

        # ── 키 입력 ───────────────────────────────
        key = cv2.waitKey(frame_delay) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord(' ') and is_video:
            paused = not paused
        elif key == ord('r'):
            _reset_all(eye_ana, mouth_ana, head_ana, drowsy_cls, distract_cls)
            print("[DMS] 리셋 완료.")
        elif is_video and key == 81:
            _seek(cap, -5)
        elif is_video and key == 83:
            _seek(cap, +5)

    cap.release()
    face_det.close()
    cv2.destroyAllWindows()
    print("[DMS] 종료.")


# ════════════════════════════════════════════════════
#  유틸
# ════════════════════════════════════════════════════

def _seek(cap: cv2.VideoCapture, delta_sec: float):
    cur = cap.get(cv2.CAP_PROP_POS_MSEC)
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, cur + delta_sec * 1000))


def _reset_all(eye_ana, mouth_ana, head_ana, drowsy_cls, distract_cls):
    eye_ana.reset()
    mouth_ana.reset()
    head_ana.reset()
    drowsy_cls.reset()
    distract_cls.reset()


if __name__ == "__main__":
    main()
