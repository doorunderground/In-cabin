# main.py - DMS(운전자 모니터링 시스템) 실행 진입점
#
# 실행 방법:
#   python main.py                                       # 카메라 사용
#   python main.py --source test_video/sleep_test1.mp4  # 영상 파일 사용
#   python main.py --source test_video/sleep_test1.mp4 --loop --speed 0.5
#
# 키 조작:
#   Q / ESC : 종료
#   SPACE   : 일시정지 / 재개 (영상 모드)
#   R       : 분석기 초기화
#   ← / →  : 5초 앞뒤 이동 (영상 모드)

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
from dms.alert.visual_overlay                import draw_overlay, draw_landmarks, draw_gaze_arrow
from dms.utils.fps_counter                   import FPSCounter
from dms.utils.config                        import FRAME_WIDTH, FRAME_HEIGHT

# ── 얼굴 미감지 시 사용할 기본값 딕셔너리들 ──────────────
# 얼굴이 감지되지 않으면 이 값들을 화면에 표시합니다.

_DEF_EYE = {
    "ear": 0.0, "left_ear": 0.0, "right_ear": 0.0,
    "eyes_closed": False, "closed_duration": 0.0,
    "perclos": 0.0,
}

_DEF_MOUTH = {
    "mar": 0.0, "mouth_open": False,
    "open_duration": 0.0, "yawn_detected": False, "yawn_count": 0,
}

_DEF_HEAD = {
    "pitch": 0.0, "yaw": 0.0, "roll": 0.0,
    "yaw_corr": 0.0, "pitch_corr": 0.0,
    "calibrated": False, "calib_progress": 0,
    "fused_gaze_2d": (0.0, 0.0),
    "gaze_zone":     "ROAD",
    "gaze_valid":    False,
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


def _parse_args():
    """커맨드라인 인수를 파싱합니다."""
    p = argparse.ArgumentParser(description="DMS Phase 1+2")
    p.add_argument("--source", default="0",
                   help="카메라 인덱스(0,1,...) 또는 영상 파일 경로")
    p.add_argument("--loop",  action="store_true",
                   help="영상 파일 반복 재생")
    p.add_argument("--speed", type=float, default=1.0,
                   help="영상 재생 속도 배율 (기본: 1.0)")
    return p.parse_args()


def main():
    args = _parse_args()

    # ── 소스 판별: 숫자면 카메라, 문자열이면 파일 ──────
    try:
        source   = int(args.source)   # 숫자로 변환 가능 → 카메라
        is_video = False
        src_name = f"Camera {source}"
    except ValueError:
        source   = args.source        # 숫자 변환 실패 → 파일 경로
        is_video = True
        if not os.path.exists(source):
            print(f"[ERROR] 파일 없음: {source}")
            sys.exit(1)
        src_name = os.path.basename(source)

    print(f"[DMS] 소스  : {src_name}")
    print(f"[DMS] 모드  : {'영상 파일' if is_video else '실시간 카메라'}")

    # ── 영상 소스 열기 ─────────────────────────────────
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] 소스를 열 수 없습니다: {source}")
        sys.exit(1)

    # 카메라라면 해상도/FPS 설정
    if not is_video:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)

    # 영상 정보 읽기
    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_video else 0
    total_sec    = total_frames / video_fps if is_video else 0.0
    # 재생 속도에 따라 프레임 간 대기 시간 계산
    frame_delay  = max(1, int(1000 / video_fps / max(args.speed, 0.1))) if is_video else 1

    # 실제 해상도 읽기
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if is_video:
        print(f"[DMS] 해상도: {actual_w}x{actual_h}  |  FPS: {video_fps:.1f}"
              f"  |  총 길이: {total_sec:.1f}s  |  재생속도: x{args.speed}")

    # ── AI 분석 모듈 초기화 ───────────────────────────
    print("[DMS] AI 모듈 초기화 중...")
    face_det     = FaceDetector()
    eye_ana      = EyeAnalyzer()
    mouth_ana    = MouthAnalyzer()
    head_ana     = HeadPoseAnalyzer(frame_w=actual_w, frame_h=actual_h)
    drowsy_cls   = DrowsinessClassifier()
    distract_cls = DistractionClassifier()
    fps_ctr      = FPSCounter(window=30)

    # 기본값으로 초기화
    eye_data     = _DEF_EYE.copy()
    mouth_data   = _DEF_MOUTH.copy()
    head_data    = _DEF_HEAD.copy()
    drowsy_res   = _DEF_DROWSY.copy()
    distract_res = _DEF_DISTRACT.copy()

    paused       = False    # 일시정지 여부
    cached_frame = None     # 일시정지 중 표시할 마지막 프레임

    print("[DMS] 실행 중.  Q/ESC: 종료 | SPACE: 정지/재개 | R: 리셋")
    WIN = f"DMS Phase 1+2  |  {src_name}"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, max(actual_w, FRAME_WIDTH), max(actual_h, FRAME_HEIGHT))

    # ── 메인 루프 ─────────────────────────────────────
    while True:

        # 일시정지 상태: 마지막 프레임을 계속 표시
        if paused and cached_frame is not None:
            display = cached_frame.copy()
            cv2.putText(display, "|| PAUSED",
                        (actual_w // 2 - 60, actual_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                        (200, 200, 200), 3, cv2.LINE_AA)
            cv2.imshow(WIN, display)
            key = cv2.waitKey(50) & 0xFF
            if key in (ord('q'), 27):      break          # 종료
            elif key == ord(' '):          paused = False  # 재개
            elif is_video and key == 81:   _seek(cap, -5)  # 5초 뒤로
            elif is_video and key == 83:   _seek(cap, +5)  # 5초 앞으로
            continue

        # 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            if is_video and args.loop:
                # 영상 끝나면 처음부터 반복
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                _reset_all(eye_ana, mouth_ana, head_ana, drowsy_cls, distract_cls)
                continue
            elif is_video:
                print("[DMS] 영상 재생 완료.")
                break
            else:
                continue   # 카메라 프레임 읽기 실패 시 재시도

        fps_ctr.tick()   # FPS 측정용 타이밍 기록

        # ── 미디어 타임스탬프 계산 ─────────────────────
        # MP4: 영상 내 실제 위치 기준 → MediaPipe·분석기 모두 영상 시간 사용
        # 카메라: None → 각 모듈이 내부적으로 perf_counter() 사용
        if is_video:
            _pos_ms       = cap.get(cv2.CAP_PROP_POS_MSEC)
            media_ts_ms   = int(_pos_ms)          # MediaPipe 용 (정수 ms)
            media_now_sec = _pos_ms / 1000.0      # 분석기 용 (소수 초)
        else:
            media_ts_ms   = None
            media_now_sec = None

        # 영상 재생 위치 정보 (사이드바 하단 진행 바에 사용)
        video_info = {
            "is_video":     is_video,
            "src_name":     src_name,
            "pos_sec":      cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if is_video else 0.0,
            "total_sec":    total_sec,
            "frame_pos":    int(cap.get(cv2.CAP_PROP_POS_FRAMES)) if is_video else 0,
            "total_frames": total_frames,
            "speed":        args.speed,
            "is_paused":    paused,
        }

        # ── 얼굴 감지 및 분석 ────────────────────────
        frame_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # BGR → RGB 변환
        landmarks  = face_det.detect(frame_rgb, timestamp_ms=media_ts_ms)
        face_found = landmarks is not None

        if face_found:
            # 얼굴이 감지되면 각 분석기 업데이트
            eye_data     = eye_ana.update(landmarks,            now=media_now_sec)
            mouth_data   = mouth_ana.update(landmarks,          now=media_now_sec)
            head_data    = head_ana.update(landmarks,           now=media_now_sec)
            drowsy_res   = drowsy_cls.update(eye_data, mouth_data, now=media_now_sec)
            distract_res = distract_cls.update(head_data,      now=media_now_sec)

            # 눈·입 외곽선 및 시선 화살표 그리기
            draw_landmarks(frame, landmarks, drowsy_res["level"])
            draw_gaze_arrow(frame, head_data, landmarks)
        else:
            # 얼굴이 없으면 모두 초기화
            _reset_all(eye_ana, mouth_ana, head_ana, drowsy_cls, distract_cls)
            eye_data     = _DEF_EYE.copy()
            mouth_data   = _DEF_MOUTH.copy()
            head_data    = _DEF_HEAD.copy()
            drowsy_res   = _DEF_DROWSY.copy()
            distract_res = _DEF_DISTRACT.copy()

        # 대시보드(사이드바, 경보 배너 등) 그리기
        draw_overlay(
            frame, eye_data, mouth_data,
            drowsy_res, head_data, distract_res,
            fps_ctr.fps, face_found, video_info,
        )

        cached_frame = frame.copy()   # 일시정지 대비 저장
        cv2.imshow(WIN, frame)

        # ── 키 입력 처리 ─────────────────────────────
        key = cv2.waitKey(frame_delay) & 0xFF
        if key in (ord('q'), 27):
            break                                                     # 종료
        elif key == ord(' ') and is_video:
            paused = not paused                                       # 일시정지 토글
        elif key == ord('r'):
            _reset_all(eye_ana, mouth_ana, head_ana, drowsy_cls, distract_cls)
            print("[DMS] 리셋 완료.")
        elif is_video and key == 81:
            _seek(cap, -5)                                            # 5초 뒤로
        elif is_video and key == 83:
            _seek(cap, +5)                                            # 5초 앞으로

    # ── 종료 처리 ─────────────────────────────────────
    cap.release()
    face_det.close()
    cv2.destroyAllWindows()
    print("[DMS] 종료.")


def _seek(cap: cv2.VideoCapture, delta_sec: float):
    """영상 재생 위치를 delta_sec 초만큼 이동합니다."""
    cur = cap.get(cv2.CAP_PROP_POS_MSEC)
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, cur + delta_sec * 1000))


def _reset_all(eye_ana, mouth_ana, head_ana, drowsy_cls, distract_cls):
    """모든 분석기의 상태를 초기화합니다."""
    eye_ana.reset()
    mouth_ana.reset()
    head_ana.reset()
    drowsy_cls.reset()
    distract_cls.reset()


if __name__ == "__main__":
    main()
