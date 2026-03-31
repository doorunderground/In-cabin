# visual_overlay.py
# 화면에 DMS 대시보드를 그리는 파일입니다.
#
# 화면 구성:
#   ┌── 사이드바(285px) ──┬────── 카메라 영상 ──────┐
#   │  DMS / FPS          │                         │
#   │  얼굴 감지 상태     │  [경보 배너]             │
#   │  EAR / PERCLOS      │                         │
#   │  MAR / 하품         │  눈·입 외곽선            │
#   │  졸음 레벨          │  ← 시선 방향 화살표 →   │
#   │  ─────────────────  │                         │
#   │  Pitch / Yaw / Roll │     ┌──────────────┐    │
#   │  시선 분산 게이지   │     │  우하단 패널  │    │
#   │  시선 레벨          │     └──────────────┘    │
#   │  [영상 진행 바]     │                         │
#   └─────────────────────┴─────────────────────────┘

import time

import cv2
import numpy as np

from dms.classifiers.drowsiness_classifier  import DrowsinessLevel
from dms.classifiers.distraction_classifier import DistractionLevel
from dms.analyzers.eye_analyzer             import RIGHT_EYE_IDX, LEFT_EYE_IDX
from dms.analyzers.mouth_analyzer           import MOUTH_IDX
from dms.utils.config                       import (
    ALERT_FLASH_HZ, YAW_ALERT_MAX, PITCH_ALERT_MAX, CALIB_FRAMES, GAZE_EYE_SCALE,
)

# 홍채 랜드마크 인덱스
_RIGHT_IRIS = 468
_LEFT_IRIS  = 473

# ── 색상 (BGR 순서) ───────────────────────────────────
C_GREEN  = (50,  220,  50)
C_YELLOW = (0,   210, 255)
C_ORANGE = (0,   140, 255)
C_RED    = (30,   30, 230)
C_BLUE   = (220, 130,  30)   # 시선 화살표 기본색 (파란계열)
C_WHITE  = (240, 240, 240)
C_LGRAY  = (160, 160, 160)
C_DGRAY  = (50,   50,  50)

# 졸음/시선 레벨별 색상 매핑
DROWSY_COLORS = {
    DrowsinessLevel.NORMAL:   C_GREEN,
    DrowsinessLevel.EARLY:    C_YELLOW,
    DrowsinessLevel.WARNING:  C_ORANGE,
    DrowsinessLevel.CRITICAL: C_RED,
}
DISTRACT_COLORS = {
    DistractionLevel.NORMAL:     C_GREEN,
    DistractionLevel.GLANCE:     C_YELLOW,
    DistractionLevel.DISTRACTED: C_ORANGE,
    DistractionLevel.CRITICAL:   C_RED,
}

SIDEBAR_W = 285                  # 사이드바 가로 폭 (픽셀)
FONT      = cv2.FONT_HERSHEY_SIMPLEX


# ── 자주 쓰는 그리기 헬퍼 함수들 ─────────────────────

def _text(frame, txt, x, y, scale=0.5, color=C_WHITE, thickness=1):
    # 화면에 텍스트를 그립니다
    cv2.putText(frame, txt, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def _hline(frame, y, x0=10, x1=SIDEBAR_W - 10, color=C_DGRAY):
    # 사이드바에 가로 구분선을 그립니다
    cv2.line(frame, (x0, y), (x1, y), color, 1)


def _gauge(frame, x, y, w, h, value, max_val, label, color=C_GREEN, show_pct=False):
    # 값을 막대 게이지로 그립니다
    ratio  = min(max(value / max_val, 0.0), 1.0)   # 0~1 사이로 클램프
    fill_w = int(w * ratio)
    cv2.rectangle(frame, (x, y), (x + w, y + h), C_DGRAY, -1)   # 배경
    cv2.rectangle(frame, (x, y), (x + w, y + h), C_LGRAY,  1)   # 테두리
    if fill_w > 0:
        cv2.rectangle(frame, (x + 1, y + 2), (x + fill_w, y + h - 2), color, -1)   # 채움
    val_str = f"{value * 100:.0f}%" if show_pct else f"{value:.3f}"
    _text(frame, f"{label}: {val_str}", x, y - 4, 0.42, C_WHITE)


def _draw_poly(frame, landmarks, indices, color, thickness=1):
    # 여러 랜드마크 포인트를 연결해 외곽선을 그립니다
    pts = landmarks[indices].astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [pts], isClosed=True, color=color,
                  thickness=thickness, lineType=cv2.LINE_AA)


def _draw_dots(frame, landmarks, indices, color, r=2):
    # 랜드마크 포인트 위치에 점을 찍습니다
    for i in indices:
        cv2.circle(frame, tuple(landmarks[i].astype(int)), r, color, -1, cv2.LINE_AA)


# ── 공개 함수 (main.py에서 호출) ────────────────────

def draw_overlay(
    frame,
    eye_data,
    mouth_data,
    drowsy_result,
    head_data,
    distract_result,
    fps,
    face_detected,
    video_info=None,
):
    """프레임 위에 전체 DMS 대시보드를 그립니다."""
    h, w  = frame.shape[:2]
    d_lv  = drowsy_result["level"]
    dt_lv = distract_result["level"]
    now   = time.perf_counter()

    # 사이드바 반투명 배경
    ovl = frame.copy()
    cv2.rectangle(ovl, (0, 0), (SIDEBAR_W, h), (15, 15, 15), -1)
    cv2.addWeighted(ovl, 0.65, frame, 0.35, 0, frame)

    # 제목 및 FPS
    _text(frame, "DMS", 10, 26, 0.80, C_WHITE, 2)
    _text(frame, "Phase 1+2  |  Drowsiness & Gaze", 10, 44, 0.36, C_LGRAY)
    _text(frame, f"FPS {fps:5.1f}", 10, 62, 0.44, C_LGRAY)
    _hline(frame, 70)

    # 얼굴 감지 여부 표시
    if face_detected:
        _text(frame, "[O] Face: DETECTED",  10, 88, 0.47, C_GREEN)
    else:
        _text(frame, "[X] Face: NOT FOUND", 10, 88, 0.47, C_RED)
    _hline(frame, 96)

    # 얼굴이 감지된 경우에만 분석 데이터 표시
    if face_detected:
        _draw_phase1_section(frame, eye_data, mouth_data, drowsy_result)
        _draw_phase2_section(frame, head_data, distract_result)

    # 사이드바 오른쪽 경계선
    cv2.line(frame, (SIDEBAR_W, 0), (SIDEBAR_W, h), C_LGRAY, 1)

    # 영상 진행 바 (사이드바 하단)
    if video_info:
        _draw_video_info(frame, h, video_info)

    # 경보 배너 (위험 상태일 때 화면 상단에 표시)
    if face_detected:
        _draw_alert_banners(frame, d_lv, dt_lv, now, w)

    # 우하단 레벨 바 패널
    if face_detected:
        _draw_compact_panel(frame, h, w, d_lv, dt_lv)

    return frame


def draw_landmarks(frame: np.ndarray, landmarks: np.ndarray,
                   drowsy_level: DrowsinessLevel):
    """눈과 입의 외곽선, 랜드마크 점을 그립니다. 졸음 레벨에 따라 색이 바뀝니다."""
    d_color = DROWSY_COLORS[drowsy_level]

    # 눈·입 외곽선 그리기
    _draw_poly(frame, landmarks, RIGHT_EYE_IDX, d_color, 1)
    _draw_poly(frame, landmarks, LEFT_EYE_IDX,  d_color, 1)
    _draw_poly(frame, landmarks, MOUTH_IDX,     d_color, 1)
    # 각 랜드마크 위치에 점 찍기
    _draw_dots(frame, landmarks, RIGHT_EYE_IDX, d_color, 2)
    _draw_dots(frame, landmarks, LEFT_EYE_IDX,  d_color, 2)
    _draw_dots(frame, landmarks, MOUTH_IDX,     d_color, 2)


def draw_gaze_arrow(frame: np.ndarray, head_data: dict,
                    landmarks: np.ndarray | None = None):
    """각 눈의 홍채 위치에서 시선 방향으로 화살표를 그립니다.
    - gaze_valid=True  : 머리+홍채 퓨전 방향 사용
    - gaze_valid=False : solvePnP 실패 시 홍채 위치만으로 방향 계산 (fallback)
    """
    if landmarks is None or len(landmarks) <= max(_RIGHT_IRIS, _LEFT_IRIS):
        return

    ARROW_LEN = 150   # 화살표 길이 (픽셀)

    # 시선이 도로 방향이면 파란색, 이탈이면 빨간색
    zone       = head_data.get("gaze_zone", "ROAD")
    color      = C_BLUE if zone == "ROAD" else C_RED
    gaze_valid = head_data.get("gaze_valid", False)
    fused_avg  = head_data.get("fused_gaze_2d", (0.0, 0.0))

    # 눈별로 화살표 그리기 (eye_idx, 홍채 인덱스)
    per_eye = [
        (RIGHT_EYE_IDX, _RIGHT_IRIS),
        (LEFT_EYE_IDX,  _LEFT_IRIS),
    ]

    for eye_idx, iris_idx in per_eye:
        cx = int(landmarks[iris_idx][0])
        cy = int(landmarks[iris_idx][1])

        # 홍채 위치에 작은 채움 원
        cv2.circle(frame, (cx, cy), 4, color, -1, cv2.LINE_AA)

        if gaze_valid:
            # 정상: 퓨전된 시선 방향 사용
            gdx, gdy = fused_avg
        else:
            # fallback: 홍채가 눈 중심에서 얼마나 치우쳤는지로 방향 계산
            center = (landmarks[eye_idx[0]] + landmarks[eye_idx[3]]) / 2.0
            width  = float(np.linalg.norm(
                landmarks[eye_idx[0]] - landmarks[eye_idx[3]]
            )) + 1e-6
            offset = (landmarks[iris_idx] - center) / width
            gdx = float(offset[0]) * GAZE_EYE_SCALE
            gdy = float(offset[1]) * GAZE_EYE_SCALE

        mag = float(np.sqrt(gdx * gdx + gdy * gdy))
        if mag < 1e-4:
            # 방향 정보 없음 → 정면 짧은 선
            cv2.circle(frame, (cx, cy), 7, color, 1, cv2.LINE_AA)
            cv2.line(frame, (cx, cy), (cx, cy - 12), color, 2, cv2.LINE_AA)
        else:
            # 방향 벡터를 정규화해서 화살표 끝점 계산
            dx = int(gdx / mag * ARROW_LEN)
            dy = int(gdy / mag * ARROW_LEN)
            cv2.line(frame, (cx, cy), (cx + dx, cy + dy), color, 2, cv2.LINE_AA)
            cv2.circle(frame, (cx + dx, cy + dy), 5, color, -1, cv2.LINE_AA)


# ── 내부 섹션 렌더러들 ───────────────────────────────

def _draw_phase1_section(frame, eye_data, mouth_data, drowsy_result):
    """사이드바에 EAR, PERCLOS, MAR, 졸음 레벨을 표시합니다."""
    d_lv    = drowsy_result["level"]
    d_color = DROWSY_COLORS[d_lv]

    # EAR 게이지
    ear_c = C_GREEN if not eye_data["eyes_closed"] else C_RED
    _gauge(frame, 15, 118, 250, 14, eye_data["ear"], 0.45, "EAR", ear_c)
    _text(frame,
          f"  L:{eye_data['left_ear']:.3f}  R:{eye_data['right_ear']:.3f}",
          15, 143, 0.37, C_LGRAY)

    # PERCLOS 게이지 (눈 감긴 비율)
    p  = eye_data["perclos"]
    pc = C_GREEN if p < 0.30 else (C_YELLOW if p < 0.50 else C_RED)
    _gauge(frame, 15, 163, 250, 14, p, 1.0, "PERCLOS", pc, show_pct=True)

    # 눈 감김 지속 시간
    dur = eye_data["closed_duration"]
    dc  = C_GREEN if dur < 1.5 else (C_YELLOW if dur < 2.5 else C_RED)
    _text(frame, f"Eye Closed : {dur:.1f}s", 15, 196, 0.46, dc)

    _hline(frame, 204)

    # MAR 게이지 (입 열린 정도)
    mc = C_YELLOW if mouth_data["mouth_open"] else C_GREEN
    _gauge(frame, 15, 224, 250, 14, mouth_data["mar"], 1.0, "MAR", mc)
    yawn_str = (
        f"Yawning... ({mouth_data['open_duration']:.1f}s)"
        if mouth_data["mouth_open"] else
        f"Yawns: {mouth_data['yawn_count']}"
    )
    _text(frame, yawn_str, 15, 254, 0.44, mc)

    _hline(frame, 262)

    # 졸음 레벨 블록 (0~3)
    _text(frame, "DROWSINESS", 15, 278, 0.43, C_LGRAY)
    bc_list = [C_GREEN, C_YELLOW, C_ORANGE, C_RED]
    for i in range(4):
        bx = 15 + i * 62
        cv2.rectangle(frame, (bx, 284), (bx + 55, 306),
                      bc_list[i] if i <= int(d_lv) else C_DGRAY, -1)
        cv2.rectangle(frame, (bx, 284), (bx + 55, 306), C_LGRAY, 1)

    _text(frame, drowsy_result["label"],    15, 326, 0.50, d_color, 2)
    _text(frame, drowsy_result["label_ko"], 15, 346, 0.46, d_color)


# 시선 존별 색상
_ZONE_COLOR = {
    "ROAD":  C_GREEN,
    "LEFT":  C_ORANGE,
    "RIGHT": C_ORANGE,
    "DOWN":  C_YELLOW,
}


def _draw_phase2_section(frame, head_data, distract_result):
    """사이드바에 Pitch/Yaw/Roll, 시선 존, 캘리브레이션 상태, 분산 레벨을 표시합니다."""
    _hline(frame, 356)

    dt_lv = distract_result["level"]

    _text(frame, "HEAD POSE", 15, 374, 0.43, C_LGRAY)

    pitch = head_data["pitch"]
    yaw   = head_data["yaw"]
    roll  = head_data["roll"]

    def _angle_color(val, alert, critical):
        # 각도가 클수록 색이 위험해짐
        return C_GREEN if abs(val) < alert else (C_YELLOW if abs(val) < critical else C_RED)

    _text(frame, f"  Pitch: {pitch:+6.1f} deg", 15, 390, 0.42,
          _angle_color(pitch, PITCH_ALERT_MAX, 55))
    _text(frame, f"  Yaw  : {yaw:+6.1f} deg",   15, 406, 0.42,
          _angle_color(yaw, YAW_ALERT_MAX, 60))
    _text(frame, f"  Roll : {roll:+6.1f} deg",   15, 422, 0.42,
          _angle_color(roll, 25, 50))

    # 시선 존 표시
    zone   = head_data.get("gaze_zone", "ROAD")
    zone_c = _ZONE_COLOR.get(zone, C_WHITE)
    _text(frame, f"  Zone : {zone}", 15, 438, 0.45, zone_c,
          2 if zone != "ROAD" else 1)

    # 캘리브레이션 상태 표시
    calibrated = head_data.get("calibrated", False)
    calib_prog = head_data.get("calib_progress", 0)
    yaw_c      = head_data.get("yaw_corr", 0.0)
    pit_c      = head_data.get("pitch_corr", 0.0)
    if calibrated:
        _text(frame,
              f"  CAL:OK  dY={yaw_c:+.1f} dP={pit_c:+.1f}",
              15, 454, 0.37, C_GREEN)
    else:
        _text(frame,
              f"  CAL: {calib_prog}/{CALIB_FRAMES}  collecting...",
              15, 454, 0.37, C_YELLOW)

    _hline(frame, 462)

    # 시선 분산 게이지
    _text(frame, "GAZE DISTRACTION", 15, 478, 0.43, C_LGRAY)

    dur   = head_data["distraction_duration"]
    ratio = head_data["distraction_ratio"]

    dur_c   = C_GREEN if dur < 2 else (C_YELLOW if dur < 5 else C_RED)
    ratio_c = C_GREEN if ratio < 0.30 else (C_YELLOW if ratio < 0.60 else C_RED)

    _text(frame, f"  Duration: {dur:.1f}s", 15, 494, 0.42, dur_c)
    _gauge(frame, 15, 510, 250, 14, ratio, 1.0, "Ratio", ratio_c, show_pct=True)

    # 시선 분산 레벨 블록 (0~3)
    bc_list = [C_GREEN, C_YELLOW, C_ORANGE, C_RED]
    for i in range(4):
        bx = 15 + i * 62
        cv2.rectangle(frame, (bx, 532), (bx + 55, 550),
                      bc_list[i] if i <= int(dt_lv) else C_DGRAY, -1)
        cv2.rectangle(frame, (bx, 532), (bx + 55, 550), C_LGRAY, 1)

    dt_color = DISTRACT_COLORS[dt_lv]
    _text(frame, distract_result["label"],    15, 570, 0.50, dt_color, 2)
    _text(frame, distract_result["label_ko"], 15, 588, 0.46, dt_color)


# ── 우하단 컴팩트 바 패널 ────────────────────────────

_BAR_COLS = [C_GREEN, C_YELLOW, C_ORANGE, C_RED]   # 레벨 0~3 색상
_PANEL_W  = 260
_PANEL_H  = 118
_MARGIN   = 12


def _draw_compact_panel(frame, fh, fw, drowsy_lv, distract_lv):
    """화면 우하단에 졸음/시선분산 레벨을 막대 그래프로 표시합니다."""
    pw, ph = _PANEL_W, _PANEL_H
    px = fw - pw - _MARGIN
    py = fh - ph - _MARGIN

    # 반투명 배경
    ovl = frame.copy()
    cv2.rectangle(ovl, (px, py), (px + pw, py + ph), (15, 15, 15), -1)
    cv2.addWeighted(ovl, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (px, py), (px + pw, py + ph), C_LGRAY, 1)

    col_w   = (pw - 30) // 2   # 컬럼 폭
    labels  = ["Distraction", "Drowsiness"]
    levels  = [int(distract_lv), int(drowsy_lv)]
    col_xs  = [px + 8, px + 8 + col_w + 14]

    bar_h    = 16
    bar_gap  = 5
    n_bars   = 4
    label_y  = py + 18
    bars_top = label_y + 8

    for col_i, (label, level, cx) in enumerate(zip(labels, levels, col_xs)):
        _text(frame, label, cx, label_y, 0.38, C_LGRAY)

        # 위가 레벨3(위험), 아래가 레벨0(정상)으로 막대 4개 그리기
        for bar_i in range(n_bars - 1, -1, -1):
            bar_y  = bars_top + (n_bars - 1 - bar_i) * (bar_h + bar_gap)
            filled = bar_i <= level
            color  = _BAR_COLS[bar_i] if filled else C_DGRAY
            cv2.rectangle(frame, (cx, bar_y), (cx + col_w, bar_y + bar_h), color, -1)
            cv2.rectangle(frame, (cx, bar_y), (cx + col_w, bar_y + bar_h), C_LGRAY, 1)

        # 두 컬럼 사이 구분선
        if col_i == 0:
            mid_x = col_xs[0] + col_w + 7
            cv2.line(frame, (mid_x, py + 4), (mid_x, py + ph - 4), C_DGRAY, 1)


# ── 경보 배너 ────────────────────────────────────────

def _draw_alert_banners(frame, d_lv, dt_lv, now, fw):
    """졸음 경고와 시선 분산 경고 배너를 화면 상단에 표시합니다."""
    banner_h = 52
    flash_on = (int(now * ALERT_FLASH_HZ * 2) % 2 == 0)   # 깜빡임 타이밍 계산

    # 졸음 경보 배너 (상단)
    if d_lv >= DrowsinessLevel.WARNING:
        _draw_one_banner(frame, fw, 0,
                         DROWSY_COLORS[d_lv], banner_h, flash_on,
                         d_lv == DrowsinessLevel.CRITICAL,
                         "!!! CRITICAL - DROWSY !!!" if d_lv == DrowsinessLevel.CRITICAL
                         else "** DROWSY WARNING **")

    # 시선 분산 경보 배너 (졸음 배너 아래)
    if dt_lv >= DistractionLevel.DISTRACTED:
        y_off = banner_h if d_lv >= DrowsinessLevel.WARNING else 0
        _draw_one_banner(frame, fw, y_off,
                         DISTRACT_COLORS[dt_lv], banner_h, flash_on,
                         dt_lv == DistractionLevel.CRITICAL,
                         "!!! LOOK AT ROAD !!!" if dt_lv == DistractionLevel.CRITICAL
                         else "** EYES OFF ROAD **")


def _draw_one_banner(frame, fw, y_offset, color, bh, flash_on, is_critical, msg):
    """배너 하나를 그립니다. CRITICAL이면 항상 표시, 아니면 깜빡임."""
    if is_critical or flash_on:
        ovl = frame.copy()
        cv2.rectangle(ovl, (SIDEBAR_W, y_offset), (fw, y_offset + bh), color, -1)
        alpha = 0.72 if is_critical else 0.55
        cv2.addWeighted(ovl, alpha, frame, 1 - alpha, 0, frame)
    _text(frame, msg, SIDEBAR_W + 18, y_offset + 34, 0.85, C_WHITE, 2)


# ── 영상 진행 바 ─────────────────────────────────────

def _draw_video_info(frame, fh, info):
    """사이드바 하단에 영상 재생 위치와 진행 바를 표시합니다."""
    bar_h  = 36
    y_base = fh - bar_h - 4

    cv2.rectangle(frame, (0, y_base - 18), (SIDEBAR_W, fh), (20, 20, 20), -1)
    cv2.line(frame, (10, y_base - 18), (SIDEBAR_W - 10, y_base - 18), C_DGRAY, 1)

    if not info.get("is_video"):
        _text(frame, f"CAM  {info.get('src_name', '')}", 12, y_base - 4, 0.40, C_LGRAY)
        return

    name  = info.get("src_name", "")[:25]
    speed = info.get("speed", 1.0)
    spd   = f"  x{speed:.1f}" if speed != 1.0 else ""
    _text(frame, f"{name}{spd}", 12, y_base - 4, 0.37, C_LGRAY)

    pos   = info.get("pos_sec",   0.0)
    total = info.get("total_sec", 0.0)

    def _fmt(s):
        # 초를 MM:SS 형식으로 변환
        m, sec = divmod(int(s), 60)
        return f"{m:02d}:{sec:02d}"

    _text(frame, f"{_fmt(pos)} / {_fmt(total)}", 12, y_base + 12, 0.40, C_WHITE)

    # 진행 바
    bx, by = 12, y_base + 20
    bw, bh = SIDEBAR_W - 24, 8
    ratio  = (pos / total) if total > 0 else 0.0
    fill_w = int(bw * min(ratio, 1.0))

    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), C_DGRAY, -1)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), C_LGRAY,  1)
    if fill_w > 0:
        cv2.rectangle(frame, (bx + 1, by + 1), (bx + fill_w, by + bh - 1), C_WHITE, -1)

    if info.get("is_paused"):
        _text(frame, "|| PAUSED", bx + bw - 68, y_base + 12, 0.40, C_YELLOW)
