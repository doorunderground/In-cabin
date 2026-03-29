"""
visual_overlay.py
DMS Phase 1 + Phase 2 대시보드 오버레이 렌더러.

레이아웃
─────────────────────────────────────────────────
  ┌── 사이드바(285px) ──┬────── 비디오 영역 ──────┐
  │  DMS / FPS          │                         │
  │  Face 상태          │  [경보 배너]             │
  │  EAR / PERCLOS      │                         │
  │  MAR / 하품         │  얼굴 랜드마크 (눈/입)  │
  │  졸음 레벨          │  ← 시선 방향 화살표 →   │
  │  ─────────────────  │                         │
  │  Pitch / Yaw / Roll │                         │
  │  시선 분산 게이지   │     ┌──────────────┐    │
  │  시선 레벨          │     │  Distraction │    │  ← 우하단
  │                     │     │  Drowsiness  │    │     바 패널
  │  [영상 진행 바]     │     └──────────────┘    │
  └─────────────────────┴─────────────────────────┘
"""

import time

import cv2
import numpy as np

from dms.classifiers.drowsiness_classifier  import DrowsinessLevel
from dms.classifiers.distraction_classifier import DistractionLevel
from dms.analyzers.eye_analyzer             import RIGHT_EYE_IDX, LEFT_EYE_IDX
from dms.analyzers.mouth_analyzer           import MOUTH_IDX
from dms.utils.config                       import ALERT_FLASH_HZ, YAW_ALERT_MAX, PITCH_ALERT_MAX

# ── 색상 팔레트 (BGR) ─────────────────────────────────
C_GREEN  = (50,  220,  50)
C_YELLOW = (0,   210, 255)
C_ORANGE = (0,   140, 255)
C_RED    = (30,   30, 230)
C_BLUE   = (220, 130,  30)   # 시선 화살표 기본색 (파란계열)
C_WHITE  = (240, 240, 240)
C_LGRAY  = (160, 160, 160)
C_DGRAY  = (50,   50,  50)

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

SIDEBAR_W = 285
FONT      = cv2.FONT_HERSHEY_SIMPLEX


# ════════════════════════════════════════════════════
#  헬퍼 함수
# ════════════════════════════════════════════════════

def _text(frame, txt, x, y, scale=0.5, color=C_WHITE, thickness=1):
    cv2.putText(frame, txt, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def _hline(frame, y, x0=10, x1=SIDEBAR_W - 10, color=C_DGRAY):
    cv2.line(frame, (x0, y), (x1, y), color, 1)


def _gauge(frame, x, y, w, h, value, max_val,
           label, color=C_GREEN, show_pct=False):
    ratio  = min(max(value / max_val, 0.0), 1.0)
    fill_w = int(w * ratio)
    cv2.rectangle(frame, (x, y), (x + w, y + h), C_DGRAY, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), C_LGRAY,  1)
    if fill_w > 0:
        cv2.rectangle(frame, (x + 1, y + 2),
                      (x + fill_w, y + h - 2), color, -1)
    val_str = f"{value * 100:.0f}%" if show_pct else f"{value:.3f}"
    _text(frame, f"{label}: {val_str}", x, y - 4, 0.42, C_WHITE)


def _draw_poly(frame, landmarks, indices, color, thickness=1):
    pts = landmarks[indices].astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [pts], isClosed=True, color=color,
                  thickness=thickness, lineType=cv2.LINE_AA)


def _draw_dots(frame, landmarks, indices, color, r=2):
    for i in indices:
        cv2.circle(frame, tuple(landmarks[i].astype(int)), r, color, -1, cv2.LINE_AA)


# ════════════════════════════════════════════════════
#  공개 API
# ════════════════════════════════════════════════════

def draw_overlay(
    frame:          np.ndarray,
    eye_data:       dict,
    mouth_data:     dict,
    drowsy_result:  dict,
    head_data:      dict,
    distract_result: dict,
    fps:            float,
    face_detected:  bool,
    video_info:     dict | None = None,
) -> np.ndarray:
    """프레임 위에 전체 DMS 대시보드를 렌더링."""
    h, w   = frame.shape[:2]
    d_lv   = drowsy_result["level"]
    dt_lv  = distract_result["level"]
    now    = time.perf_counter()

    # ── 사이드바 반투명 배경 ──────────────────────────
    ovl = frame.copy()
    cv2.rectangle(ovl, (0, 0), (SIDEBAR_W, h), (15, 15, 15), -1)
    cv2.addWeighted(ovl, 0.65, frame, 0.35, 0, frame)

    # ── 제목 ──────────────────────────────────────────
    _text(frame, "DMS", 10, 26, 0.80, C_WHITE, 2)
    _text(frame, "Phase 1+2  |  Drowsiness & Gaze", 10, 44, 0.36, C_LGRAY)
    _text(frame, f"FPS {fps:5.1f}", 10, 62, 0.44, C_LGRAY)
    _hline(frame, 70)

    # ── 얼굴 감지 ─────────────────────────────────────
    if face_detected:
        _text(frame, "[O] Face: DETECTED",  10, 88, 0.47, C_GREEN)
    else:
        _text(frame, "[X] Face: NOT FOUND", 10, 88, 0.47, C_RED)
    _hline(frame, 96)

    if face_detected:
        _draw_phase1_section(frame, eye_data, mouth_data, drowsy_result)
        _draw_phase2_section(frame, head_data, distract_result)

    # ── 사이드바 경계선 ───────────────────────────────
    cv2.line(frame, (SIDEBAR_W, 0), (SIDEBAR_W, h), C_LGRAY, 1)

    # ── 영상 진행 바 (사이드바 하단) ─────────────────
    if video_info:
        _draw_video_info(frame, h, video_info)

    # ── 경보 배너 ─────────────────────────────────────
    if face_detected:
        _draw_alert_banners(frame, d_lv, dt_lv, now, w)

    # ── 우하단 바 패널 ────────────────────────────────
    if face_detected:
        _draw_compact_panel(frame, h, w, d_lv, dt_lv)

    return frame


def draw_landmarks(frame: np.ndarray, landmarks: np.ndarray,
                   drowsy_level: DrowsinessLevel,
                   distract_level: DistractionLevel):
    """눈·입 외곽선, 시선 화살표 렌더링."""
    d_color  = DROWSY_COLORS[drowsy_level]
    dt_color = DISTRACT_COLORS[distract_level]

    # 눈·입 외곽선 (졸음 색상)
    _draw_poly(frame, landmarks, RIGHT_EYE_IDX, d_color, 1)
    _draw_poly(frame, landmarks, LEFT_EYE_IDX,  d_color, 1)
    _draw_poly(frame, landmarks, MOUTH_IDX,     d_color, 1)
    _draw_dots(frame, landmarks, RIGHT_EYE_IDX, d_color, 2)
    _draw_dots(frame, landmarks, LEFT_EYE_IDX,  d_color, 2)
    _draw_dots(frame, landmarks, MOUTH_IDX,     d_color, 2)


def draw_gaze_arrow(frame: np.ndarray, head_data: dict,
                    distract_level: DistractionLevel,
                    landmarks: np.ndarray | None = None):
    """Yaw/Pitch 기반 두 눈 평행 시선 화살표 (레퍼런스 이미지 스타일).

    부호 규약 (head_pose_analyzer 기준):
      Yaw   + = person's right = image 왼쪽 → dx 음수
      Pitch + = 아래                        → dy 양수
    두 눈 모두 동일 방향벡터 → 평행 화살표.
    """
    if landmarks is None:
        return

    yaw   = head_data.get("yaw",   0.0)
    pitch = head_data.get("pitch", 0.0)

    ARROW_LEN = 60
    fdx = -ARROW_LEN * np.sin(np.radians(yaw))
    fdy =  ARROW_LEN * np.sin(np.radians(pitch))

    # 1° 미만이면 화살표 생략 (정면 주시)
    if abs(fdx) < 1.0 and abs(fdy) < 1.0:
        return

    dx = int(fdx)
    dy = int(fdy)

    for eye_idx in [RIGHT_EYE_IDX, LEFT_EYE_IDX]:
        pts = landmarks[eye_idx]              # (6, 2)
        cx  = int(pts[:, 0].mean())
        cy  = int(pts[:, 1].mean())

        cv2.arrowedLine(
            frame,
            (cx, cy), (cx + dx, cy + dy),
            C_BLUE,
            thickness=2,
            line_type=cv2.LINE_AA,
            tipLength=0.3,
        )


# ════════════════════════════════════════════════════
#  내부 섹션 렌더러
# ════════════════════════════════════════════════════

def _draw_phase1_section(frame, eye_data, mouth_data, drowsy_result):
    """사이드바 Phase 1 영역 (EAR / PERCLOS / MAR / 졸음 레벨)."""
    d_lv    = drowsy_result["level"]
    d_color = DROWSY_COLORS[d_lv]

    # EAR
    ear_c = C_GREEN if not eye_data["eyes_closed"] else C_RED
    _gauge(frame, 15, 118, 250, 14, eye_data["ear"], 0.45, "EAR", ear_c)
    _text(frame,
          f"  L:{eye_data['left_ear']:.3f}  R:{eye_data['right_ear']:.3f}",
          15, 143, 0.37, C_LGRAY)

    # PERCLOS
    p  = eye_data["perclos"]
    pc = C_GREEN if p < 0.30 else (C_YELLOW if p < 0.50 else C_RED)
    _gauge(frame, 15, 163, 250, 14, p, 1.0, "PERCLOS", pc, show_pct=True)

    # 눈 감김 지속
    dur = eye_data["closed_duration"]
    dc  = C_GREEN if dur < 1.5 else (C_YELLOW if dur < 2.5 else C_RED)
    _text(frame, f"Eye Closed : {dur:.1f}s", 15, 196, 0.46, dc)

    _hline(frame, 204)

    # MAR
    mc = C_YELLOW if mouth_data["mouth_open"] else C_GREEN
    _gauge(frame, 15, 224, 250, 14, mouth_data["mar"], 1.0, "MAR", mc)
    yawn_str = (
        f"Yawning... ({mouth_data['open_duration']:.1f}s)"
        if mouth_data["mouth_open"] else
        f"Yawns: {mouth_data['yawn_count']}"
    )
    _text(frame, yawn_str, 15, 254, 0.44, mc)

    _hline(frame, 262)

    # 졸음 레벨 블록
    _text(frame, "DROWSINESS", 15, 278, 0.43, C_LGRAY)
    bc_list = [C_GREEN, C_YELLOW, C_ORANGE, C_RED]
    for i in range(4):
        bx = 15 + i * 62
        cv2.rectangle(frame, (bx, 284), (bx + 55, 306),
                      bc_list[i] if i <= int(d_lv) else C_DGRAY, -1)
        cv2.rectangle(frame, (bx, 284), (bx + 55, 306), C_LGRAY, 1)

    _text(frame, drowsy_result["label"],    15, 326, 0.50, d_color, 2)
    _text(frame, drowsy_result["label_ko"], 15, 346, 0.46, d_color)


def _draw_phase2_section(frame, head_data, distract_result):
    """사이드바 Phase 2 영역 (Pitch/Yaw/Roll + 시선 분산 레벨)."""
    _hline(frame, 356)

    dt_lv = distract_result["level"]

    _text(frame, "HEAD POSE", 15, 374, 0.43, C_LGRAY)

    pitch = head_data["pitch"]
    yaw   = head_data["yaw"]
    roll  = head_data["roll"]

    def _angle_color(val, alert, critical):
        return C_GREEN if abs(val) < alert else (C_YELLOW if abs(val) < critical else C_RED)

    _text(frame, f"  Pitch: {pitch:+6.1f} deg", 15, 390, 0.42,
          _angle_color(pitch, PITCH_ALERT_MAX, 55))
    _text(frame, f"  Yaw  : {yaw:+6.1f} deg",   15, 406, 0.42,
          _angle_color(yaw, YAW_ALERT_MAX, 60))
    _text(frame, f"  Roll : {roll:+6.1f} deg",   15, 422, 0.42,
          _angle_color(roll, 25, 50))

    _hline(frame, 430)

    # 시선 분산 게이지
    _text(frame, "GAZE DISTRACTION", 15, 448, 0.43, C_LGRAY)

    dur   = head_data["distraction_duration"]
    ratio = head_data["distraction_ratio"]

    dur_c   = C_GREEN if dur < 2 else (C_YELLOW if dur < 5 else C_RED)
    ratio_c = C_GREEN if ratio < 0.30 else (C_YELLOW if ratio < 0.60 else C_RED)

    _text(frame, f"  Duration: {dur:.1f}s", 15, 464, 0.42, dur_c)
    _gauge(frame, 15, 482, 250, 14, ratio, 1.0, "Ratio", ratio_c, show_pct=True)

    # 시선 분산 레벨 블록
    bc_list = [C_GREEN, C_YELLOW, C_ORANGE, C_RED]
    for i in range(4):
        bx = 15 + i * 62
        cv2.rectangle(frame, (bx, 504), (bx + 55, 522),
                      bc_list[i] if i <= int(dt_lv) else C_DGRAY, -1)
        cv2.rectangle(frame, (bx, 504), (bx + 55, 522), C_LGRAY, 1)

    dt_color = DISTRACT_COLORS[dt_lv]
    _text(frame, distract_result["label"],    15, 542, 0.50, dt_color, 2)
    _text(frame, distract_result["label_ko"], 15, 560, 0.46, dt_color)


# ════════════════════════════════════════════════════
#  우하단 컴팩트 바 패널 (이미지 참조 스타일)
# ════════════════════════════════════════════════════

_BAR_COLS   = [C_GREEN, C_YELLOW, C_ORANGE, C_RED]   # 레벨 0~3 색상
_PANEL_W    = 260
_PANEL_H    = 118
_MARGIN     = 12


def _draw_compact_panel(frame, fh: int, fw: int,
                        drowsy_lv: DrowsinessLevel,
                        distract_lv: DistractionLevel):
    """
    우하단 컴팩트 패널.
    두 컬럼 (Distraction / Drowsiness), 각 4개 수평 바.
    레벨에 따라 아래→위로 바가 채워짐.
    """
    pw, ph = _PANEL_W, _PANEL_H
    px = fw - pw - _MARGIN
    py = fh - ph - _MARGIN

    # 배경
    ovl = frame.copy()
    cv2.rectangle(ovl, (px, py), (px + pw, py + ph), (15, 15, 15), -1)
    cv2.addWeighted(ovl, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (px, py), (px + pw, py + ph), C_LGRAY, 1)

    # 컬럼 설정
    col_w   = (pw - 30) // 2   # 각 컬럼 폭
    labels  = ["Distraction", "Drowsiness"]
    levels  = [int(distract_lv), int(drowsy_lv)]
    col_xs  = [px + 8, px + 8 + col_w + 14]

    bar_h    = 16
    bar_gap  = 5
    n_bars   = 4
    label_y  = py + 18
    bars_top  = label_y + 8

    for col_i, (label, level, cx) in enumerate(zip(labels, levels, col_xs)):
        # 컬럼 제목
        _text(frame, label, cx, label_y, 0.38, C_LGRAY)

        # 4개 바 (위가 level 3, 아래가 level 0)
        for bar_i in range(n_bars - 1, -1, -1):
            bar_y = bars_top + (n_bars - 1 - bar_i) * (bar_h + bar_gap)
            filled = bar_i <= level
            color  = _BAR_COLS[bar_i] if filled else C_DGRAY
            cv2.rectangle(frame,
                          (cx, bar_y),
                          (cx + col_w, bar_y + bar_h),
                          color, -1)
            cv2.rectangle(frame,
                          (cx, bar_y),
                          (cx + col_w, bar_y + bar_h),
                          C_LGRAY, 1)

        # 컬럼 사이 구분선
        if col_i == 0:
            mid_x = col_xs[0] + col_w + 7
            cv2.line(frame,
                     (mid_x, py + 4), (mid_x, py + ph - 4),
                     C_DGRAY, 1)


# ════════════════════════════════════════════════════
#  경보 배너
# ════════════════════════════════════════════════════

def _draw_alert_banners(frame, d_lv, dt_lv, now, fw):
    """졸음 + 시선 분산 경보 배너 (최대 2개 동시 표시)."""
    banner_h  = 52
    flash_on  = (int(now * ALERT_FLASH_HZ * 2) % 2 == 0)

    # 졸음 경보 (상단)
    if d_lv >= DrowsinessLevel.WARNING:
        _draw_one_banner(frame, fw, 0,
                         DROWSY_COLORS[d_lv],
                         banner_h, flash_on,
                         d_lv == DrowsinessLevel.CRITICAL,
                         "!!! CRITICAL - DROWSY !!!" if d_lv == DrowsinessLevel.CRITICAL
                         else "** DROWSY WARNING **")

    # 시선 분산 경보 (졸음 배너 아래)
    if dt_lv >= DistractionLevel.DISTRACTED:
        y_off = banner_h if d_lv >= DrowsinessLevel.WARNING else 0
        _draw_one_banner(frame, fw, y_off,
                         DISTRACT_COLORS[dt_lv],
                         banner_h, flash_on,
                         dt_lv == DistractionLevel.CRITICAL,
                         "!!! LOOK AT ROAD !!!" if dt_lv == DistractionLevel.CRITICAL
                         else "** EYES OFF ROAD **")


def _draw_one_banner(frame, fw, y_offset, color, bh, flash_on, is_critical, msg):
    if is_critical or flash_on:
        ovl = frame.copy()
        cv2.rectangle(ovl, (SIDEBAR_W, y_offset), (fw, y_offset + bh), color, -1)
        cv2.addWeighted(ovl, 0.72 if is_critical else 0.55,
                        frame, 1 - (0.72 if is_critical else 0.55), 0, frame)
    _text(frame, msg, SIDEBAR_W + 18, y_offset + 34, 0.85, C_WHITE, 2)


# ════════════════════════════════════════════════════
#  영상 진행 바 (사이드바 하단)
# ════════════════════════════════════════════════════

def _draw_video_info(frame, fh: int, info: dict):
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
        m, sec = divmod(int(s), 60)
        return f"{m:02d}:{sec:02d}"

    _text(frame, f"{_fmt(pos)} / {_fmt(total)}", 12, y_base + 12, 0.40, C_WHITE)

    bx, by = 12, y_base + 20
    bw, bh = SIDEBAR_W - 24, 8
    ratio  = (pos / total) if total > 0 else 0.0
    fill_w = int(bw * min(ratio, 1.0))

    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), C_DGRAY, -1)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), C_LGRAY,  1)
    if fill_w > 0:
        cv2.rectangle(frame, (bx + 1, by + 1),
                      (bx + fill_w, by + bh - 1), C_WHITE, -1)

    if info.get("is_paused"):
        _text(frame, "|| PAUSED", bx + bw - 68, y_base + 12, 0.40, C_YELLOW)
