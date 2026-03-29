# ─────────────────────────────────────────────
#  DMS Phase 1 & 2 – Configuration & Thresholds
# ─────────────────────────────────────────────

# ── Camera ────────────────────────────────────
CAMERA_INDEX  = 0
FRAME_WIDTH   = 1280
FRAME_HEIGHT  = 720

# ── MediaPipe ─────────────────────────────────
FACE_DETECTION_CONFIDENCE  = 0.5
FACE_TRACKING_CONFIDENCE   = 0.5

# ── EAR (Eye Aspect Ratio) ────────────────────
# 정상 개안: ~0.30–0.40  /  눈 감김: < 0.25
EAR_CLOSE_THRESHOLD  = 0.25   # 이 값 미만 → 눈 감김으로 판정
EAR_WARN_THRESHOLD   = 0.22   # 이 값 미만 → 더 강한 감김

# ── Drowsiness timing (seconds) ───────────────
DROWSY_EARLY_SEC    = 1.5   # 눈 감김 지속 → EARLY 단계
DROWSY_WARNING_SEC  = 2.5   # 눈 감김 지속 → WARNING 단계
DROWSY_CRITICAL_SEC = 3.5   # 눈 감김 지속 → CRITICAL 단계

# ── PERCLOS (Percentage of Eye Closure) ───────
# 슬라이딩 윈도우(초) 내 눈 감김 비율
PERCLOS_WINDOW_SEC  = 5.0   # 측정 구간
PERCLOS_WARN_RATIO  = 0.30  # 30 % → WARNING
PERCLOS_ALERT_RATIO = 0.50  # 50 % → CRITICAL

# ── MAR (Mouth Aspect Ratio) ──────────────────
# 입 닫힘: ~0.0–0.3  /  하품: > 0.5
MAR_OPEN_THRESHOLD = 0.55   # 이 값 초과 → 입 열림(하품 후보)

# ── Yawn detection ────────────────────────────
YAWN_MIN_DURATION_SEC = 1.0  # MAR 초과 지속 시간이 이 이상이면 하품 확정
YAWN_COOLDOWN_SEC     = 6.0  # 하품 인식 후 재인식까지 대기

# ── Alert ─────────────────────────────────────
ALERT_COOLDOWN_SEC = 5.0     # 같은 레벨 경보 재발생 최소 간격
ALERT_FLASH_HZ     = 2.0     # 경보 배너 깜빡임 주파수

# ── Head Pose (degrees) ────────────────────────
# 정상 전방 주시 범위 기준값
PITCH_ALERT_MAX    = 30.0    # |Pitch| 초과 → 시선 분산 시작
PITCH_CRITICAL_MAX = 55.0    # |Pitch| 초과 → 즉각 위험

YAW_ALERT_MAX      = 35.0    # |Yaw| 초과 → 시선 분산 시작
YAW_CRITICAL_MAX   = 60.0    # |Yaw| 초과 → 즉각 위험

ROLL_ALERT_MAX     = 25.0    # |Roll| 초과 → 경고 (참고용)

# ── Distraction Detection ──────────────────────
HEAD_DISTRACTION_MIN_DURATION_SEC = 2.0   # 이 시간 이상 분산 → DISTRACTED
HEAD_DISTRACTION_WINDOW_SEC       = 5.0   # 슬라이딩 윈도우 길이
HEAD_DISTRACTION_WARN_RATIO       = 0.30  # 윈도우 내 30 % → WARNING
HEAD_DISTRACTION_ALERT_RATIO      = 0.60  # 윈도우 내 60 % → CRITICAL
