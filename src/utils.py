import math
import numpy as np
import pandas as pd

'''
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW,    R_ELBOW    = 13, 14
L_WRIST,    R_WRIST    = 15, 16
L_HIP,      R_HIP      = 23, 24
L_KNEE,     R_KNEE     = 25, 26
L_ANKLE,    R_ANKLE    = 27, 28
L_HEEL,     R_HEEL     = 29, 30
L_FI,       R_FI       = 31, 32

# -------- normalizer (optional) ----------
def normalize_landmarks(landmarks):
    if hasattr(landmarks, "landmark"):
        landmarks = landmarks.landmark
    if not landmarks or len(landmarks) != 33:
        return None

    ls, rs = landmarks[L_SHOULDER], landmarks[R_SHOULDER]
    scale = math.dist([ls.x, ls.y], [rs.x, rs.y]) or 1e-6  # avoid div/0

    lh, rh = landmarks[L_HIP], landmarks[R_HIP]
    mid_hip_x = (lh.x + rh.x) / 2.0      # use float division, not //
    mid_hip_y = (lh.y + rh.y) / 2.0

    out = []
    for lm in landmarks:
        out.append({
            "x": round((lm.x - mid_hip_x) / scale, 6),
            "y": round((lm.y - mid_hip_y) / scale, 6),
            "z": round(lm.z / scale, 6),
            "visibility": round(float(getattr(lm, "visibility", 1.0)), 6),
        })
    return out

# -------- helpers used by angle features ----------
def _as_xyz(p):
    """Return (x,y,z) from MediaPipe landmark, dict, or (x,y[,z]) list/array."""
    if hasattr(p, "x"):            # MediaPipe NormalizedLandmark
        return float(p.x), float(p.y), float(getattr(p, "z", 0.0))
    if isinstance(p, dict):        # {'x':..,'y':..,'z':..}
        return float(p["x"]), float(p["y"]), float(p.get("z", 0.0))
    arr = np.asarray(p, dtype=float)
    if arr.shape[0] >= 3:
        return float(arr[0]), float(arr[1]), float(arr[2])
    elif arr.shape[0] == 2:
        return float(arr[0]), float(arr[1]), 0.0  # treat 2D as z=0
    raise ValueError("Point must have length 2 or 3")

def calculate_angle_between_landmarks(a, b, c):
    """
    Angle at point b formed by a-b-c in degrees, range [0, 180].
    Accepts MediaPipe landmarks, dicts, or (x,y[,z]) arrays.
    """
    x1, y1, _ = _as_xyz(a)
    x2, y2, _ = _as_xyz(b)
    x3, y3, _ = _as_xyz(c)

    # IMPORTANT: atan2(dy, dx)
    ang1 = math.atan2(y1 - y2, x1 - x2)
    ang2 = math.atan2(y3 - y2, x3 - x2)
    deg = abs(math.degrees(ang1 - ang2))
    return 360 - deg if deg > 180 else deg

def calculate_distance(p, q):
    p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
    return math.hypot(p[0] - q[0], p[1] - q[1])

def compute_feature_vector(points33):
    """
    points33: (33,3) or (33,2) array-like of landmark coords in the SAME scale.
    Returns (feat_vector, feat_names)
    """
    pts = np.asarray(points33, dtype=float)
    if pts.shape[1] not in (2, 3):
        raise ValueError("points33 must have shape (33,2) or (33,3)")

    # If (33,3), we’ll just use x,y for angles below; _as_xyz already accepts 2D.
    if pts.shape[1] == 3:
        pts2 = pts[:, :2]
    else:
        pts2 = pts

    # core angles
    left_elbow   = calculate_angle_between_landmarks(pts2[L_SHOULDER], pts2[L_ELBOW],  pts2[L_WRIST])
    right_elbow  = calculate_angle_between_landmarks(pts2[R_SHOULDER], pts2[R_ELBOW],  pts2[R_WRIST])
    left_shoulder  = calculate_angle_between_landmarks(pts2[L_ELBOW],   pts2[L_SHOULDER], pts2[L_HIP])
    right_shoulder = calculate_angle_between_landmarks(pts2[R_ELBOW],   pts2[R_SHOULDER], pts2[R_HIP])
    left_hip     = calculate_angle_between_landmarks(pts2[L_SHOULDER], pts2[L_HIP],    pts2[L_KNEE])
    right_hip    = calculate_angle_between_landmarks(pts2[R_SHOULDER], pts2[R_HIP],    pts2[R_KNEE])
    left_knee    = calculate_angle_between_landmarks(pts2[L_HIP],      pts2[L_KNEE],   pts2[L_ANKLE])
    right_knee   = calculate_angle_between_landmarks(pts2[R_HIP],      pts2[R_KNEE],   pts2[R_ANKLE])
    left_ankle   = calculate_angle_between_landmarks(pts2[L_KNEE],     pts2[L_ANKLE],  pts2[L_FI])
    right_ankle  = calculate_angle_between_landmarks(pts2[R_KNEE],     pts2[R_ANKLE],  pts2[R_FI])

    # posture cues (use 2D distances)
    mid_hip      = ( (pts2[L_HIP][0]+pts2[R_HIP][0])/2, (pts2[L_HIP][1]+pts2[R_HIP][1])/2 )
    mid_shoulder = ( (pts2[L_SHOULDER][0]+pts2[R_SHOULDER][0])/2, (pts2[L_SHOULDER][1]+pts2[R_SHOULDER][1])/2 )
    torso_vec    = (mid_shoulder[0]-mid_hip[0], mid_shoulder[1]-mid_hip[1])
    torso_tilt   = abs(math.degrees(math.atan2(torso_vec[0], -torso_vec[1])))

    shoulder_w   = calculate_distance(pts2[L_SHOULDER], pts2[R_SHOULDER]) + 1e-6
    feet_w       = calculate_distance(pts2[L_ANKLE],    pts2[R_ANKLE])
    feet_over_shoulder = feet_w / shoulder_w

    left_hand_up  = 1.0 if pts2[L_WRIST][1] < pts2[L_SHOULDER][1] else 0.0
    right_hand_up = 1.0 if pts2[R_WRIST][1] < pts2[R_SHOULDER][1] else 0.0

    feat_names = [
        "L_elbow","R_elbow","L_shoulder","R_shoulder","L_hip","R_hip",
        "L_knee","R_knee","L_ankle","R_ankle",
        "torso_tilt","feet_over_shoulder","L_hand_up","R_hand_up"
    ]
    feats = [
        left_elbow, right_elbow, left_shoulder, right_shoulder, left_hip, right_hip,
        left_knee, right_knee, left_ankle, right_ankle,
        torso_tilt, feet_over_shoulder, left_hand_up, right_hand_up
    ]
    return np.array(feats, dtype=float), feat_names
'''

'''
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW,    R_ELBOW    = 13, 14
L_WRIST,    R_WRIST    = 15, 16
L_HIP,      R_HIP      = 23, 24
L_KNEE,     R_KNEE     = 25, 26
L_ANKLE,    R_ANKLE    = 27, 28
L_FI,       R_FI       = 31, 32

def _angle(a, b, c):
    ax, ay = a[0], a[1]; bx, by = b[0], b[1]; cx, cy = c[0], c[1]
    ang1 = math.atan2(ay - by, ax - bx)  # atan2(dy, dx)
    ang2 = math.atan2(cy - by, cx - bx)
    deg = abs(math.degrees(ang1 - ang2))
    return 360 - deg if deg > 180 else deg

def _dist(p, q):
    return math.hypot(p[0]-q[0], p[1]-q[1])

FEATURE_NAMES = [
    "L_elbow","R_elbow","L_shoulder","R_shoulder","L_hip","R_hip",
    "L_knee","R_knee","L_ankle","R_ankle",
    "torso_tilt","feet_over_shoulder","L_hand_up","R_hand_up"
]

def row_to_points33(row: pd.Series) -> np.ndarray:
    """Rebuild (33,3) from columns x0..z32, guaranteed numeric by earlier coercion."""
    pts = np.empty((33, 3), dtype=float)
    for i in range(33):
        pts[i, 0] = row[f"x{i}"]
        pts[i, 1] = row[f"y{i}"]
        pts[i, 2] = row[f"z{i}"]
    return pts

def compute_feature_vector_from_points(pts33: np.ndarray) -> np.ndarray:
    pts = pts33[:, :2] if pts33.shape[1] == 3 else pts33  # use x,y for angles

    left_elbow   = _angle(pts[L_SHOULDER], pts[L_ELBOW],  pts[L_WRIST])
    right_elbow  = _angle(pts[R_SHOULDER], pts[R_ELBOW],  pts[R_WRIST])
    left_shoulder  = _angle(pts[L_ELBOW],   pts[L_SHOULDER], pts[L_HIP])
    right_shoulder = _angle(pts[R_ELBOW],   pts[R_SHOULDER], pts[R_HIP])
    left_hip     = _angle(pts[L_SHOULDER], pts[L_HIP],    pts[L_KNEE])
    right_hip    = _angle(pts[R_SHOULDER], pts[R_HIP],    pts[R_KNEE])
    left_knee    = _angle(pts[L_HIP],      pts[L_KNEE],   pts[L_ANKLE])
    right_knee   = _angle(pts[R_HIP],      pts[R_KNEE],   pts[R_ANKLE])
    left_ankle   = _angle(pts[L_KNEE],     pts[L_ANKLE],  pts[L_FI])
    right_ankle  = _angle(pts[R_KNEE],     pts[R_ANKLE],  pts[R_FI])

    mid_hip      = ((pts[L_HIP][0]+pts[R_HIP][0])/2, (pts[L_HIP][1]+pts[R_HIP][1])/2)
    mid_shoulder = ((pts[L_SHOULDER][0]+pts[R_SHOULDER][0])/2, (pts[L_SHOULDER][1]+pts[R_SHOULDER][1])/2)
    torso_vec    = (mid_shoulder[0]-mid_hip[0], mid_shoulder[1]-mid_hip[1])
    torso_tilt   = abs(math.degrees(math.atan2(torso_vec[0], -torso_vec[1])))

    shoulder_w   = _dist(pts[L_SHOULDER], pts[R_SHOULDER]) + 1e-6
    feet_w       = _dist(pts[L_ANKLE],    pts[R_ANKLE])
    feet_over_shoulder = feet_w / shoulder_w

    left_hand_up  = 1.0 if pts[L_WRIST][1] < pts[L_SHOULDER][1] else 0.0
    right_hand_up = 1.0 if pts[R_WRIST][1] < pts[R_SHOULDER][1] else 0.0

    return np.array([
        left_elbow, right_elbow, left_shoulder, right_shoulder, left_hip, right_hip,
        left_knee, right_knee, left_ankle, right_ankle,
        torso_tilt, feet_over_shoulder, left_hand_up, right_hand_up
    ], dtype=float)
'''
import math
import numpy as np
import pandas as pd

# MediaPipe landmark indices 
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW,    R_ELBOW    = 13, 14
L_WRIST,    R_WRIST    = 15, 16
L_HIP,      R_HIP      = 23, 24
L_KNEE,     R_KNEE     = 25, 26
L_ANKLE,    R_ANKLE    = 27, 28
L_FI,       R_FI       = 31, 32        # foot index (big toe)
NOSE                      = 0

def _angle_3pt(a, b, c):
    """
    Angle at b formed by a-b-c (degrees in [0, 180]).
    a,b,c are 2D points (x,y) or 3D (x,y,z); only x,y are used.
    """
    ax, ay = a[0], a[1]
    bx, by = b[0], b[1]
    cx, cy = c[0], c[1]
    v1x, v1y = ax - bx, ay - by
    v2x, v2y = cx - bx, cy - by
    ang1 = math.atan2(v1y, v1x)  
    ang2 = math.atan2(v2y, v2x)
    deg = abs(math.degrees(ang1 - ang2))
    return 360 - deg if deg > 180 else deg

def _orient_deg(p, q):
    """
    Orientation of the line p->q in degrees modulo 180 (0° = horizontal, 90° = vertical).
    """
    dx, dy = q[0] - p[0], q[1] - p[1]
    return (math.degrees(math.atan2(dy, dx)) + 360.0) % 180.0

def _tilt_vs_horizontal(p, q):
    """How far a line deviates from horizontal, in [0..90]."""
    o = _orient_deg(p, q)
    return o if o <= 90 else 180 - o

def _tilt_vs_vertical(p, q):
    """How far a line deviates from vertical, in [0..90]. 0 = perfectly vertical/upright."""
    return abs(90.0 - _tilt_vs_horizontal(p, q))

def _angle_diff(a, b):
    """
    Minimal difference between two orientations in degrees,
    where orientations are modulo 180 (lines w/o direction).
    """
    d = abs(a - b)
    return d if d <= 90 else 180 - d

def _dist(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])


def row_to_points33(row: pd.Series) -> np.ndarray:
    """
    Rebuild a (33,3) array from CSV columns x0..z32.
    Assumes columns are numeric; coerce before calling (see make_angles_csv.py).
    """
    pts = np.empty((33, 3), dtype=float)
    for i in range(33):
        pts[i, 0] = row[f"x{i}"]
        pts[i, 1] = row[f"y{i}"]
        pts[i, 2] = row[f"z{i}"]
    return pts

FEATURE_NAMES = [
    # 10 joint angles (0–180°)
    "L_elbow", "R_elbow", "L_shoulder", "R_shoulder",
    "L_hip", "R_hip", "L_knee", "R_knee", "L_ankle", "R_ankle",
    # 2 arm elevation angles vs horizontal (0=out to side, 90=overhead)
    "arm_elev_L", "arm_elev_R",
    # global posture & stance
    "torso_tilt", "pelvis_tilt", "shoulder_tilt", "shoulder_vs_hip_twist",
    "feet_over_shoulder", "knee_spread_over_shoulder",
    # asymmetry
    "knee_asymmetry", "arm_asymmetry",
    # inversion/fold booleans
    "hip_above_shoulders", "head_below_hips",
    # head/neck
    "head_pitch"
]

def compute_feature_vector_from_points(pts33: np.ndarray) -> np.ndarray:
    """
    Compute a compact, robust feature vector from 33 landmarks.
    Inputs:
        pts33: (33,3) or (33,2) array-like of landmark coords in the SAME scale (MediaPipe normalized or pixels).
               Only x,y are used for angles and distances.
    Returns:
        1D np.array with len(FEATURE_NAMES).
    """
    pts = np.asarray(pts33, dtype=float)
    if pts.shape[1] == 3:
        pts2 = pts[:, :2]
    elif pts.shape[1] == 2:
        pts2 = pts
    else:
        raise ValueError("pts33 must have shape (33,2) or (33,3)")

    L_elbow  = _angle_3pt(pts2[L_SHOULDER], pts2[L_ELBOW],  pts2[L_WRIST])
    R_elbow  = _angle_3pt(pts2[R_SHOULDER], pts2[R_ELBOW],  pts2[R_WRIST])
    L_shldr  = _angle_3pt(pts2[L_ELBOW],    pts2[L_SHOULDER], pts2[L_HIP])
    R_shldr  = _angle_3pt(pts2[R_ELBOW],    pts2[R_SHOULDER], pts2[R_HIP])
    L_hip    = _angle_3pt(pts2[L_SHOULDER], pts2[L_HIP],     pts2[L_KNEE])
    R_hip    = _angle_3pt(pts2[R_SHOULDER], pts2[R_HIP],     pts2[R_KNEE])
    L_knee   = _angle_3pt(pts2[L_HIP],      pts2[L_KNEE],    pts2[L_ANKLE])
    R_knee   = _angle_3pt(pts2[R_HIP],      pts2[R_KNEE],    pts2[R_ANKLE])
    L_ankle  = _angle_3pt(pts2[L_KNEE],     pts2[L_ANKLE],   pts2[L_FI])
    R_ankle  = _angle_3pt(pts2[R_KNEE],     pts2[R_ANKLE],   pts2[R_FI])

    arm_elev_L = _tilt_vs_horizontal(pts2[L_SHOULDER], pts2[L_WRIST])  # ~0: out to side, ~90: overhead
    arm_elev_R = _tilt_vs_horizontal(pts2[R_SHOULDER], pts2[R_WRIST])

    mid_hip      = ((pts2[L_HIP][0] + pts2[R_HIP][0]) / 2.0, (pts2[L_HIP][1] + pts2[R_HIP][1]) / 2.0)
    mid_shoulder = ((pts2[L_SHOULDER][0] + pts2[R_SHOULDER][0]) / 2.0, (pts2[L_SHOULDER][1] + pts2[R_SHOULDER][1]) / 2.0)

    torso_tilt   = _tilt_vs_vertical(mid_hip, mid_shoulder)         # 0=upright, ~90=horizontal (boat/side plank)
    pelvis_tilt  = _tilt_vs_horizontal(pts2[L_HIP], pts2[R_HIP])    # level pelvis ~0°
    shoulder_tilt= _tilt_vs_horizontal(pts2[L_SHOULDER], pts2[R_SHOULDER])

    hip_orient   = _orient_deg(pts2[L_HIP], pts2[R_HIP])
    sh_orient    = _orient_deg(pts2[L_SHOULDER], pts2[R_SHOULDER])
    shoulder_vs_hip_twist = _angle_diff(sh_orient, hip_orient)

    shoulder_w   = _dist(pts2[L_SHOULDER], pts2[R_SHOULDER]) + 1e-6
    feet_w       = _dist(pts2[L_ANKLE],    pts2[R_ANKLE])
    knees_w      = _dist(pts2[L_KNEE],     pts2[R_KNEE])

    feet_over_shoulder       = feet_w  / shoulder_w
    knee_spread_over_shoulder= knees_w / shoulder_w

    knee_asymmetry = abs(L_knee - R_knee)
    arm_asymmetry  = abs(L_shldr - R_shldr)

    hip_above_shoulders = 1.0 if mid_hip[1] < mid_shoulder[1] else 0.0   # image y grows down; smaller y = higher
    head_below_hips     = 1.0 if pts2[NOSE][1] > mid_hip[1] else 0.0

    head_pitch = _tilt_vs_vertical(mid_shoulder, pts2[NOSE])  # 0=vertical head/neck; larger when looking up/down

    return np.array([
        L_elbow, R_elbow, L_shldr, R_shldr, L_hip, R_hip, L_knee, R_knee, L_ankle, R_ankle,
        arm_elev_L, arm_elev_R,
        torso_tilt, pelvis_tilt, shoulder_tilt, shoulder_vs_hip_twist,
        feet_over_shoulder, knee_spread_over_shoulder,
        knee_asymmetry, arm_asymmetry,
        hip_above_shoulders, head_below_hips,
        head_pitch
    ], dtype=float)
