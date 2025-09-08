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
L_FI,       R_FI       = 31, 32        
NOSE                      = 0

def _angle_3pt(a, b, c):
    """
    Angle at b formed by a-b-c (degrees in [0, 180]).
    a,b,c are 2D points (x,y) or 3D (x,y,z)
    only x,y are used.
    """
    ax, ay = a[0], a[1]
    bx, by = b[0], b[1]
    cx, cy = c[0], c[1]
    v1x, v1y = ax - bx, ay - by
    v2x, v2y = cx - bx, cy - by
    ang1 = math.atan2(v1y, v1x)  
    ang2 = math.atan2(v2y, v2x)
    deg = abs(math.degrees(ang1 - ang2))
    if deg > 180:
        return 360 - deg
    return deg

def _orient_deg(p, q):
    """
    Orientation of the line p->q in degrees modulo 180 (0° = horizontal, 90° = vertical).
    """
    dx, dy = q[0] - p[0], q[1] - p[1]
    return (math.degrees(math.atan2(dy, dx)) + 360.0) % 180.0

def _tilt_vs_horizontal(p, q):
    '''
    How far a line deviates from horizontal, in [0..90].
    '''
    o = _orient_deg(p, q)
    if o <= 90:
        return 0
    return 180 - o

def _tilt_vs_vertical(p, q):
    '''
    How far a line deviates from vertical, in [0..90]. 
    0 = perfectly vertical/upright
    90 = horizontal
    '''
    return abs(90.0 - _tilt_vs_horizontal(p, q))

def _angle_diff(a, b):
    '''
    Minimal difference between two orientations in degrees,
    where orientations are modulo 180 (lines w/o direction).
    '''
    d = abs(a - b)
    return d if d <= 90 else 180 - d

def _dist(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])


def row_to_points33(row: pd.Series) -> np.ndarray:
    '''
    Rebuild a (33,3) array from CSV columns x0..z32.
    '''
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
    ''''
    Compute feature vector from 33 landmarks.
    Inputs:
        pts33: (33,3) or (33,2) array-like of landmark coords in the SAME scale (MediaPipe normalized or pixels).
               Only x,y are used for angles and distances.
    Returns:
        1D np.array with len(FEATURE_NAMES).
    '''
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

    arm_elev_L = _tilt_vs_horizontal(pts2[L_SHOULDER], pts2[L_WRIST]) 
    arm_elev_R = _tilt_vs_horizontal(pts2[R_SHOULDER], pts2[R_WRIST])

    mid_hip      = ((pts2[L_HIP][0] + pts2[R_HIP][0]) / 2.0, (pts2[L_HIP][1] + pts2[R_HIP][1]) / 2.0)
    mid_shoulder = ((pts2[L_SHOULDER][0] + pts2[R_SHOULDER][0]) / 2.0, (pts2[L_SHOULDER][1] + pts2[R_SHOULDER][1]) / 2.0)

    torso_tilt   = _tilt_vs_vertical(mid_hip, mid_shoulder)         
    pelvis_tilt  = _tilt_vs_horizontal(pts2[L_HIP], pts2[R_HIP])    
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

    hip_above_shoulders = 1.0 if mid_hip[1] < mid_shoulder[1] else 0.0   
    head_below_hips     = 1.0 if pts2[NOSE][1] > mid_hip[1] else 0.0

    head_pitch = _tilt_vs_vertical(mid_shoulder, pts2[NOSE])  

    return np.array([
        L_elbow, R_elbow, L_shldr, R_shldr, L_hip, R_hip, L_knee, R_knee, L_ankle, R_ankle,
        arm_elev_L, arm_elev_R,
        torso_tilt, pelvis_tilt, shoulder_tilt, shoulder_vs_hip_twist,
        feet_over_shoulder, knee_spread_over_shoulder,
        knee_asymmetry, arm_asymmetry,
        hip_above_shoulders, head_below_hips,
        head_pitch
    ], dtype=float)
