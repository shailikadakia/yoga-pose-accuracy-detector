import math

def normalize_landmarks(landmarks):

  if not landmarks or len(landmarks) != 33:
    return None
  
  left_shoulder = landmarks[11]
  right_shoulder = landmarks[12]
  scale = math.dist([left_shoulder.x, left_shoulder.y], [right_shoulder.x, right_shoulder.y])

  left_hip = landmarks[23]
  right_hip = landmarks[24]
  mid_hip_x = (left_hip.x + right_hip.x) // 2
  mid_hip_y = (left_hip.y + right_hip.y) // 2

  normalized_landmarks = []
  for landmark in landmarks:
    normalized_landmarks.append({
      'x': round(((landmark.x - mid_hip_x) / scale), 6),
      'y': round(((landmark.y - mid_hip_y) / scale), 6),
      'z': round((landmark.z / scale), 6),
      'visibility': round(landmark.visibility, 6),
    })

    
  return normalized_landmarks

def _as_xyz(p):
    """Return (x,y,z) from MediaPipe landmark, dict, or (x,y,z) tuple/list."""
    if hasattr(p, "x"):                  # MediaPipe NormalizedLandmark
        return float(p.x), float(p.y), float(p.z)
    if isinstance(p, dict):              # {'x':..,'y':..,'z':..}
        return float(p["x"]), float(p["y"]), float(p["z"])
    # assume tuple/list [x, y, z]
    return float(p[0]), float(p[1]), float(p[2])


def calculate_angle_between_landmarks(landmark1, landmark2, landmark3):
  x1, y1, _ = _as_xyz(landmark1)
  x2, y2, _ = _as_xyz(landmark2)
  x3, y3, _ = _as_xyz(landmark3)

  line1 = math.atan2(x1 - x2, y1 - y2)
  line2 = math.atan2(x3 - x2, y3 - y2)

  angle = math.degrees(line1 -  line2)

  if angle < 0:
    return angle + 360
  return angle


angle = calculate_angle_between_landmarks((558, 326, 0), (642, 333, 0), (718, 321, 0))

# Display the calculated angle.
print(f'The calculated angle is {angle}')