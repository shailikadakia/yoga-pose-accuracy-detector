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



