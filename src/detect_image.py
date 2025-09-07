import mediapipe as mp 
import cv2
import matplotlib.pyplot as plt

# Initializing mediapipe drawing class 
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

def detect_pose(image, display=True, return_results=False):
    if image is None:   
        raise ValueError("Image is None. Check your path/filename.")

    h, w, _ = image.shape
    img_copy = image.copy()

    # Show the input image (optional)
    if display:
        plt.figure(figsize=(6, 6))
        plt.title("Sample Image"); plt.axis('off')
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()

    # Run pose
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    landmarks = []
    if results.pose_landmarks:
        # Draw 2D landmarks on the image
        mp_drawing.draw_landmarks(
            image=img_copy,
            landmark_list=results.pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS
        )

        # Collect per-landmark pixel coords + z
        for lm in results.pose_landmarks.landmark:   # <-- .landmark
            x_px = int(lm.x * w)
            y_px = int(lm.y * h)
            z_val = lm.z                 # keep as float
            landmarks.append((x_px, y_px, z_val))

    if display:
        # Show the 2D annotated image
        plt.figure(figsize=(6, 6))
        plt.title("Output (2D)"); plt.axis('off')
        plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
        plt.show()

        # Show the 3D world landmark plot (if available)
        if results.pose_world_landmarks:
            mp_drawing.plot_landmarks(
                results.pose_world_landmarks,
                mp_pose.POSE_CONNECTIONS
            )
            plt.show()   
    if return_results:
        return results
    return landmarks

# image = '../training-data/boat/boat_1.png'
# img = cv2.imread(image)  

# print(detect_pose(img, False))