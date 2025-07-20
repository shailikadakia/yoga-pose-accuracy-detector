# yoga-pose-accuracy-detector

Installation Instructions

`pip install mediapipe opencv-python scikit-learn matplotlib`

If there are issues with media pipe versions, create a conda environment:
1. `conda create -n mediapipe-env python=3.9 -y`
2. `conda activate mediapipe-env`
3. `pip uninstall numpy `
4. `pip install "numpy<2.0"`
5. `pip install mediapipe`
6. `pip install opencv-python-headless==4.7.0.72`
7. `python -c "import cv2, mediapipe, numpy; print('✅ All good:', numpy.__version__)"`
Should see the below printed in terminal:
`✅ All good: 1.26.4`

