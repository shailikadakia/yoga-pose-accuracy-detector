# yoga-pose-accuracy-detector

Installation Instructions

`pip install mediapipe opencv-python scikit-learn matplotlib`

If there are issues with media pipe versions, create a conda environment:
`conda create -n mediapipe-env python=3.9 -y`
`conda activate mediapipe-env`
`pip uninstall numpy `
`pip install "numpy<2.0"`
`pip install mediapipe`
`pip install opencv-python-headless==4.7.0.72`
`python -c "import cv2, mediapipe, numpy; print('✅ All good:', numpy.__version__)"`
Should see the below printed in terminal:
`✅ All good: 1.26.4`

