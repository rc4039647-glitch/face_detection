

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

print("Python version:", sys.version.split()[0])
print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)

import sklearn
print("scikit-learn version:", sklearn.__version__)

import tensorflow as tf
print("TensorFlow version:", tf.__version__)

import cv2
print("OpenCV version:", cv2.__version__)

gpu_devices = tf.config.list_physical_devices("GPU")
if gpu_devices:
    print(f"\n✅ GPU available: {gpu_devices[0].name} — Project 2 will run faster.")
else:
    print("\nℹ️ No GPU detected — everything will still run, just a bit slower on Project 2.")
    print("   In Colab: Runtime → Change runtime type → Hardware accelerator → GPU")

print("\nAll core libraries imported successfully. You're ready to go! 🚀")

import matplotlib.pyplot as plt

print(cv2.__file__)
print(cv2.__version__)
print(hasattr(cv2, "CascadeClassifier"))


print("OpenCV version:", cv2.__version__)

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

face_detector = cv2.CascadeClassifier(cascade_path)

print("Face detector loaded:", not face_detector.empty())

import os
import urllib.request

FALLBACK_FACE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
FALLBACK_FACE_PATH = "sample_face.jpg"
face_image_path = None

# Option A: upload your own image (works in Google Colab)
try:
    from google.colab import files
    print("📤 Upload an image with one or more faces. Cancel/skip to use a sample image instead.")
    uploaded = files.upload()
    if uploaded:
        face_image_path = list(uploaded.keys())[0]
except ImportError:
    pass  # not running in Colab

# Option B: guaranteed fallback so the demo always works
if not face_image_path:
    if not os.path.exists(FALLBACK_FACE_PATH):
        urllib.request.urlretrieve(FALLBACK_FACE_URL, FALLBACK_FACE_PATH)
    face_image_path = FALLBACK_FACE_PATH
    print(f"ℹ️ Using sample image: {face_image_path} (re-run this cell to upload your own, or use the webcam cell below)")

face_image = cv2.imread(face_image_path)
print("Image loaded:", face_image_path, "| Shape:", face_image.shape if face_image is not None else None)

def capture_webcam_photo(filename="webcam_photo.jpg", quality=0.8):
    """Captures one photo from the webcam inside Google Colab and saves it locally."""
    from IPython.display import display, Javascript
    from google.colab.output import eval_js
    from base64 import b64decode

    js = Javascript("""
        async function takePhoto(quality) {
            const div = document.createElement('div');
            const capture = document.createElement('button');
            capture.textContent = 'Capture';
            div.appendChild(capture);

            const video = document.createElement('video');
            video.style.display = 'block';
            const stream = await navigator.mediaDevices.getUserMedia({video: true});

            document.body.appendChild(div);
            div.appendChild(video);
            video.srcObject = stream;
            await video.play();

            google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);
            await new Promise((resolve) => capture.onclick = resolve);

            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);
            stream.getVideoTracks()[0].stop();
            div.remove();
            return canvas.toDataURL('image/jpeg', quality);
        }
    """)
    display(js)
    data = eval_js("takePhoto({})".format(quality))
    binary = b64decode(data.split(",")[1])
    with open(filename, "wb") as f:
        f.write(binary)
    return filename

try:
    face_image_path = capture_webcam_photo()
    face_image = cv2.imread(face_image_path)
    print("Webcam photo captured:", face_image_path)
except Exception:
    print("Webcam capture only works in Google Colab with camera permission granted.")
    print("Skipping — the image from the previous cell will be used instead.")

    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

faces = face_detector.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)
if face_image is not None:
     output_image = face_image.copy()
for (x, y, w, h) in faces:
        cv2.rectangle(
            output_image,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

output_rgb = cv2.cvtColor(
        output_image,
        cv2.COLOR_BGR2RGB
    )

plt.figure(figsize=(10, 7))
plt.imshow(output_rgb)
plt.axis("off")
plt.title(f"Detected Faces: {len(faces)}")
plt.show()

