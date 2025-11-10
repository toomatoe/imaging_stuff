from deepface import DeepFace
import cv2
import numpy as np

def detect_emotion_from_frame(frame_bytes: bytes) -> str:
    np_arr = np.frombuffer(frame_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
    return result[0]['dominant_emotion']