from deepface import DeepFace
import os
import logging

logger = logging.getLogger(__name__)

def start_emotion_detection_stream():
    # Create a temp folder for db_path (required by DeepFace.stream)
    temp_db = "temp_db"
    if not os.path.exists(temp_db):
        os.makedirs(temp_db)
    
    # DeepFace.stream for live emotion analysis
    DeepFace.stream(db_path=temp_db, enable_face_analysis=True)

def detect_emotion_from_webcam():
    return start_emotion_detection_stream()


def analyze_frame(frame, actions=("emotion",), enforce_detection=False, detector_backend="opencv", silent=True):
        try:
                result = DeepFace.analyze(
                        frame,
                        actions=actions,
                        enforce_detection=enforce_detection,
                        detector_backend=detector_backend,
                        silent=silent,
                )
                return result
        except Exception as e:
                logger.exception("DeepFace.analyze failed")
                return {"error": str(e)}

# Test the emotion detection
if __name__ == "__main__":
    print("Starting emotion detection stream...")
    print("Press 'q' to quit the webcam window")
    start_emotion_detection_stream()
