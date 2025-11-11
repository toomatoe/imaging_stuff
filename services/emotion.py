from deepface import DeepFace
import os

def start_emotion_detection_stream():
    # Create a temp folder for db_path (required by DeepFace.stream)
    temp_db = "temp_db"
    if not os.path.exists(temp_db):
        os.makedirs(temp_db)
    
    # DeepFace.stream for live emotion analysis
    DeepFace.stream(db_path=temp_db, enable_face_analysis=True)

def detect_emotion_from_webcam():
    return start_emotion_detection_stream()

# Test the emotion detection
if __name__ == "__main__":
    print("Starting emotion detection stream...")
    print("Press 'q' to quit the webcam window")
    start_emotion_detection_stream()
