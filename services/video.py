import cv2

def real_time_face_detection():
   
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)

    while True:
        
        ret, frame = cap.read()
        import cv2
        import threading
        import time
        import numpy as np

        from services.emotion import analyze_frame


        def real_time_face_detection():
            # Load pre-trained face detection model
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

            # Open webcam
            cap = cv2.VideoCapture(0)

            # Shared state between main loop and analyzer thread
            latest_face = None
            latest_result = None
            lock = threading.Lock()
            running = True

'''
            def analyzer_worker():
                nonlocal latest_face, latest_result, running
                while running:
                    face = None
                    with lock:
                        if latest_face is not None:
                            face = latest_face
                            latest_face = None
                    if face is None:
                        time.sleep(0.01)
                        continue

                    try:
                        face_small = cv2.resize(face, (224, 224))
                    except Exception:
                        face_small = face

                    # This is the slow call but it's running in background thread
                    result = analyze_frame(face_small, actions=("emotion",), enforce_detection=False, silent=True)

                    with lock:
                        latest_result = result

            # Start analyzer thread
            worker = threading.Thread(target=analyzer_worker, daemon=True)
            worker.start()

            try:
                while True:
                    # Read a frame from the webcam
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Convert to grayscale (required for Haar cascades)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # Detect faces (fast)
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                    # Draw rectangles around detected faces and submit first face for analysis
                    for i, (x, y, w, h) in enumerate(faces):
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                        # Prepare face crop (BGR)
                        face_crop = frame[y : y + h, x : x + w]

                        # Submit latest face crop for analysis (non-blocking)
                        with lock:
                            latest_face = face_crop.copy()

                        # For now only submit one face per frame (the first)
                        break

                    # Overlay latest analysis result if available
                    with lock:
                        res = latest_result

                    if res:
                        # DeepFace.analyze may return a list or a dict
                        try:
                            if isinstance(res, list) and len(res) > 0:
                                r = res[0]
                            else:
                                r = res
                            emotion = r.get("dominant_emotion") if isinstance(r, dict) else str(r)
                        except Exception:
                            emotion = None

                        if emotion:
                            cv2.putText(frame, str(emotion), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                    # Display the frame
                    cv2.imshow('Face Detection', frame)

                    # Break the loop on 'q' key press
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            finally:
                running = False
                worker.join(timeout=1.0)
                # Release the webcam and close windows
                cap.release()
                cv2.destroyAllWindows()

'''
        # Call the function for testing
        if __name__ == "__main__":
            # For simple face detection + async emotion demo
            real_time_face_detection()