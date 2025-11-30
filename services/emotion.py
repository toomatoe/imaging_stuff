import cv2
import time
import threading
from queue import Queue, Empty
from deepface import DeepFace

# Try to detect GPU availability via TensorFlow or PyTorch. If GPU is present,
# we'll analyze more frequently (possibly every frame). If not, keep a larger
# skip factor to avoid overwhelming the CPU.
GPU_AVAILABLE = False
try:
    import tensorflow as _tf
    try:
        GPU_AVAILABLE = len(_tf.config.list_physical_devices("GPU")) > 0
    except Exception:
        GPU_AVAILABLE = False
except Exception:
    # TensorFlow might not be installed in this environment; try PyTorch
    try:
        import torch as _torch

        GPU_AVAILABLE = _torch.cuda.is_available()
    except Exception:
        GPU_AVAILABLE = False

# Analysis frequency: if GPU is available, analyze more often (1-2 frames).
# On CPU we'll keep a conservative default to reduce latency.
ANALYZE_EVERY_N_FRAMES = 2 if GPU_AVAILABLE else 6  # Analyze more often if GPU is available
FRAME_RESIZE_DIM = (320, 240)  # Resize frames to 320x240 for faster processing

print(f"GPU_AVAILABLE={GPU_AVAILABLE}; ANALYZE_EVERY_N_FRAMES={ANALYZE_EVERY_N_FRAMES}")

def analyzer_worker(in_q: Queue, out_q: Queue, stop_event: threading.Event):
  
    while not stop_event.is_set():
        try:
            frame = in_q.get(timeout=0.1)
        except Empty:
            continue

        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            if result and len(result) > 0:
                dominant = result[0]['dominant_emotion']
                emotions = result[0].get('emotion', {})
                out_q.put((dominant, emotions))
        except Exception as e:
            # put an error marker or skip
            out_q.put(("error", {"error": str(e)}))
        finally:
            # mark task done if using task_done semantics
            try:
                in_q.task_done()
            except Exception:
                pass


def start_emotion_detection_stream():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    in_q = Queue(maxsize=1) 
    out_q = Queue(maxsize=2) 
    stop_event = threading.Event()

    worker = threading.Thread(target=analyzer_worker, args=(in_q, out_q, stop_event), daemon=True)
    worker.start()

    frame_count = 0
    latest_result = None
    last_analysis_time = 0.0


    EMO_TEXT_POS = (10, 30)
    EMO_FONT = cv2.FONT_HERSHEY_SIMPLEX
    EMO_FONT_SCALE = 1.0
    EMO_COLOR = (0, 255, 0)
    EMO_THICKNESS = 2

    ERROR_TEXT_POS = (10, 30)
    ERROR_FONT_SCALE = 0.7
    ERROR_COLOR = (0, 0, 255)
    ERROR_THICKNESS = 2

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Resize frame to reduce processing time
            resized_frame = cv2.resize(frame, FRAME_RESIZE_DIM)

            if frame_count % ANALYZE_EVERY_N_FRAMES == 0:
                try:
                    if in_q.full():
                        try:
                            in_q.get_nowait()
                        except Empty:
                            pass
                    in_q.put_nowait(resized_frame.copy())
                except Exception:
                    pass

            try:
                while True:  
                    latest_result = out_q.get_nowait()
            except Empty:
                pass

            # Overlay the latest_result on the frame
            if latest_result and latest_result[0] != "error":
                dominant, emotions = latest_result

                conf = emotions.get(dominant, 0.0)
                # use named overlay variables for text positioning and style
                cv2.putText(
                    frame,
                    f"{dominant}: {conf:.1f}%",
                    EMO_TEXT_POS,
                    EMO_FONT,
                    EMO_FONT_SCALE,
                    EMO_COLOR,
                    EMO_THICKNESS,
                )
            elif latest_result and latest_result[0] == "error":
                err = latest_result[1].get("error", "")
                cv2.putText(
                    frame,
                    "Analyze error",
                    ERROR_TEXT_POS,
                    EMO_FONT,
                    ERROR_FONT_SCALE,
                    ERROR_COLOR,
                    ERROR_THICKNESS,
                )

         

            cv2.imshow("Emotion Detection (press q to quit)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        stop_event.set()
        cap.release()
        cv2.destroyAllWindows()
