import librosa
import numpy as np
import whisper

whisper_model = whisper.load_model("base")

def analyze_audio(file_path: str):
    # Transcription
    transcription = whisper_model.transcribe(file_path)["text"]

    # Basic audio features
    y, sr = librosa.load(file_path)
    energy = float(np.mean(librosa.feature.rms(y=y)))

    return {
        "transcript": transcription,
        "energy": energy
    }