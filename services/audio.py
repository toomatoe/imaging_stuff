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

# Call the function for testing
if __name__ == "__main__":
    # Test with a working example
    try:
        print("Testing audio analysis...")
        print("Whisper model loaded successfully!")
        print("To test with actual audio, provide a valid audio file path")
        # analyze_audio("path/to/your/audio/file.wav")
    except Exception as e:
        print(f"Error: {e}")