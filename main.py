from fastapi import FastAPI, UploadFile, File
from models import Context
from services import emotion, audio, response

app = FastAPI()
#Need to do the fastAPi wiring for backend testing 
@app.post("/detect_emotion/")
async def detect_emotion(image: UploadFile = File(...)):
    contents = await image.read()
    detected = emotion.detect_emotion_from_frame(contents)
    return {"emotion": detected}

@app.post("/analyze_audio/")
async def analyze_audio(audio_file: UploadFile = File(...)):
    path = f"temp_audio.wav"
    with open(path, "wb") as f:
        f.write(await audio_file.read())
    result = audio.analyze_audio(path)
    return result

@app.post("/generate_response/")
async def generate_response(context: Context):
    reply = response.generate_response(context)
    return {"response": reply}