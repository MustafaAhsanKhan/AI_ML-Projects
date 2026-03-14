import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import requests
import tempfile
import subprocess
from faster_whisper import WhisperModel

# ----------------------------
# CONFIG
# ----------------------------
MODEL_NAME = "llama3:8b-instruct-q4_0"
OLLAMA_URL = "http://localhost:11434/api/generate"
WHISPER_MODEL_SIZE = "base"
SAMPLE_RATE = 16000
DURATION = 5  # seconds to record per push

# Load Whisper model once
print("Loading Whisper model...")
whisper_model = WhisperModel(WHISPER_MODEL_SIZE)

def record_audio():
    print("Recording... Speak now.")
    audio = sd.rec(int(DURATION * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE,
                   channels=1,
                   dtype="int16")
    sd.wait()
    print("Recording complete.")
    return audio

def transcribe(audio):
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmpfile:
        wav.write(tmpfile.name, SAMPLE_RATE, audio)
        segments, _ = whisper_model.transcribe(tmpfile.name)
        text = " ".join([segment.text for segment in segments])
    return text

def query_ollama(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]

def speak(text):
    subprocess.run(["say", text])

# ----------------------------
# MAIN LOOP
# ----------------------------

print("Push Enter to talk. Type 'q' and press Enter to quit.")

while True:
    command = input()

    if command.lower() == "q":
        break

    audio = record_audio()
    user_text = transcribe(audio)

    print("You said:", user_text)

    if not user_text.strip():
        print("Didn't catch that.")
        continue

    reply = query_ollama(user_text)

    print("Assistant:", reply)

    speak(reply)