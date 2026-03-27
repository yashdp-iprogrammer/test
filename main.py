import os
from groq import Groq

# Initialize client
client = Groq(api_key="")

# Path to your Hindi audio file
audio_file_path = "/home/yashdp/Yash/test/Unlimited_calls_plan_hi.m4a"

# Open audio file
with open(audio_file_path, "rb") as file:
    transcription = client.audio.transcriptions.create(
        file=file,
        model="whisper-large-v3",   # Groq Whisper model
        response_format="json",
        language="hi"
    )

# Print result
print("Transcription:")
print(transcription.text)