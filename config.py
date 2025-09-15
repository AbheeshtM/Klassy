
# config.py

import os

# --- Gemini Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL_NAME = "gemini-1.5-flash-latest"

# --- Groq Configuration ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL_NAME = "llama3-8b-8192"  # Adjust as needed

# --- OpenWeatherMap Configuration ---
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")

# --- WIT.ai Token (if used) ---
WIT_API_TOKEN = os.getenv("WIT_API_TOKEN", "")

# --- Vosk Configuration ---
# Local path to the Vosk STT model (for Pi only; cloud API does not use this)
VOSK_MODEL_PATH = r"model/vosk-model-en-in-0.5"
