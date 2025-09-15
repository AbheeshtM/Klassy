import json
import os

# 📁 File where conversation history is stored
HISTORY_FILE = "chat_history.json"

def load_history():
    """
    Load previous conversation history from a JSON file.
    Returns an empty list if the file doesn't exist or is invalid.
    """
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []

def save_history(history):
    """
    Save the current conversation history to a JSON file.
    """
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
