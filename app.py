# app.py
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from new_engine import ask_question
from memory import load_history, save_history
import os

app = FastAPI(title="Klassy Cloud API", version="1.0")

# --- Request / Response Models ---
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

# --- Optional: Load API keys from environment variables ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

if not GEMINI_API_KEY and not GROQ_API_KEY:
    print("⚠️ Warning: No API keys set. Responses may fail.")

# --- Endpoints ---
@app.get("/")
def root():
    return {"message": "Klassy Cloud API is running!"}

@app.post("/ask", response_model=QueryResponse)
def ask_query(request: QueryRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Load conversation history (optional, keeps context)
    history = load_history()
    
    try:
        answer = ask_question(query, history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

    # Save updated history
    save_history(history)

    return {"answer": answer}

