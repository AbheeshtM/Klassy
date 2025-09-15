import os
import json
from sentence_transformers import SentenceTransformer, util
from config import GEMINI_API_KEY, GROQ_API_KEY, GEMINI_MODEL_NAME, GROQ_MODEL_NAME
from groq import Groq
import requests

# --- Load RAG Data ---
with open("new_rag.json", "r", encoding="utf-8") as f:
    RAG_DATA = json.load(f)

QA_TEXTS = []
for e in RAG_DATA:
    if 'question' in e and 'answer' in e:
        QA_TEXTS.append(f"{e['question']} {e['answer']}")
    elif 'title' in e and 'content' in e:
        QA_TEXTS.append(f"{e['title']} {e['content']}")
    elif 'content' in e:
        QA_TEXTS.append(e['content'])

model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Embedding Setup ---
QA_EMBEDDINGS = model.encode(QA_TEXTS, convert_to_tensor=True)

# --- Gemini Setup ---
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent"
HEADERS_GEMINI = {
    "Content-Type": "application/json",
    "x-goog-api-key": GEMINI_API_KEY
}

# --- Groq Setup ---
try:
    client_groq = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    client_groq = None

# --- Semantic RAG ---
def retrieve_relevant_chunks(query, top_k=3):
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, QA_EMBEDDINGS)[0]
    top_indices = scores.argsort(descending=True)[:top_k]
    return [QA_TEXTS[i] for i in top_indices]

def ask_gemini_with_context(query, history=None):
    chunks = retrieve_relevant_chunks(query)
    context = "\n".join(chunks)
    prompt = f"""
You are Klassy, the school receptionist at Delhi Public School Shaheedpath, Lucknow.

Be brief, polite, and speak naturally, like a real human receptionist. 
Only answer school-related queries based on the info below.
If the info isn't available, politely say so and suggest checking the school website or contact.

Context:
{context}

Question: {query}
Answer:"""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }

    try:
        response = requests.post(GEMINI_ENDPOINT, headers=HEADERS_GEMINI, json=payload)
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except:
        return None

def ask_groq_with_context(query, history=None):
    chunks = retrieve_relevant_chunks(query)
    context = "\n".join(chunks)
    if not client_groq:
        return None

    messages = [
        {"role": "system", "content": "You are a school receptionist. Use the following info to answer politely and concisely:\n" + context},
        {"role": "user", "content": query}
    ]

    try:
        response = client_groq.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=messages,
            temperature=0.5,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except:
        return None

def ask_groq_general(query, history=None):
    if not client_groq:
        return None

    messages = [
        {"role": "system", "content": "You are a helpful school receptionist. Keep responses friendly and under 2 lines."},
        {"role": "user", "content": query}
    ]

    try:
        response = client_groq.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=messages,
            temperature=0.7,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except:
        return None

# --- Response Polisher ---
def humanize_response(response, query=""):
    if not response:
        return "I'm sorry, I couldn't find that right now. You may try visiting our website or calling the school directly."

    response = response.strip()

    # Address questions – overwrite
    if any(word in query.lower() for word in ["location", "address", "where is", "kahan", "kaha"]):
        return "The school is located opposite Medanta Hospital, on Shaheed Path, Lucknow."

    # Shorten long replies to 2 lines max
    if len(response.split()) > 30:
        sentences = response.split(". ")
        return ". ".join(sentences[:2]).strip() + "."

    # Add polite closing
    if not response.endswith((".", "!", "?")):
        response += "."

    return response

# --- Final Handler ---
def ask_question(query: str, history=None) -> str:
    print(f"\n--- Querying: '{query}' ---")

    # --- Direct location intercept ---
    query_lower = query.lower()
    if any(kw in query_lower for kw in [
        "where is the school", "location of school", "how to reach", 
        "address of school", "school situated", "school location", 
        "kaha hai", "school kahan", "school address"
    ]):
        return humanize_response(None, query)

    gemini_response = ask_gemini_with_context(query)
    if gemini_response and not any(phrase in gemini_response.lower() for phrase in [
        "i don’t know", "i do not know", "not available", "check the school website"
    ]):
        return humanize_response(gemini_response, query)

    groq_rag = ask_groq_with_context(query)
    if groq_rag and "i don't know" not in groq_rag.lower():
        return humanize_response(groq_rag, query)

    groq_general = ask_groq_general(query)
    return humanize_response(groq_general, query)
