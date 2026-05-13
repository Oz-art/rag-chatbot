# main.py
# Exposes your RAG chatbot as a REST API
# Run with: uvicorn main:app --reload

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import get_answer  # ← imports your working rag.py

app = FastAPI(
    title="RAG Chatbot API",
    description="Ask questions about your document",
    version="1.0.0"
)

# ── Allow frontend (Streamlit) to talk to this API ──────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Define what the request body looks like ─────────────────────
class Question(BaseModel):
    query: str

# ── Health check endpoint ────────────────────────────────────────
# Visit http://localhost:8000/health to confirm API is running
@app.get("/health")
def health():
    return {"status": "ok"}

# ── Main endpoint — ask a question ──────────────────────────────
@app.post("/ask")
def ask(q: Question):
    result = get_answer(q.query)
    return {
        "answer": result["answer"],
        "source_chunks": result["source_chunks"]
    }