# ingest.py
# Run this ONCE to process your PDF and build the vector database.
# After running, it creates the chroma_db/ folder automatically.

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

# Load your API key from .env
load_dotenv()

# ── Step 4: Read and chunk the PDF ──────────────────────────────
print("Reading PDF...")
reader = PdfReader("docs/Cooking-Basics.pdf")
text = "\n".join(page.extract_text() for page in reader.pages)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50
)
chunks = splitter.split_text(text)
print(f"✓ Created {len(chunks)} chunks")

# ── Step 5: Create embeddings model ─────────────────────────────
print("Loading embeddings model...")
embeddings_model = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY")
)
print("✓ Embeddings model ready")

# ── Step 6: Store chunks in ChromaDB ────────────────────────────
print("Building vector database...")
vectorstore = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings_model,
    persist_directory="./chroma_db"
)
print("✓ Vector database saved to chroma_db/")
print("\nDone! You only need to run this file once.")