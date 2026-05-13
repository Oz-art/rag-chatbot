# RAG Document Chatbot

A document Q&A chatbot built with Retrieval-Augmented Generation (RAG).
Ask questions about any PDF document and get accurate answers with source references.

## Tech Stack
- **LangChain** — RAG pipeline and LLM orchestration
- **ChromaDB** — vector database for semantic search
- **OpenAI** — embeddings and LLM (gpt-4o-mini)
- **FastAPI** — REST API endpoint
- **Streamlit** — chat UI
- **Docker** — containerisation

## How it works
1. PDF is chunked and converted to vector embeddings
2. Embeddings are stored in ChromaDB
3. User question is converted to a vector and matched against stored chunks
4. Top 3 most relevant chunks are sent to the LLM with the question
5. LLM answers based only on the retrieved chunks

## Run locally
1. Clone the repo
2. Create a `.env` file with your `OPENAI_API_KEY`
3. Add your PDF to `docs/`
4. Run:
   pip install -r requirements.txt
   python ingest.py
   streamlit run app.py