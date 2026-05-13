# app.py
# The chat UI for your RAG chatbot
# Run with: streamlit run app.py

import streamlit as st
from rag import get_answer  # ← imports your working rag.py

# ── Page config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Document Chatbot",
    page_icon="🤖",
    layout="centered"
)

# ── Title and description ────────────────────────────────────────
st.title("🤖 Document Chatbot")
st.caption("Ask anything about your document. Powered by RAG + OpenAI.")
st.caption("Created by Oz-art")
st.divider()

# ── Initialize chat history ─────────────────────────────────────
# st.session_state keeps the chat history alive while the app runs
# Every time you ask a question, Streamlit reruns the whole script
# session_state is how we remember previous messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Display chat history ─────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        # If it's an assistant message, show source chunks in expander
        if msg["role"] == "assistant" and "source_chunks" in msg:
            with st.expander("📄 View source chunks used"):
                for i, chunk in enumerate(msg["source_chunks"], 1):
                    st.caption(f"Chunk {i}")
                    st.info(chunk[:300] + "..." if len(chunk) > 300 else chunk)

# ── Chat input ───────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about your document..."):

    # Add user message to history and display it
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    with st.chat_message("user"):
        st.write(prompt)

    # Get answer from rag.py and display it
    with st.chat_message("assistant"):
        with st.spinner("Searching document..."):
            result = get_answer(prompt)

        st.write(result["answer"])

        # Show which chunks were used — great for demos
        with st.expander("📄 View source chunks used"):
            for i, chunk in enumerate(result["source_chunks"], 1):
                st.caption(f"Chunk {i}")
                st.info(chunk[:300] + "..." if len(chunk) > 300 else chunk)

    # Save assistant message to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "source_chunks": result["source_chunks"]
    })

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.write("This chatbot answers questions based on your uploaded document using RAG.")
    st.divider()

    # Clear chat history button
    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Built with LangChain + ChromaDB + OpenAI + Streamlit")