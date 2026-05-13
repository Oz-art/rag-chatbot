# rag.py
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

# ── 1. LOAD the vector database from disk ───────────────────────
embeddings_model = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY")
)
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings_model
)

# ── 2. CREATE retriever (finds top 3 relevant chunks) ───────────
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ── 3. SET UP the LLM ───────────────────────────────────────────
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

# ── 4. CREATE a prompt template ─────────────────────────────────
#    This is the instruction we send to the LLM every time.
#    {context} = the chunks retrieved from chroma_db
#    {question} = the user's question
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below.
If the answer is not in the context, say "I don't know based on the document."

Context:
{context}

Question:
{question}
""")

# ── 5. DEFINE the get_answer function ───────────────────────────
#    This is what app.py and main.py will import and call
def get_answer(question: str) -> dict:
    # Retrieve relevant chunks from chroma_db
    docs = retriever.invoke(question)

    # Combine chunk texts into one context string
    context = "\n\n".join(doc.page_content for doc in docs)

    # Build the prompt and send to LLM
    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "question": question
    })

    return {
        "answer": response.content,
        "source_chunks": [doc.page_content for doc in docs]
    }


# ── 6. TEST it directly in terminal ─────────────────────────────
#    This only runs when you do: python rag.py
#    It won't run when app.py imports this file
if __name__ == "__main__":
    while True:
        question = input("\nYou: ")
        if question.lower() == "exit":
            break
        result = get_answer(question)
        print(f"\nBot: {result['answer']}")
        print("\n--- Chunks used ---")
        for i, chunk in enumerate(result["source_chunks"], 1):
            print(f"\n[{i}] {chunk[:200]}...")