import os

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langsmith import traceable

# ── Environment ───────────────────────────────────────────────────────────────

load_dotenv()
os.environ["LANGSMITH_PROJECT"] = "RAG-LLM-QA"

# ── Paths ─────────────────────────────────────────────────────────────────────

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR = os.path.join(CURRENT_DIR, "db", "chroma_db")

# ── Constants ─────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL       = "openai/gpt-oss-120b"
TOP_K_RESULTS   = 4
QUERY           = "how many Mangaldas Soma is in thier?"

# ── Embedding & LLM setup ─────────────────────────────────────────────────────

embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
llm       = ChatGroq(model=LLM_MODEL)

# ── Functions ─────────────────────────────────────────────────────────────────

@traceable(name="load_vector_store")
def load_vector_store(embedding, persist_dir: str) -> Chroma:
    """Load the existing Chroma vector store from disk."""
    return Chroma(embedding_function=embedding, persist_directory=persist_dir)


@traceable(name="get_retriever")
def get_retriever(db: Chroma):
    """Build and return a similarity-based retriever."""
    return db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RESULTS},
    )


@traceable(name="show_docs")
def show_docs(relevant_docs: list) -> None:
    """Pretty-print retrieved documents."""
    for i, doc in enumerate(relevant_docs, start=1):
        print(f"Document {i}:\n{doc.page_content}\n")


@traceable(name="setup_retriever")
def setup_retriever(embedding, persist_dir: str, query: str) -> list:
    """Load vector store → build retriever → return relevant docs."""
    db            = load_vector_store(embedding, persist_dir)
    retriever     = get_retriever(db)
    relevant_docs = retriever.invoke(query)
    show_docs(relevant_docs)
    return relevant_docs


@traceable(name="build_prompt")
def build_prompt(query: str, relevant_docs: list) -> str:
    """Combine query + retrieved context into a prompt."""
    context = "\n".join(doc.page_content for doc in relevant_docs)
    return (
        f"Here are some documents:\n"
        f"{query}\n\n"
        f"Relevant Documents:\n{context}\n\n"
        f"Answer only from above."
    )


# ── Main ──────────────────────────────────────────────────────────────────────

relevant_docs = setup_retriever(embedding, PERSIST_DIR, QUERY)

messages = [
    SystemMessage("You are a helpful assistant"),
    HumanMessage(build_prompt(QUERY, relevant_docs)),
]

response = llm.invoke(messages)

print("Full LLM result:")
print(response)
print("\nContent only:")
print(response.content)