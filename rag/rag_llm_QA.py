import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langsmith import traceable

# ── Environment ───────────────────────────────────────────────────────────────

load_dotenv()
os.environ["LANSMITH_PROJECT"] = "RAG-LLM-QA"
# ── Paths ─────────────────────────────────────────────────────────────────────

CURRENT_DIR    = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR    = os.path.join(CURRENT_DIR, "db", "chroma_db")

# ── Constants ─────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL       = "openai/gpt-oss-120b"
TOP_K_RESULTS   = 4
QUERY           = "how many Mangaldas Soma is in thier?"

# ── Embedding & LLM setup ─────────────────────────────────────────────────────

# Converts text into vector embeddings for semantic search
embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

llm = ChatGroq(model=LLM_MODEL)

# ── Vector store ──────────────────────────────────────────────────────────────
@traceable(name="load_vector_store")
def load_vector_store(embedding, persist_dir: str) -> Chroma:
    """Load the existing Chroma vector store from disk."""
    return Chroma(embedding_function=embedding, persist_directory=persist_dir)

@traceable(name="retrieve_relevant_docs")
def retrieve_relevant_docs(db: Chroma) -> list:
    """
    Retrieve the top-K most similar documents for a given query.

    - search_type="similarity" : cosine / dot-product similarity search
    - k                        : number of documents to return
    """
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RESULTS},
    )
    return retriever


# ── Display ───────────────────────────────────────────────────────────────────

@traceable(name="show_docs")
def show_docs(relevant_docs: list) -> None:
    """Pretty-print retrieved documents and their source metadata."""
    for i, doc in enumerate(relevant_docs, start=1):
        print(f"Document {i}:\n{doc.page_content}\n")


# ── Prompt builder ────────────────────────────────────────────────────────────
@traceable(name="build_prompt")
def build_prompt(query: str, relevant_docs: list) -> str:
    """
    Combine the user query with retrieved context into a single prompt.

    relevant_docs structure:
        [Document(page_content="..."), Document(page_content="..."), ...]
    """
    context = "\n".join(doc.page_content for doc in relevant_docs)
    return (
        f"Here are some documents:\n"
        f"{query}\n\n"
        f"Relevant Documents:\n{context}\n\n"
        f"Answer only from above."
    )


# ── Pipeline ──────────────────────────────────────────────────────────────────
@traceable(name="run_pipeline")
def run_pipeline(embedding, persist_dir: str, query: str) -> list:
    """Load vector store → retrieve docs → display them."""
    db = load_vector_store(embedding, persist_dir)
    relevant_docs = retrieve_relevant_docs(db)
    show_docs(relevant_docs)
    return relevant_docs


# ── Main ──────────────────────────────────────────────────────────────────────

relevant_docs = run_pipeline(embedding, PERSIST_DIR, QUERY)
messages = [
    SystemMessage("You are a helpful assistant"),
    HumanMessage(build_prompt(QUERY, relevant_docs)),
]

response = llm.invoke(messages)

print("Full LLM result:")
print(response)
print("\nContent only:")
print(response.content)