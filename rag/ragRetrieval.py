from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader,PyPDFLoader
# from langchain_community.document_loaders import Textloader
from langchain_community.vectorstores import Chroma

from langchain_text_splitters import CharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.embeddings  import HuggingFaceEmbeddings
import os


curent_directory = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(curent_directory,"data","Bank_details.pdf");
persistent_directory = os.path.join(curent_directory,"db","chroma_db");


""" LLM Embedder convert text into embeddings"""
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


"""Creating a Vector store for better retrival of embeddings """
print("Creating Vector store");
db = Chroma(
    embedding_function=embedding,
    persist_directory=persistent_directory
)



