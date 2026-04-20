from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader,PyPDFLoader
# from langchain_community.document_loaders import Textloader
from langchain_community.vectorstores import Chroma

from langchain_text_splitters import CharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.embeddings  import HuggingFaceEmbeddings
import os


# doc = load_doc()
# for i, page in enumerate(doc):
#     print(page.page_content)

#     if i == len(doc) - 1:
#         print("\nReached end of document")

#print(os.path.abspath(__file__));"""path to the rag.py"""
#print(os.path.dirname(os.path.abspath(__file__)));"""path to parent folder/directory rag folder"""
curent_directory = os.path.dirname(os.path.abspath(__file__));
persistent_directory = os.path.join(curent_directory,"db","chroma_db");

"""pypdfloader loads pdf in docloader and usnig load() method we are storing into the documents variable"""
file_path = os.path.join(curent_directory,"data","Bank_details.pdf");

docloader = PyPDFLoader(file_path);
documents = docloader.load();

"""Text splitter makes the chunks of the document as mentioned in chunksize"""
Spiltter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=0) 
#------------------------------------------------------------------------------
"""charactertextsplitter tells the split_document how many chunks size and overlap  
 specify the chunkspillting using characterTextsplitter size and overlap
 docs=CharacterTextSplitter(chunk_size=1000,chunk_overlap=0).split_documents(documents)"""

"""Main work is done by the split_documennts() and textSplitter will tell the split_documents to actually perform the action of splitting """
"""textSplitter defines the spliting and split_documents actually splits the data."""

docs = Spiltter.split_documents(documents)

print("\n documents chunks info----");
print(f"Number of documents chunks:{len(docs)}\n")
print(f"Sample chunk:\n{docs[0].page_content}\n")


""" LLM Embedder convert text into embeddings"""
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("--------finished creating embeddings------------\n");

"""Creating a Vector store for better retrival of embeddings """
print("Creating Vector store");
"""Chroma.from_documents used when u haven't loaded the data and u are creating the store first after the creation we use Chroma() function
db = Chroma(
    persist_directory=...,
    embedding_function=...
)
Data is already stored
You want to search/query """

"""
| Function           | Purpose           |
| ------------------ | ----------------- |
| `from_documents()` | create + store DB |
| `Chroma(...)`      | load + use DB     |

"""
db = Chroma.from_documents(
    docs,
    embedding=embedding,
    persist_directory=persistent_directory
)


print("Vector store created");
