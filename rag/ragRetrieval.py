from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

os.environ['LANSMITH_PROJECT']="RAG Retrieval"

curent_directory = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(curent_directory,"db","chroma_db");

""" LLM Embedder convert text into embeddings"""
embedding = HuggingFaceEmbeddings( model_name="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma(embedding_function=embedding,persist_directory=persistent_directory)

"""Define the users questions"""
query="how many PATEL MANGALDAS SOMA?"; 
"""
| Thing             | Example                        |
| ----------------- | ------------------------------ |
| Chroma DB         | Library                        |
| similarity_search | You search books yourself      |
| Retriever         | Librarian who searches for you |
"""
retriever =  db.as_retriever(  # returns an retriver object 
    search_type="similarity_score_threshold",
    search_kwargs={"k":4,"score_threshold":0.2}
)

config={
    "run_name":"RAG Retrieval",
    "tags":["retrieval","vector store","chroma db"],
    "metadata":{"author":"baapokabaapbhavya",
            "model":"sentence-transformers/all-MiniLM-L6-v2",
            "k":4,"score_threshold":0.2
            }
}
#k=is how many documents u need to retrieve in this case 2
# score_threshold means how relevant or similiar the documents we need from the vector store    
    
""" rag : relevant document retrieval"""
relevant_docs = retriever.invoke(query,config=config);
# retriever.invoke() retriver will get ur query and searches in replace of like db.similarity_search()
for index,docs in enumerate(relevant_docs):
    print(f"document {index+1}:\n{docs.page_content}\n");
    if docs.metadata:
        print(f"Source:{docs.metadata.get("source","Unknown")}\n");



