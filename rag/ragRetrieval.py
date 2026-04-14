from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_community.embeddings  import HuggingFaceEmbeddings
import os


curent_directory = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(curent_directory,"db","chroma_db");

""" LLM Embedder convert text into embeddings"""
embedding = HuggingFaceEmbeddings( model_name="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma(embedding_function=embedding,persist_directory=persistent_directory)

"""Define the users questions"""
query="how many PATEL MANGALDAS SOMA?";

retriever =  db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k":4,"score_threshold":0.2}
)
#k=is how many documents u need to retrieve in this case 4
# score_threshold means how relevant or similiar the documents we need from the vector store    
    
""" rag : relevant document retrieval"""
relevant_docs = retriever.invoke(query);
for index,docs in enumerate(relevant_docs):
    print(f"document {index+1}:\n{docs.page_content}\n");
    if docs.metadata:
        print(f"Source:{docs.metadata.get("source","Unknown")}\n");



