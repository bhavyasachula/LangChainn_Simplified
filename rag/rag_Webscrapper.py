import os
from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_community.document_loaders.firecrawl import FireCrawlLoader
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
api_key = os.getenv("FIRECRAWL_API_KEY")

current_dir = os.path.dirname(os.path.abspath(__file__));
db_dir = os.path.join(current_dir,"db")
persistent_directory = os.path.join(db_dir,"chroma_db_firecrawl");

print("Begin crawling the site...")
"""loading the docs using FireCrawlLoader"""

loader = FireCrawlLoader(api_key=api_key, url="https://platform.claude.com/docs/en/about-claude/models/overview", mode="scrape")
docs = loader.load();

print("finished crawling the site....\n");

"""First document text\n"""

print("Printting the page_content of site...\n");
for doc in docs: 
    print(doc.page_content)
print("End of documents.......\n");

"""Splitting the docs......"""

splitter = CharacterTextSplitter(chunk_size=1000 , chunk_overlap=100)
split_docs = splitter.split_documents(docs);

"""Splitting Completed...."""

def create_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2");
    print(f"Creating vector store in {persistent_directory}")
    db  = Chroma.from_documents(split_docs,embedding=embeddings,persist_directory=persistent_directory);
    print("finished vector store creation ");


print(f"\nSplitted docs length {len(split_docs)}");
print(f"Sample chunk {split_docs[0].page_content}\n\n")

if not os.path.exists(persistent_directory):
    create_vector_store()
else:
    print(f"Vector store {persistent_directory} already exists. No need to initialize.")
    

"""loading the Existing vector store"""

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2");
db = Chroma(embedding_function=embeddings,persist_directory=persistent_directory);

"""retrieval from the db """
def load_vector_store(query):
    retriever=db.as_retriever(
        search_type = "similarity",
        search_kwargs ={"k":4}
    )
    relevant_docs = retriever.invoke(query);

    for index,docs in enumerate(relevant_docs):
        print(f"document index: {index+1}\n{docs.page_content}");

query="Claude 4.7";

load_vector_store(query=query);