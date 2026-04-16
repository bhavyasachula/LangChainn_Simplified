from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_community.embeddings  import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage,SystemMessage
import os
from dotenv import load_dotenv
load_dotenv();

curent_directory = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(curent_directory,"db","chroma_db");

""" LLM Embedder convert text into embeddings"""
embedding = HuggingFaceEmbeddings( model_name="sentence-transformers/all-MiniLM-L6-v2")

"""loads the existing vector store"""
db = Chroma(embedding_function=embedding,persist_directory=persistent_directory)

retriever =  db.as_retriever(  
    search_type="similarity",
    search_kwargs={"k":2}
)
query = "how many Mangaldas Soma is in thier?";
#k=is how many documents u need to retrieve in this case 4
# score_threshold means how relevant or similiar the documents we need from the vector store    
  

relevant_docs = retriever.invoke(query);
# retriever.invoke() retriver will get ur query and searches in replace of like db.similarity_search()
for index,docs in enumerate(relevant_docs):
    print(f"document {index+1}:\n{docs.page_content}\n");
    if docs.metadata:
        print(f"Source:{docs.metadata.get("source","Unknown")}\n");


# relevant_docs looks like:
# [
#   Document(page_content="Text 1"),
#   Document(page_content="Text 2"),
#   Document(page_content="Text 3")
# ]
# 👉 This is already a list of Document objects

combined_input = f"""Here are some documents:
{query}
Relevant Documents:
{"\n\n".join([doc.page_content for doc in relevant_docs])} 
Answer only from above.
""".strip();

message=[
    SystemMessage("You are a helpfull assistant"),
    HumanMessage(combined_input)
]
llm = ChatGroq(model="openai/gpt-oss-120b");
response = llm.invoke(message);
print("full result of llm");
print(response);
print("Only content");
print(response.content);