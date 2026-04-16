from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader,PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_history_aware_retriever,create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import CharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_community.embeddings  import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage,SystemMessage
from dotenv import load_dotenv
import os
load_dotenv();

curent_directory = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(curent_directory,"db","chroma_db");

""" LLM Embedder convert text into embeddings"""
embedding = HuggingFaceEmbeddings( model_name="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma(embedding_function=embedding,persist_directory=persistent_directory)

retriever =  db.as_retriever(  
    search_type="similarity",
    search_kwargs={"k":5}
)

qa_system_prompt = (
    "you are an assitant for question-answering tasks. use"
    "the following pieces of retriveed context to answer the"
    "question. if you dont know the answer, just say that you"
    "dont knoww. use three sentences maximum and keep the answer"
    "concise.\n\n"
    "{context}"
)
LLM = ChatGroq(model="openai/gpt-oss-120b");

contextualize_q_systemPrompt = (
   "Given a chat history and the latest user question "
"which might reference context in the chat history, "
"formulate a standalone question which can be understood "
"without the chat history. Do NOT answer the question, just "
"reformulate it if needed and otherwise return it as is."
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human","{input}")
    ]
)

history_aware_retriever = create_history_aware_retriever(
    LLM,
    retriever,
    qa_prompt
)



question_answer_chain = create_stuff_documents_chain(LLM,qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain);

def chat():

    print("Start Chatting with Ai If break use 'exit' keyword");
chat_history = []
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break

    result = rag_chain.invoke({
        "input": query,
        "chat_history": chat_history
    })

    print(f"AI: {result['answer']}")
    chat_history.append(HumanMessage(content=query))
    chat_history.append(SystemMessage(content=result["answer"]))

if __name__ == "__main__":
    chat()