from dotenv import load_dotenv
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_retrieval_chain
from langsmith import Client
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_classic.agents import create_react_agent, AgentExecutor,create_structured_chat_agent
from langchain_classic.tools import Tool
from langchain_core.messages import SystemMessage,AIMessage,HumanMessage
import os
from langchain_huggingface import HuggingFaceEmbeddings
load_dotenv()

curent_directory = os.path.dirname(os.path.abspath(__file__)) # agent_and_tools directory
db_directory = os.path.join(curent_directory,"..","rag","db"); # rag directory into db directory
persistent_directory = os.path.join(db_directory,"chroma_db");  #rag/db/chroma_db

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2");

db = Chroma(embedding_function=embeddings,persist_directory=persistent_directory);
retriver = db.as_retriever( 
    search_type="similarity",
    search_kwargs={"k":2}
    );

llm =ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct");



""" “Take the user’s question + past conversation
 and convert it into a clear, complete question” """

contextualize_q_systemPrompt = (
   "Given a chat history and the latest user question "
"which might reference context in the chat history, "
"formulate a standalone question which can be understood "
"without the chat history. Do NOT answer the question, just "
"reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",contextualize_q_systemPrompt),
        MessagesPlaceholder("chat_history"), #placeholder for chat history
        ("human","{input}")
    ]
)

"""
| Function                         | Meaning                    |
| -------------------------------- | -------------------------- |
| `create_history_aware_retriever` | Fix question using history |
| `create_stuff_documents_chain`   | Answer using documents     |
| `create_retrieval_chain`         | Connect everything         |
"""

qa_system_prompt = (
    "you are an assitant for question-answering tasks. use"
    "the following pieces of retriveed context to answer the"
    "question. if you dont know the answer, just say that you"
    "dont knoww. use three sentences maximum and keep the answer"
    "concise.\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",qa_system_prompt),
        MessagesPlaceholder("chat_history"), #placeholder for chat history
        ("human","{input}")
    ]
)
history_aware_retriver = create_history_aware_retriever(
    llm,
    retriver,
    contextualize_q_prompt
    )

question_answer_chain = create_stuff_documents_chain(llm,qa_prompt);

rag_chain = create_retrieval_chain(history_aware_retriver,question_answer_chain);

client = Client()
react_docstore_prompt = client.pull_prompt("hwchase17/react");

def answer_tool(input):
    return rag_chain.invoke({"input":input,"chat_history":[]});

tools=[
    Tool(
        name="Answer_Question",
        func=answer_tool,
        description="Usefull for when you need to anwser question about the context"
    )
]
agent = create_react_agent(
    llm,
    tools,
    prompt=react_docstore_prompt
    )
agentEzqter = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors = True
)

chat_history = [];
print("Chat with Ai");
while True:
    user_input = input("You :");
    if user_input.lower() == "exit":
        break;
    response = agentEzqter.invoke({"input":user_input,"chat_history":chat_history});
    print("Ai :",response.get("output"));

    chat_history.append(HumanMessage(content=user_input));
    chat_history.append(AIMessage(content=response["output"])); 
