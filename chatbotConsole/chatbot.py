from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
model = ChatGroq(model="openai/gpt-oss-120b")

chat_history = []
systemMsg = SystemMessage(content="You are a helping assitant do not use so much of tokens just to the point")
chat_history.append(systemMsg)

while(True):
    query = input("You: ")
    if query.lower() == "exit":
        break;
    response = model.invoke(query)
    chat_history.append(AIMessage(content=response.content))
    print("Ai: " + response.content)

print("Chat history\n" + chat_history)