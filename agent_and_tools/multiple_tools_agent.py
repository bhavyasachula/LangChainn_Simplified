from dotenv import load_dotenv
from langsmith import Client
from langchain_groq import ChatGroq
from langchain_classic.agents import create_react_agent, AgentExecutor,create_structured_chat_agent
from langchain_classic.tools import Tool
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage,AIMessage,HumanMessage
load_dotenv()

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
client = Client()
prompt = client.pull_prompt("hwchase17/react")

def get_datetime(*args, **kwargs):
    import datetime
    now =datetime.datetime.now()
    return str(now)

def wikipedia_summary(query):
    from wikipedia import summary
    result = summary(query,sentence=3)
    return(result)

tools = [
    Tool(
        name="time",
        func=get_datetime,
        description="Useful for when you need to know the current time"
    ),
    Tool(
        name="wikipedia",
        func=wikipedia_summary,
        description="to find the information about any topic"
    )
]

agent = create_structured_chat_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

memory = ConversationBufferMemory(
  memory_key="chat_history",
  return_messages=True
)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory,
    handle_parsing_errors =True
)
initialmsg = "You are an helpfull assistant that provides an answers if u dont know the answer just say i dont know";
memory.chat_memory.add_message(SystemMessage(content=initialmsg));

while True:
    print("Chat with Ai");
    user_input=input("You :");

    if user_input.lower() == "exit":
        break;

    memory.chat_memory.add_message(HumanMessage(content=user_input));

    response = agent_executor.invoke({"input":user_input});
    print("Ai :"+ response["output"]);
    memory.chat_memory.add_message(AIMessage(content=response["output"]));
    break

  
    