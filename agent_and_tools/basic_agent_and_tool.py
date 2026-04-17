from dotenv import load_dotenv
from langsmith import Client
from langchain_groq import ChatGroq
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic.tools import Tool

load_dotenv()

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
client = Client()
prompt = client.pull_prompt("hwchase17/react")

def get_datetime(*args, **kwargs):
    import datetime
    return str(datetime.datetime.now())

# ✅ Correct tool format for react agent
tools = [
    Tool(
        name="time",
        func=get_datetime,
        description="Useful for when you need to know the current time"
    )
]

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True
)

response = agent_executor.invoke({"input": "What time is it?"})

print("response:", response["output"])