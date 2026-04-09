from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from langchain_core.output_parsers import StrOutputParser # stroutputparser is function 
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
model = ChatGroq(model="openai/gpt-oss-120b")

prompttemplate = ChatPromptTemplate.from_messages(
    [("system","You are a comedian who tells a joke on {topic}"),
    ("human","tell me {joke_count} joke. ")
     ]
)
#chain 
chain = prompttemplate | model | StrOutputParser()
result = chain.invoke({"topic":'water','joke_count':2})

print(result)