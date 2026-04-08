from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
model = ChatGroq(model="openai/gpt-oss-120b")

template = "tell me a joke about {topic}"
prompt_template = ChatPromptTemplate.from_template(template)

print("------ prompt from template  --------")
prompt = prompt_template.invoke({"topic":"cats"})
print(prompt)

response = model.invoke(prompt)
print(response.content)