from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
load_dotenv()
model = ChatGroq(model="openai/gpt-oss-120b")

template = "tell me a joke about {topic}"
prompt_template = ChatPromptTemplate.from_template(template)

print("------ prompt from template  --------")
prompt = prompt_template.invoke({"topic":"cats"})
print(prompt)

#response = model.invoke(prompt)
#print(response.content)
#only a template variable 
print("------------------------------------")
template_multiple = """ 
    human:Tell me a {adjective} short story about a {animal}. 
"""
#main code chatprompttemplate.fromtemplate(pass the variable)
#created the template where using invoke we can replace the 
prompt_multiple = ChatPromptTemplate.from_template(template_multiple)
# this replaces the values from the template multiple.......
response = prompt_multiple.invoke({"adjective":"funny","animal":"panda"})
print(response) 
msg = HumanMessage(content=response.to_string())
print(msg.content)
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)