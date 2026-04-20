"""Langsmith is a unified observability & evaluation platform where you can debug, test and monitor Ai app performance. This demo shows how to use LangSmith to track and evaluate your Ai app performance."""
from langchain_classic.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b");

prompt = PromptTemplate.from_template("{question}");



chain = prompt | llm | StrOutputParser()

response = chain.invoke({"question":"what is the capital of france?"});

print(response);