import os
from dotenv import load_dotenv
from langsmith import Client
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.agents import create_react_agent,AgentExecutor
from langchain.tools import tool
load_dotenv()