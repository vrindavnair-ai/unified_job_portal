from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import os
from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model
import re
import streamlit as st
import fitz  # PyMuPDF for PDF
#import docx
from docx import Document
#To retry if the model is not working
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
#import openai
load_dotenv()  # loads .env file

model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")


search = TavilySearch(max_results=2)
tools = [search]

agent_executor = create_react_agent(model, tools)
input_message = {"role": "user", "content": """ You are a recruitment assistant helping in findind the 
suitable job links for an ai developer at dubai location.You have to provide specific link to the job posted on any job board
                 posted"""}
#response = agent_executor.invoke({"messages": [input_message]})
#To just get the output

response = agent_executor.invoke({"messages": [input_message]})
#print(response)
#print(response['messages']) #it is a list


messages = response['messages']

urls = []
url_pattern = re.compile(r'https?://[^\s\)\]"]+')

for msg in messages:
    content = msg.content
    found_urls = url_pattern.findall(content)
    urls.extend(found_urls)

# Remove duplicates
urls = list(set(urls))

seen = set()
unique_urls = []
for url in urls:
    if url not in seen:
        unique_urls.append(url)
        seen.add(url)

urls = unique_urls
"""
url_str = ""

url_str = "\n".join(urls)

print(url_str)
"""


for url in urls:
    print(url)
    




