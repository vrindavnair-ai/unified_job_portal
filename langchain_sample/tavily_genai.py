""" 
from langchain_tavily import TavilySearch
import os

os.environ["TAVILY_API_KEY"] = os.getenv("tavily_api_key")
search = TavilySearch(max_results=2)
search_results = search.invoke("What is the weather in SF")
print(search_results)
"""
from dotenv import load_dotenv
import os
from langchain_tavily import TavilySearch

load_dotenv()  # loads .env file
search = TavilySearch(max_results=2)
tools = [search]
"""
search_results = search.invoke("find the suitable job links for an ai developer")
print(search_results)
print("\n\n")
print(search_results['results'][0]['content'])

"""

from langchain.chat_models import init_chat_model

model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
"""
query = "Hi!"
response = model.invoke([{"role": "user", "content": query}])
print(response.text())
"""
model_with_tools = model.bind_tools(tools)
query = "find the suitable job links for an ai developer"
response = model_with_tools.invoke([{"role": "user", "content": query}])
print(response.text())
#print(f"Message content: {response.text()}\n")
print(f"Tool calls: {response.tool_calls}")