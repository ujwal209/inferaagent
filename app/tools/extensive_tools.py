from langchain_core.tools import tool
from pydantic import BaseModel, Field
import os
import requests
from dotenv import load_dotenv
import random

# Load environment
load_dotenv()

# ==========================================
# TAVILY WEB SEARCH ENGINE (Load Balanced)
# ==========================================
TAVILY_KEYS = [k.strip() for k in os.getenv("TAVILY_API_KEYS", os.getenv("TAVILY_API_KEY", "")).split(",") if k.strip()]
random.shuffle(TAVILY_KEYS)

class TavilySearcher:
    def __init__(self, keys: list[str]):
        self.keys = keys
        self.current_index = 0
    
    def search(self, query: str, search_depth: str = "advanced", max_results: int = 8) -> list:
        if not self.keys:
            return []
        
        max_retries = len(self.keys)
        
        for _ in range(max_retries):
            idx = self.current_index
            key = self.keys[idx]
            self.current_index = (self.current_index + 1) % len(self.keys)
            
            try:
                payload = {
                    "api_key": key,
                    "query": query,
                    "search_depth": search_depth,
                    "max_results": max_results
                }
                response = requests.post("https://api.tavily.com/search", json=payload, timeout=15)
                if response.status_code == 200:
                    return response.json().get("results", [])
            except:
                pass
        return []

tavily_engine = TavilySearcher(TAVILY_KEYS)

# ==========================================
# SCHEMAS
# ==========================================
class GeneralSearchInput(BaseModel):
    keyword: str = Field(description="A highly specific search query to find courses, salaries, or engineering info. Example: 'Top Machine Learning courses NPTEL Coursera India'")

# ==========================================
# 1. ONLY TOOL: PRIMARY WEB SEARCH
# ==========================================

@tool(args_schema=GeneralSearchInput)
def web_search(keyword: str) -> str:
    """
    Use this tool to search the live internet for courses, career paths, salaries, colleges, or news.
    Formulate a highly specific search query based on what the user needs.
    """
    results = tavily_engine.search(keyword, max_results=8)
    
    if not results:
        return f"No live results found for '{keyword}'. Inform the user and suggest a broader search."
        
    formatted = f"RAW SEARCH RESULTS FOR '{keyword}':\n\n"
    for i, r in enumerate(results, 1):
        # Explicitly passing URL so the LLM doesn't lose it
        formatted += f"[{i}] TITLE: {r.get('title')}\nURL: {r.get('url')}\nCONTENT: {r.get('content')}\n\n"
    
    return formatted

# ==========================================
# TOOL LIST EXPORT
# ==========================================

tool_list = [
    web_search
]