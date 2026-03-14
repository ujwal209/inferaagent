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
    keyword: str = Field(description="The exact search query to find the requested information.")

# ==========================================
# 1. ONLY TOOL: PRIMARY WEB SEARCH
# ==========================================

@tool(args_schema=GeneralSearchInput)
def web_search(keyword: str) -> str:
    """
    Use this tool to find ANY information the user requests (courses, comparisons, college rankings, salary trends).
    """
    # Force Indian Context to prevent hallucinations of US/European data
    search_query = keyword
    if "india" not in keyword.lower():
        search_query = f"{keyword} engineering India 2024 2025"
        
    # max_results is set slightly higher to ensure enough data for a good comparison table
    results = tavily_engine.search(search_query, max_results=8)
    
    if not results:
        return "Live web search returned no results. Tell the user you cannot find the information."
        
    formatted = f"LIVE WEB SEARCH RESULTS FOR '{search_query}':\n\n"
    for r in results:
        formatted += f"- Title: {r.get('title')}\n  URL: {r.get('url')}\n  Content: {r.get('content')}\n\n"
    
    formatted += "\nINSTRUCTION: Synthesize this data into a SINGLE comprehensive Markdown table, followed immediately by the Summary section."
    return formatted

# ==========================================
# TOOL LIST EXPORT (Just the one tool)
# ==========================================

tool_list = [
    web_search
]