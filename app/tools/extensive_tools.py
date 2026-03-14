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
    
    def search(self, query: str, search_depth: str = "advanced", max_results: int = 6) -> list:
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
    keyword: str = Field(description="The search query for rankings, salaries, or general engineering info.")

class CourseFetchInput(BaseModel):
    topic: str = Field(description="The engineering branch or specific skill to find courses for (e.g., 'CSE', 'Machine Learning', 'VLSI').")


# ==========================================
# 1. PRIMARY WEB SEARCH TOOL
# ==========================================

@tool(args_schema=GeneralSearchInput)
def web_search(keyword: str) -> str:
    """
    Use this tool to find latest news, college rankings (NIRF), salary trends (LPA), 
    and general engineering career queries in India. 
    DO NOT use this for finding specific courses (use fetch_courses_api instead).
    """
    # Force Indian Context to prevent hallucinations of US/European data
    search_query = keyword
    if "india" not in keyword.lower():
        search_query = f"{keyword} engineering India 2024 2025"
        
    results = tavily_engine.search(search_query, max_results=6)
    
    if not results:
        return "Live web search returned no results. Tell the user you cannot find the information."
        
    formatted = f"LIVE WEB SEARCH RESULTS FOR '{search_query}':\n\n"
    for r in results:
        formatted += f"- Title: {r.get('title')}\n  URL: {r.get('url')}\n  Content: {r.get('content')}\n\n"
    
    formatted += "\nINSTRUCTION: Synthesize this data for the user. Do not invent numbers."
    return formatted


# ==========================================
# 2. COURSE FETCHING API TOOL
# ==========================================

@tool(args_schema=CourseFetchInput)
def fetch_courses_api(topic: str) -> str:
    """
    Use this tool EVERY TIME the user asks for courses, certifications, learning roadmaps, 
    or tutorials. It acts as an API pulling direct courses from NPTEL, Coursera, and Udemy.
    """
    # Create an advanced search query strictly targeting course websites
    search_query = f"top {topic} online course certification India site:nptel.ac.in OR site:coursera.org OR site:udemy.com OR site:edx.org"
    
    results = tavily_engine.search(search_query, max_results=8)
    
    if not results:
        return f"No direct courses found for '{topic}'. Advise the user to broaden their search term."
        
    table_data = "### COURSE API RESULTS (FORMAT AS A MARKDOWN TABLE IN YOUR RESPONSE):\n\n"
    table_data += "| Course Name | Platform/Source | Direct Link |\n"
    table_data += "| --- | --- | --- |\n"
    
    for r in results:
        title = r.get('title', 'Course Link').replace('|', '-')
        url = r.get('url', '')
        
        # Simple logic to determine platform for the table
        platform = "Web Resource"
        if "nptel.ac.in" in url: platform = "NPTEL (Govt of India)"
        elif "coursera.org" in url: platform = "Coursera"
        elif "udemy.com" in url: platform = "Udemy"
        elif "edx.org" in url: platform = "edX"
        
        table_data += f"| {title} | {platform} | [View Course]({url}) |\n"
        
    return table_data

# ==========================================
# TOOL LIST EXPORT (Just the essentials)
# ==========================================

tool_list = [
    web_search, 
    fetch_courses_api
]