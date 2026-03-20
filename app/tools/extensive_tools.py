import os
import json
import random
import logging
import requests
import http.client
from dotenv import load_dotenv
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Load environment variables securely
load_dotenv()

# ==========================================
# API KEY SETUP (From .env)
# ==========================================
SERPER_KEYS = [k.strip() for k in os.getenv("SERPER_API_KEYS", os.getenv("SERPER_API_KEY", "")).split(",") if k.strip()]
TAVILY_KEYS = [k.strip() for k in os.getenv("TAVILY_API_KEYS", os.getenv("TAVILY_API_KEY", "")).split(",") if k.strip()]

random.shuffle(SERPER_KEYS)
random.shuffle(TAVILY_KEYS)

# ==========================================
# PRIMARY ENGINE: SERPER (Google Search)
# ==========================================
class SerperSearcher:
    """Round-robin load-balanced Serper API client for Google Search."""
    def __init__(self, keys: list[str]):
        self.keys = keys
        self.current_index = 0

    def search(self, query: str, num_results: int = 5) -> dict:
        if not self.keys:
            return {}

        for _ in range(len(self.keys)):
            idx = self.current_index
            key = self.keys[idx]
            self.current_index = (self.current_index + 1) % len(self.keys)

            try:
                conn = http.client.HTTPSConnection("google.serper.dev", timeout=15)
                payload = json.dumps({"q": query, "num": num_results})
                headers = {
                    'X-API-KEY': key,
                    'Content-Type': 'application/json'
                }
                
                conn.request("POST", "/search", payload, headers)
                res = conn.getresponse()
                
                if res.status == 200:
                    data = res.read()
                    return json.loads(data.decode("utf-8"))
                else:
                    logging.warning(f"⚠️ Serper key index {idx} failed with status {res.status}")
                    
            except Exception as e:
                logging.error(f"⚠️ Serper key index {idx} exception: {str(e)[:80]}")
                continue

        return {}

# ==========================================
# FALLBACK ENGINE: TAVILY (Deep Scrape)
# ==========================================
class TavilySearcher:
    """Round-robin load-balanced Tavily client."""
    def __init__(self, keys: list[str]):
        self.keys = keys
        self.current_index = 0

    def search(self, query: str, max_results: int = 4):
        if not self.keys:
            return [], ""

        for _ in range(len(self.keys)):
            idx = self.current_index
            key = self.keys[idx]
            self.current_index = (self.current_index + 1) % len(self.keys)

            try:
                payload = {
                    "api_key": key,
                    "query": query,
                    "search_depth": "advanced", 
                    "include_answer": True,
                    "include_raw_content": True,
                    "max_results": max_results,
                }
                resp = requests.post("https://api.tavily.com/search", json=payload, timeout=20)
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("results", []), data.get("answer", "")
            except Exception as e:
                logging.warning(f"⚠️ Tavily key index {idx} failed: {str(e)[:80]}")
                continue

        return [], ""

# Instantiate the engines
serper_engine = SerperSearcher(SERPER_KEYS)
tavily_engine = TavilySearcher(TAVILY_KEYS)

def _run_robust_search(query: str, max_results: int = 5) -> str:
    """Tries Google Serper first, falls back to Tavily if Serper fails or has no keys."""
    print(f"\n🌐 [ROBUST SEARCH INITIATED] Query: '{query}'")
    
    # --- ATTEMPT 1: SERPER (Google Search) ---
    if SERPER_KEYS:
        serper_data = serper_engine.search(query, num_results=max_results)
        organic_results = serper_data.get("organic", [])
        
        if organic_results:
            print(f"✅ [SERPER HIT] Retrieved {len(organic_results)} organic results.")
            formatted = f"GOOGLE SEARCH REPORT FOR '{query}':\n\n"
            
            # Add Quick Answer if available
            if "answerBox" in serper_data and "snippet" in serper_data["answerBox"]:
                formatted += f"### QUICK ANSWER:\n{serper_data['answerBox']['snippet']}\n\n"
                
            formatted += "### ORGANIC SEARCH RESULTS:\n"
            for i, r in enumerate(organic_results, 1):
                formatted += (
                    f"[{i}] TITLE: {r.get('title', 'No Title')}\n"
                    f"URL: {r.get('link', 'No URL')}\n"
                    f"SNIPPET: {r.get('snippet', 'No Snippet')}\n"
                    f"{'-'*50}\n"
                )
            return formatted
        else:
            print("⚠️ [SERPER MISS] No organic results. Falling back to Tavily...")

    # --- ATTEMPT 2: TAVILY (Fallback) ---
    if TAVILY_KEYS:
        tavily_results, tavily_answer = tavily_engine.search(query, max_results=max_results)
        
        if tavily_results or tavily_answer:
            print(f"✅ [TAVILY HIT] Retrieved {len(tavily_results)} deep results.")
            formatted = f"DEEP SEARCH REPORT FOR '{query}':\n\n"
            
            if tavily_answer:
                short_ans = tavily_answer[:500] + "..." if len(tavily_answer) > 500 else tavily_answer
                formatted += f"### AI SUMMARY FROM WEB:\n{short_ans}\n\n### DEEP SCRAPED SOURCES:\n"
                
            for i, r in enumerate(tavily_results, 1):
                raw_url = r.get('url', '')
                clean_url = raw_url.split("?utm_")[0].split("?ranMID=")[0].split("?couponCode=")[0]
                content = r.get('raw_content') or r.get('content') or ''
                content = " ".join(content.split())
                
                if len(content) > 800:
                    content = content[:800] + "..."
                    
                formatted += (
                    f"[{i}] SOURCE TITLE: {r.get('title')}\n"
                    f"URL: {clean_url}\n"
                    f"EXTRACTED PAGE CONTENT: {content}\n"
                    f"{'-'*50}\n"
                )
            return formatted
            
    return f"⚠️ No live results found for '{query}'. Both search engines failed or are missing API keys."

# ==========================================
# SCHEMAS & TOOLS
# ==========================================
class GeneralSearchInput(BaseModel):
    keyword: str = Field(
        description="A highly specific Google Search query to find courses, salaries, college stats, or engineering news. Use operators like 'site:youtube.com' if needed."
    )

class FounderInfoInput(BaseModel):
    query: str = Field(description="A short string describing your query. Just say 'founders' or 'team'.")


@tool(args_schema=GeneralSearchInput)
def web_search(keyword: str) -> str:
    """
    Search the live internet using Google Search. 
    Use this tool whenever the user asks for real-world data, courses, roadmaps, or recent news.
    """
    return _run_robust_search(keyword)


_INFERA_CORE_INFO = """
INFERA CORE — Official Team & Founding Information
====================================================
Platform Overview:
INFERA CORE is an advanced artificial intelligence platform designed to transform
how users interact with knowledge, data, and intelligent decision-making technologies.

Founder & CEO: K. V. Maheedhara Kashyap (NMIT, Bangalore)
Co-Founder: Rahul C A (NMIT, Bangalore)
Managing Director: Ujwal (BMS College of Engineering, Bangalore)

Data Science & Analytics Team:
- Pratham S — Data Scientist
- Karan Sable — Data Analyst
- Harshavardhana P M — Data Architect

Sales & Marketing:
- R Rishi — Sales and Marketing Executive
"""

@tool(args_schema=FounderInfoInput)
def get_founder_info(query: str) -> str:
    """Returns authoritative information about the INFERA CORE team, founders, and developers."""
    return _INFERA_CORE_INFO

# Export the tools
tool_list = [web_search, get_founder_info]