from langchain_core.tools import tool
from pydantic import BaseModel, Field
import os
import requests
from dotenv import load_dotenv
import random

# Load environment variables
load_dotenv()

# ==========================================
# DEEP TAVILY WEB SEARCH ENGINE (Round-Robin)
# ==========================================
TAVILY_KEYS = [k.strip() for k in os.getenv("TAVILY_API_KEYS", os.getenv("TAVILY_API_KEY", "")).split(",") if k.strip()]
random.shuffle(TAVILY_KEYS)

class TavilySearcher:
    """Round-robin load-balanced Tavily client — Deep & Advanced."""

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
                    "include_raw_content": True, # CRITICAL: Enables deep page scraping
                    "max_results": max_results,
                }
                resp = requests.post(
                    "https://api.tavily.com/search", json=payload, timeout=20
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("results", []), data.get("answer", "")
            except Exception as e:
                print(f"⚠️  Tavily key index {idx} failed: {str(e)[:80]}")
                continue

        return [], ""

tavily_engine = TavilySearcher(TAVILY_KEYS)

def _run_search(query: str, max_results: int = 4) -> tuple[list, str]:
    """Try Tavily Advanced with round-robin key rotation."""
    print(f"\n🌐 [DEEP SEARCH INITIATED] Query: '{query}'")
    results, answer = tavily_engine.search(query, max_results=max_results)
    
    if results:
        print(f"✅ [TAVILY HIT] Retrieved {len(results)} raw results.")
        return results, answer
        
    return [], ""


# ==========================================
# SCHEMAS
# ==========================================
class GeneralSearchInput(BaseModel):
    keyword: str = Field(
        description=(
            "A highly specific search query to find courses, salaries, or engineering info. "
            "If searching for courses on a specific platform, use site operators. "
            "Examples: 'site:coursera.org/learn machine learning', 'site:udemy.com/course ethical hacking'"
        )
    )

class FounderInfoInput(BaseModel):
    query: str = Field(description="A short string describing your query. Just say 'founders' or 'team'.")


# ==========================================
# TOOLS
# ==========================================

@tool(args_schema=GeneralSearchInput)
def web_search(keyword: str) -> str:
    """
    Search the live internet for courses, career paths, salaries, colleges, or news using Deep Advanced Search.
    This tool extracts the actual page content. Use it to find exact statistics, rankings, and details.
    """
    results, answer = _run_search(keyword, max_results=4)

    if not results and not answer:
        return (
            f"No live results found for '{keyword}'. "
            "Inform the user and try a broader search."
        )

    formatted = f"DEEP SEARCH REPORT FOR '{keyword}':\n\n"
    if answer:
        # Keep answer brief to save tokens for the deep scrapes
        short_ans = answer[:500] + "..." if len(answer) > 500 else answer
        formatted += f"### AI SUMMARY FROM WEB:\n{short_ans}\n\n### DEEP SCRAPED SOURCES:\n"
        
    for i, r in enumerate(results, 1):
        raw_url = r.get('url', '')
        
        # Rigorous URL Cleaning to prevent 404s
        clean_url = raw_url.split("?utm_")[0].split("?ranMID=")[0].split("?couponCode=")[0].split("?matchtype=")[0]
        
        # Prioritize the deep scraped raw_content, fallback to the snippet
        content = r.get('raw_content') or r.get('content') or ''
        
        # Clean up excessive whitespace and newlines from the scrape
        content = " ".join(content.split())
        
        # 800 chars provides deep context but keeps token count incredibly safe
        if len(content) > 800:
            content = content[:800] + "..."
            
        formatted += (
            f"[{i}] SOURCE TITLE: {r.get('title')}\n"
            f"URL: {clean_url}\n"
            f"EXTRACTED PAGE CONTENT: {content}\n"
            f"{'-'*50}\n"
        )
    return formatted


_INFERA_CORE_INFO = """
INFERA CORE — Official Team & Founding Information
====================================================
Platform Overview:
INFERA CORE is an advanced artificial intelligence platform designed to transform
how users interact with knowledge, data, and intelligent decision-making technologies.
It is built on the principle of converting complex data into meaningful intelligence.

Founder & CEO: K. V. Maheedhara Kashyap (Nitte Meenakshi Institute of Technology, Bangalore)
Co-Founder: Rahul C A (Nitte Meenakshi Institute of Technology, Bangalore)
Managing Director: Ujwal (BMS College of Engineering, Bangalore)

Data Science & Analytics Team:
- Pratham S — Data Scientist (NMIT)
- Karan S — Data Analyst (NMIT)
- Harsha P M — Data Architect (NMIT)

Sales & Marketing:
- Rishi — Sales and Marketing Executive (NMIT)
"""

@tool(args_schema=FounderInfoInput)
def get_founder_info(query: str) -> str:
    """Returns authoritative information about the INFERA CORE team."""
    return _INFERA_CORE_INFO


# ==========================================
# TOOL LIST EXPORT
# ==========================================
tool_list = [
    web_search,
    get_founder_info,
]