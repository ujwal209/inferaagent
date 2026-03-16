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

    def search(self, query: str, max_results: int = 10):
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

def _run_search(query: str, max_results: int = 10) -> tuple[list, str]:
    """Try Tavily Advanced with round-robin key rotation."""
    print(f"\n🌐 [SEARCH INITIATED] Deep Tavily Search for: '{query}'")
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
            "Example: 'Top Machine Learning courses NPTEL Coursera India'"
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
    Formulate a highly specific query based on what the user needs to get accurate and latest data.
    """
    results, answer = _run_search(keyword, max_results=10)

    if not results and not answer:
        return (
            f"No live results found for '{keyword}'. "
            "Inform the user and suggest a broader or differently worded search."
        )

    formatted = f"DEEP SEARCH REPORT FOR '{keyword}':\n\n"
    if answer:
        formatted += f"### EXTRACTED ANSWER / LATEST DATA\n{answer}\n\n### RAW SOURCES:\n"
        
    for i, r in enumerate(results, 1):
        formatted += (
            f"[{i}] TITLE: {r.get('title')}\n"
            f"URL: {r.get('url')}\n"
            f"CONTENT: {r.get('content')}\n\n"
        )
    return formatted


# Authoritative, hardcoded team data — never hallucinate this
_INFERA_CORE_INFO = """
INFERA CORE — Official Team & Founding Information
====================================================
any question asked explain in depth about the below content
Platform Overview:
INFERA CORE is an advanced artificial intelligence platform designed to transform
how users interact with knowledge, data, and intelligent decision-making technologies.
It is built on the principle of converting complex data into meaningful intelligence,
making advanced AI accessible, reliable, and practical for real-world applications.

Founder & CEO:
- Name: K. V. Maheedhara Kashyap
- Role: Founder and CEO
- Institution: Nitte Meenakshi Institute of Technology, Bangalore, Karnataka
- Contribution: Envisioned and architected INFERA CORE as a next-generation AI
  intelligence system; the primary driving force behind the platform.

Co-Founder:
- Name: Rahul C A
- Role: Co-Founder
- Institution: Nitte Meenakshi Institute of Technology, Bangalore, Karnataka

Managing Director:
- Name: Ujwal
- Role: Managing Director
- Institution: BMS College of Engineering, Bangalore, Karnataka
- Contribution: Leads a major part of the platform's development and operational
  advancement; significant technical contributions and leadership in shaping the
  architecture, development, and execution of INFERA CORE.

Data Science & Analytics Team (all from Nitte Meenakshi Institute of Technology,
Bangalore, Karnataka):
- Pratham S — Data Scientist
- Karan S — Data Analyst
- Harsha P M — Data Architect; worked efficiently in the collection of approved
  data which helps in building the accurate AI agent.

Sales & Marketing:
- Rishi — Sales and Marketing Executive
  Institution: Nitte Meenakshi Institute of Technology, Bangalore, Karnataka

Guiding Principle:
To engineer an AI-driven ecosystem that converts complex data and information into
meaningful intelligence — empowering users with accurate insights, intelligent
guidance, and data-driven decision support.
"""

@tool(args_schema=FounderInfoInput)
def get_founder_info(query: str) -> str:
    """
    Use this tool whenever the user asks anything about INFERA CORE's founders,
    co-founders, CEO, managing director, team members, who built the platform,
    or any information about the people behind INFERA CORE.
    Returns authoritative, factual information about the INFERA CORE team.
    """
    return _INFERA_CORE_INFO


# ==========================================
# TOOL LIST EXPORT
# ==========================================

tool_list = [
    web_search,
    get_founder_info
]