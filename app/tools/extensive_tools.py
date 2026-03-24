import os
import re
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
# DEEP URL SCRAPER (For Serper Links)
# ==========================================
def scrape_page(url: str) -> str:
    """Visits a URL and aggressively extracts massive raw text content for deep searching."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5'
        }
        resp = requests.get(url, headers=headers, timeout=8)
        if resp.status_code == 200:
            text = resp.text
            text = re.sub(r'<(script|style).*?>.*?</\1>', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = ' '.join(text.split())
            # MASSIVE DEEP SCRAPE: Pulled to 4000 characters for Perplexity-level context
            return text[:4000] 
    except Exception:
        return ""
    return ""

# ==========================================
# PRIMARY ENGINE: SERPER (Google Search)
# ==========================================
class SerperSearcher:
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
            except Exception as e:
                logging.error(f"⚠️ Serper key index {idx} exception: {str(e)[:80]}")
                continue

        return {}

# ==========================================
# SECONDARY ENGINE: TAVILY (AI Search)
# ==========================================
class TavilySearcher:
    def __init__(self, keys: list[str]):
        self.keys = keys
        self.current_index = 0

    def search(self, query: str, max_results: int = 5):
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

serper_engine = SerperSearcher(SERPER_KEYS)
tavily_engine = TavilySearcher(TAVILY_KEYS)

def _run_robust_search(query: str, max_results: int = 5) -> str:
    """Runs BOTH Serper and Tavily simultaneously to guarantee recent, deep data."""
    print(f"\n🌐 [DUAL-ENGINE DEEP SEARCH INITIATED] Query: '{query}'")
    
    formatted_report = f"# 🔍 COMPREHENSIVE SEARCH INTELLIGENCE REPORT FOR '{query}'\n\n"
    has_results = False
    
    # --- 1. FETCH TAVILY (AI Deep Search for recent facts) ---
    if TAVILY_KEYS:
        tavily_results, tavily_answer = tavily_engine.search(query, max_results=max_results)
        
        if tavily_results or tavily_answer:
            has_results = True
            print(f"✅ [TAVILY HIT] Retrieved deep results.")
            formatted_report += "## 🧠 ENGINE 1: TAVILY ADVANCED AI SEARCH\n\n"
            
            if tavily_answer:
                short_ans = tavily_answer[:1500] + "..." if len(tavily_answer) > 1500 else tavily_answer
                formatted_report += f"**🤖 AI Web Summary:**\n{short_ans}\n\n"
                
            for i, r in enumerate(tavily_results, 1):
                raw_url = r.get('url', '')
                clean_url = raw_url.split("?utm_")[0].split("?ranMID=")[0]
                content = r.get('raw_content') or r.get('content') or ''
                content = " ".join(content.split())
                if len(content) > 3000: content = content[:3000] + "..."
                    
                formatted_report += f"### [Source {i}] {r.get('title')}\n**URL:** {clean_url}\n**Content:** {content}\n{'-'*50}\n\n"

    # --- 2. FETCH SERPER (Google for traditional site indexing) ---
    if SERPER_KEYS:
        serper_data = serper_engine.search(query, num_results=max_results)
        organic_results = serper_data.get("organic", [])
        
        if organic_results:
            has_results = True
            print(f"✅ [SERPER HIT] Retrieved {len(organic_results)} organic results. Deep Scraping pages...")
            formatted_report += "## 🌐 ENGINE 2: GOOGLE SEARCH (WITH DEEP SCRAPE)\n\n"
            
            if "answerBox" in serper_data and "snippet" in serper_data["answerBox"]:
                formatted_report += f"**⚡ Quick Answer:** {serper_data['answerBox']['snippet']}\n\n"
                
            for i, r in enumerate(organic_results[:max_results], 1):
                # Offset index to prevent citation overlapping with Tavily
                idx = i + max_results
                title = r.get('title', 'No Title')
                link = r.get('link', 'No URL')
                snippet = r.get('snippet', 'No Snippet')
                
                print(f"    ↳ Scraping: {link}")
                page_content = scrape_page(link)
                # If scraping fails or gets blocked, fallback to snippet
                final_content = page_content if len(page_content) > 100 else snippet
                
                formatted_report += f"### [Source {idx}] {title}\n**URL:** {link}\n**Content:** {final_content}\n{'-'*50}\n\n"

    if not has_results:
        return f"⚠️ No live results found for '{query}'. Try using a different keyword combination."
        
    return formatted_report

# ==========================================
# SCHEMAS & TOOLS
# ==========================================
class GeneralSearchInput(BaseModel):
    keyword: str = Field(
        description="An SEO-optimized, highly specific Google Search query. "
                    "CRITICAL: Do NOT just use the user's raw input! You MUST auto-correct typos and expand abbreviations to get good results. "
                    "(e.g., if user says 'bms 2025 placemets', you MUST query 'BMS College of Engineering Bangalore 2024 2025 placement statistics average highest package')."
    )

class FounderInfoInput(BaseModel):
    query: str = Field(description="A short string describing your query. Just say 'founders' or 'team'.")

@tool(args_schema=GeneralSearchInput)
def web_search(keyword: str) -> str:
    """Search the live internet using Both Google Search & AI Search with Deep Scraping."""
    return _run_robust_search(keyword)

_INFERA_CORE_INFO = """
INFERA CORE — Official Team & Founding Information
====================================================
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
    """Returns authoritative information about the INFERA CORE team."""
    return _INFERA_CORE_INFO

tool_list = [web_search, get_founder_info]