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
# NPTEL COURSE SEARCH
# ==========================================

NPTEL_BASE = "https://api.nptelprep.in"

class NPTELSearchInput(BaseModel):
    keyword: str = Field(
        description=(
            "A keyword or topic to search for among NPTEL courses. "
            "Examples: 'machine learning', 'data structures', 'thermodynamics'"
        )
    )

@tool(args_schema=NPTELSearchInput)
def search_nptel_courses(keyword: str) -> str:
    """
    Search NPTEL (National Programme on Technology Enhanced Learning) courses by keyword.
    Use this tool when the user asks for NPTEL courses, NPTEL links, or NPTEL content on any topic.
    Returns course names, course codes, week info, video count, and direct NPTEL links.
    """
    try:
        print(f"\n📚 [NPTEL SEARCH] Fetching all NPTEL courses for keyword: '{keyword}'")
        resp = requests.get(f"{NPTEL_BASE}/courses", timeout=15)
        if resp.status_code != 200:
            return f"NPTEL API returned status {resp.status_code}. Please try again later."

        data = resp.json()
        courses = data.get("courses", [])

        # Filter by keyword (case-insensitive)
        kw = keyword.lower()
        matched = [
            c for c in courses
            if kw in c.get("course_name", "").lower()
        ]

        if not matched:
            # Fall back to top 10 most requested if no match
            _all_sorted: list = sorted(courses, key=lambda x: x.get("request_count", 0), reverse=True)
            matched = _all_sorted[:10]
            header = f"No exact matches for '{keyword}' on NPTEL. Showing top 10 most popular courses:\n\n"
        else:
            _matched_sorted: list = sorted(matched, key=lambda x: x.get("request_count", 0), reverse=True)
            matched = _matched_sorted[:15]
            header = f"NPTEL courses matching '{keyword}':\n\n"

        output = header
        for c in matched:
            code = c.get("course_code", "")
            name = c.get("course_name", "N/A")
            weeks = [w for w in (c.get("weeks") or []) if w is not None]
            videos = c.get("video_count", 0)
            questions = c.get("question_count", 0)
            nptel_link = f"https://nptel.ac.in/courses/{code}"
            swayam_link = f"https://onlinecourses.nptel.ac.in/noc{code[:2]}_{code[2:]}"

            output += (
                f"📘 **{name}**\n"
                f"   Course Code : {code}\n"
                f"   Weeks       : {len(weeks)} week(s) — {weeks if weeks else 'Self-paced'}\n"
                f"   Videos      : {videos} | Practice Questions: {questions}\n"
                f"   NPTEL Link  : {nptel_link}\n\n"
            )
        return output.strip()
    except Exception as e:
        return f"Error fetching NPTEL courses: {str(e)}"


# ==========================================
# COURSERA COURSE SEARCH
# ==========================================

class CourseraSearchInput(BaseModel):
    keyword: str = Field(
        description=(
            "A topic or keyword to search on Coursera. "
            "Examples: 'python programming', 'deep learning', 'project management'"
        )
    )

@tool(args_schema=CourseraSearchInput)
def search_coursera_courses(keyword: str) -> str:
    """
    Search Coursera for courses matching a keyword topic.
    Use this when the user asks for Coursera courses, Coursera links, or Coursera recommendations.
    Returns course names, partner/university, workload, and direct Coursera links.
    """
    try:
        print(f"\n🎓 [COURSERA SEARCH] Searching for: '{keyword}'")
        search_url = f"https://www.coursera.org/search?query={keyword.replace(' ', '%20')}"

        # Try Coursera's catalog API (no auth needed for simple browse)
        params = {
            "fields": "name,slug,partnerIds,workload",
            "includes": "partners",
            "limit": 100,
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; InferaBot/1.0)",
            "Accept": "application/json",
        }
        resp = requests.get(
            "https://api.coursera.org/api/courses.v1",
            params=params,
            headers=headers,
            timeout=15,
        )

        if resp.status_code == 200:
            data = resp.json()
            elements = data.get("elements", [])
            linked = data.get("linked", {})
            partners_map = {
                p["id"]: p.get("name", "Unknown")
                for p in linked.get("partners.v1", [])
            }

            # Filter by keyword relevance
            kw = keyword.lower()
            matched = [c for c in elements if kw in c.get("name", "").lower()]

            if not matched:
                matched = elements[:10]  # show first 10 as fallback

            if not matched:
                return (
                    f"No Coursera courses found for '{keyword}'.\n"
                    f"👉 Search directly: {search_url}"
                )

            output = f"Coursera courses for '{keyword}':\n\n"
            for c in matched[:12]:
                name = c.get("name", "N/A")
                slug = c.get("slug", "")
                partner_ids = c.get("partnerIds", [])
                partner_names = ", ".join(
                    str(partners_map.get(str(pid), str(pid))) for pid in partner_ids
                ) if partner_ids else "Coursera"
                link = f"https://www.coursera.org/learn/{slug}" if slug else search_url
                workload = c.get("workload", "N/A")

                output += (
                    f"🎓 **{name}**\n"
                    f"   By          : {partner_names}\n"
                    f"   Workload    : {workload}\n"
                    f"   Link        : {link}\n\n"
                )
            output += f"🔍 Browse all results: {search_url}"
            return output.strip()
        else:
            return (
                f"Coursera API unavailable (status {resp.status_code}).\n"
                f"👉 Search directly: {search_url}"
            )
    except Exception as e:
        search_url = f"https://www.coursera.org/search?query={keyword.replace(' ', '%20')}"
        return (
            f"Error fetching Coursera courses: {str(e)}\n"
            f"👉 Search manually: {search_url}"
        )


# ==========================================
# UDEMY COURSE SEARCH
# ==========================================

class UdemySearchInput(BaseModel):
    keyword: str = Field(
        description=(
            "A topic or keyword to search on Udemy. "
            "Examples: 'react js', 'ethical hacking', 'AWS certification'"
        )
    )

@tool(args_schema=UdemySearchInput)
def search_udemy_courses(keyword: str) -> str:
    """
    Search Udemy for courses matching a keyword topic.
    Use this when the user asks for Udemy courses, Udemy links, or Udemy recommendations.
    Returns course titles, instructors, ratings, student counts, price, and direct Udemy links.
    Since Udemy's public API requires authentication, this tool uses the best available
    open alternatives and always provides a direct Udemy search link.
    """
    try:
        print(f"\n🛒 [UDEMY SEARCH] Searching for: '{keyword}'")
        search_url = f"https://www.udemy.com/courses/search/?q={keyword.replace(' ', '+')}&sort=relevance"

        # Try Open Educational Resources API (no auth needed)
        params = {
            "search": keyword,
            "page": 1,
            "page_size": 10,
            "ordering": "relevance",
            "language": "en",
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; InferaBot/1.0)",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        # Udemy requires API key — use their public course catalog endpoint as a best-effort
        resp = requests.get(
            "https://www.udemy.com/api-2.0/courses/",
            params=params,
            headers=headers,
            timeout=15,
        )

        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", [])
            if results:
                output = f"Udemy courses for '{keyword}':\n\n"
                for c in results:
                    title = c.get("title", "N/A")
                    url_path = c.get("url", "")
                    link = f"https://www.udemy.com{url_path}" if url_path else search_url
                    instructors = c.get("visible_instructors", [])
                    instructor_names = ", ".join(
                        i.get("display_name", "") for i in instructors
                    ) if instructors else "N/A"
                    rating = c.get("rating", "N/A")
                    reviews = c.get("num_reviews", 0)
                    students = c.get("num_subscribers", 0)
                    price = c.get("price", "N/A")

                    output += (
                        f"🛒 **{title}**\n"
                        f"   Instructor  : {instructor_names}\n"
                        f"   Rating      : ⭐ {rating} ({reviews:,} reviews) | {students:,} students\n"
                        f"   Price       : {price}\n"
                        f"   Link        : {link}\n\n"
                    )
                output += f"🔍 Browse all results: {search_url}"
                return output.strip()

        # Graceful fallback — Udemy blocks unauthenticated API access
        # Provide the user with direct curated search links instead
        encoded = keyword.replace(' ', '+')
        return (
            f"Udemy courses for '{keyword}':\n\n"
            f"Udemy's API requires authentication for programmatic access. "
            f"Here are direct links to find the best courses:\n\n"
            f"🛒 **Search Udemy for '{keyword}'**\n"
            f"   Link: {search_url}\n\n"
            f"🛒 **Top Rated '{keyword}' courses on Udemy**\n"
            f"   Link: https://www.udemy.com/courses/search/?q={encoded}&sort=highest-rated&rating=4.5\n\n"
            f"🛒 **Free '{keyword}' courses on Udemy**\n"
            f"   Link: https://www.udemy.com/courses/search/?q={encoded}&price=price-free\n\n"
            f"💡 Tip: Filter by 'Highest Rated' and check reviews before enrolling."
        )
    except Exception as e:
        search_url = f"https://www.udemy.com/courses/search/?q={keyword.replace(' ', '+')}"
        return (
            f"Error fetching Udemy courses: {str(e)}\n"
            f"👉 Search manually: {search_url}"
        )


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
    get_founder_info,
    search_nptel_courses,
    search_coursera_courses,
    search_udemy_courses,
]