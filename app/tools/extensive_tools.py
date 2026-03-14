from langchain_core.tools import tool
from typing import Optional, List, Dict, Any
from app.models.schemas import *
import os
import re
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Initialize Supabase Client
url: str = os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL")
key: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")

if not url or not key:
    print("CRITICAL WARNING: Supabase Environment Variables are missing!")
    url = "https://placeholder.supabase.co"
    key = "placeholder_key"

supabase: Client = create_client(url, key)

# ==========================================
# TAVILY WEB SEARCH ENGINE (Load Balanced)
# ==========================================
import random
TAVILY_KEYS = [k.strip() for k in os.getenv("TAVILY_API_KEYS", os.getenv("TAVILY_API_KEY", "")).split(",") if k.strip()]
random.shuffle(TAVILY_KEYS) # Distribute initial load across restarts

class TavilySearcher:
    def __init__(self, keys: list[str]):
        self.keys = keys
        self.current_index = 0
    
    def search(self, query: str, search_depth: str = "advanced") -> list:
        if not self.keys:
            return []
        
        import requests
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
                    "max_results": 7 # Pulled up to 7 to give the AI more context
                }
                response = requests.post("https://api.tavily.com/search", json=payload, timeout=15)
                if response.status_code == 200:
                    return response.json().get("results", [])
            except:
                pass
        return []

tavily_engine = TavilySearcher(TAVILY_KEYS)

# ==========================================
# CORE ENGINE HELPERS
# ==========================================

def format_results(data: Any, tool_name: str = "unknown") -> str:
    if not data:
        return f"No results found for '{tool_name}'. WARNING: YOU MUST USE THE 'web_search' TOOL NOW."
    
    if isinstance(data, list):
        count = len(data)
        if count > 20:
            return str(data[:20]) + f"\n\n(Showing top 20 of {count} results found.)"
    
    return str(data)

BRANCH_ALIAS_MAP = {
    "cse": "Computer Science and Engineering",
    "cs": "Computer Science and Engineering",
    "eee": "Electrical and Electronics Engineering",
    "ee": "Electrical Engineering",
    "ece": "Electronics and Communication Engineering",
    "electronics": "Electronics and Communication Engineering",
    "it": "Information Technology",
    "ai": "Artificial Intelligence and Data Science",
    "ds": "Data Science",
    "mech": "Mechanical Engineering",
    "aero": "Aerospace Engineering",
    "civil": "Civil Engineering",
    "chem": "Chemical Engineering",
    "auto": "Automobile Engineering",
    "biotech": "Biotechnology Engineering",
    "mining": "Mining Engineering",
    "marine": "Marine Engineering",
    "robotics": "Robotics and Automation",
    "cyber": "Cyber Security",
    "software": "Software Engineering"
}

def resolve_branch(name: str) -> str:
    clean = str(name).lower().strip().replace("engineering", "").strip()
    return BRANCH_ALIAS_MAP.get(clean, name)

def db_query(table: str, branch: str = None, col: str = "branch_name", limit: int = 15):
    """Universal query helper with fuzzy branch support."""
    try:
        q = supabase.table(table).select("*")
        if branch:
            canonical = resolve_branch(branch)
            q = q.ilike(col, f"%{canonical}%")
        res = q.order('id').limit(limit).execute()
        return res.data
    except Exception as e:
        return []

# ==========================================
# 1. PRIMARY WEB SEARCH TOOL (TOP PRIORITY)
# ==========================================

@tool(args_schema=CourseSearchInput)
def web_search(keyword: str) -> str:
    """
    [PRIMARY TOOL] Use this FIRST for almost all user queries, especially for finding courses, 
    latest industry salaries, emerging career paths, and real-time university/college news.
    DO NOT hallucinate data; if the database tools fail, you MUST call this tool.
    """
    # Force Indian Context to prevent hallucinations of US/European data
    search_query = keyword
    if "india" not in keyword.lower() and "nptel" not in keyword.lower():
        search_query = f"{keyword} engineering in India"
        
    results = tavily_engine.search(search_query)
    
    if not results:
        return "Live web search returned no results. Tell the user you cannot find the information."
        
    formatted = f"LIVE WEB SEARCH RESULTS FOR '{search_query}':\n\n"
    for r in results:
        formatted += f"- Title: {r.get('title')}\n  URL: {r.get('url')}\n  Content Snippet: {r.get('content')}\n\n"
    
    formatted += "\nINSTRUCTION: Extract the exact URLs and Titles from above and present them to the user in a Markdown Table."
    return formatted

# ==========================================
# 2. BRANCH META TOOLS 
# ==========================================

@tool
def list_all_branches() -> str:
    """List every single AICTE recognized engineering branch available in the system."""
    res = supabase.table('aicte_branches').select('branch_name').order('branch_name').execute()
    return format_results([r['branch_name'] for r in res.data if r.get('branch_name')], "all_branches")

@tool(args_schema=BranchSearchInput)
def search_branch_details(branch_name: str) -> str:
    """Get the AICTE category and full name of a branch."""
    return format_results(db_query('aicte_branches', branch_name), "branch_details")

@tool(args_schema=BranchSearchInput)
def get_branch_category(branch_name: str) -> str:
    """Find the specific category (e.g. 'Electronics', 'Chemical') a branch belongs to."""
    data = db_query('aicte_branches', branch_name)
    if isinstance(data, list) and len(data) > 0:
        return f"Branch: {data[0].get('branch_name')} | Category: {data[0].get('category')}"
    return "Category not found. USE web_search tool."

@tool(args_schema=CategorySearchInput)
def get_branches_by_category(category_name: str) -> str:
    """List all branches under a specific category like 'Information Technology'."""
    res = supabase.table('aicte_branches').select('branch_name').ilike('category', f'%{category_name}%').execute()
    return format_results(res.data, "branches_by_category")

@tool(args_schema=BranchSearchInput)
def get_branch_technical_domains(branch_name: str) -> str:
    """Find core technical domains and skill areas for a branch (e.g. VLSI, Thermodynamics)."""
    return format_results(db_query('branch_technical_domains', branch_name), "tech_domains")

@tool(args_schema=CourseSearchInput)
def get_branches_by_technical_domain(keyword: str) -> str:
    """Search for branches specializing in a specific domain (e.g. searching 'AI' will find CSE-AI, IT, etc)."""
    return format_results(db_query('branch_technical_domains', keyword, col='technical_domains'), "branches_by_domain")

# ==========================================
# 3. CAREER & SALARY TOOLS
# ==========================================

@tool(args_schema=BranchSearchInput)
def search_job_roles_by_branch(branch_name: str) -> str:
    """List career roles and job titles for graduates of a specific branch."""
    return format_results(db_query('branch_job_roles', branch_name), "job_roles")

@tool(args_schema=JobRoleSearchInput)
def find_branches_by_job_role(job_role: str) -> str:
    """Find which engineering branches lead to a specific job role (e.g. 'Data Scientist')."""
    return format_results(db_query('branch_job_roles', job_role, col='job_role'), "branches_by_role")

@tool(args_schema=BranchSearchInput)
def search_salary_insights(branch_name: str) -> str:
    """Get domestic salary data (LPA) for freshers and experienced professionals in a branch."""
    return format_results(db_query('branch_salaries', branch_name), "domestic_salary")

@tool(args_schema=BranchSearchInput)
def get_international_salary_insights(branch_name: str) -> str:
    """Get international salary benchmarks (USD) and source links for a specific branch."""
    return format_results(db_query('branch_salaries', branch_name), "intl_salary")

@tool(args_schema=SalaryFilterInput)
def get_top_paying_branches(min_salary_lpa: float = 8.0) -> str:
    """Find high-paying engineering branches with freshers' salary above a certain LPA."""
    res = supabase.table('branch_salaries').select('*').execute()
    high_pay = []
    for r in res.data:
        match = re.search(r'([0-9.]+)', str(r.get('freshers_salary_lpa', '0')))
        if match and float(match.group(1)) >= min_salary_lpa:
            high_pay.append(r)
    return format_results(sorted(high_pay, key=lambda x: x.get('freshers_salary_lpa', '0'), reverse=True), "top_paying")

@tool(args_schema=BranchSearchInput)
def search_career_roadmaps(branch_name: str) -> str:
    """Get the full career path roadmap and preparation guide for a branch."""
    return format_results(db_query('branch_roadmaps', branch_name), "roadmaps")

@tool(args_schema=BranchSearchInput)
def get_project_ideas_by_branch(branch_name: str) -> str:
    """Get specific project ideas (Mini/Major) for a specific engineering branch."""
    res = db_query('branch_roadmaps', branch_name)
    if isinstance(res, list) and res:
        return f"Project Ideas for {branch_name}:\n{res[0].get('project_ideas')}"
    return "No project ideas found. USE web_search tool."

@tool(args_schema=CourseSearchInput)
def search_project_ideas_globally(keyword: str) -> str:
    """Search for specific project terms (e.g. 'Robotics') across all engineering fields."""
    return format_results(db_query('branch_roadmaps', keyword, col='project_ideas'), "global_projects")

@tool(args_schema=BranchSearchInput)
def get_internship_prep_tips(branch_name: str) -> str:
    """Get branch-specific tips for securing internships and cracking interviews."""
    res = db_query('branch_roadmaps', branch_name)
    if isinstance(res, list) and res:
        return f"Internship Guide for {branch_name}:\n{res[0].get('internship_preparation')}"
    return "No internship tips found. USE web_search tool."

@tool(args_schema=BranchSearchInput)
def get_branch_career_summary(branch_name: str) -> str:
    """Get a combined view of Job Roles + Salaries + Domains for a branch."""
    results = {
        "roles": db_query('branch_job_roles', branch_name),
        "salaries": db_query('branch_salaries', branch_name),
        "domains": db_query('branch_technical_domains', branch_name)
    }
    return str(results)

# ==========================================
# 4. EDUCATION & COURSE TOOLS
# ==========================================

@tool(args_schema=BranchSearchInput)
def get_courses(branch_name: str) -> str:
    """If a user asks for courses, ALWAYS prioritize web_search directly. This acts as a fallback."""
    db_results = db_query('branch_courses', branch_name, limit=10)
    if db_results:
        return format_results(db_results, "get_courses")
    return "Database has no courses for this branch. YOU MUST EXECUTE `web_search` NOW."

@tool(args_schema=JobRoleSearchInput)
def get_courses_by_job_role(job_role: str) -> str:
    """Find top courses tailored for a specific career role (e.g. 'Cloud Engineer')."""
    return format_results(db_query('branch_courses', job_role, col='job_role'), "role_courses")

@tool(args_schema=PlatformSearchInput)
def get_courses_by_platform(platform: str) -> str:
    """Browse courses available on specific platforms like 'Coursera', 'Udemy', or 'NPTEL'."""
    return format_results(db_query('branch_courses', platform, col='platform'), "platform_courses")

@tool(args_schema=CourseSearchInput)
def search_courses_by_keyword(keyword: str) -> str:
    """Search for specific course titles (e.g. 'Python for beginners', 'Hydraulics') globally."""
    return format_results(db_query('branch_courses', keyword, col='course_name'), "keyword_courses")

@tool
def list_all_course_platforms() -> str:
    """Get a list of all course providers mentioned in the database."""
    res = supabase.table('branch_courses').select('platform').execute()
    platforms = sorted(list(set(r['platform'] for r in res.data if r['platform'])))
    return f"Available platforms: {', '.join(platforms)}"

@tool(args_schema=ComparisonInput)
def get_courses_filtered(branch_name: str, platform: str) -> str:
    """Filter courses by both Branch AND Platform (e.g. NPTEL courses for Mechanical)."""
    res = supabase.table('branch_courses').select('*').ilike('branch_name', f'%{resolve_branch(branch_name)}%').ilike('platform', f'%{platform}%').execute()
    return format_results(res.data, "filtered_courses")

@tool(args_schema=BranchSearchInput)
def get_course_count_by_branch(branch_name: str) -> str:
    """Count how many total courses are listed for a specific branch."""
    data = db_query('branch_courses', branch_name)
    return f"Total courses found for {branch_name}: {len(data)}"

@tool
def get_top_rated_courses() -> str:
    """List the most featured/premium courses across all engineering sectors."""
    return format_results(supabase.table('branch_courses').select('*').limit(20).execute().data, "top_courses")

@tool
def get_free_learning_resources() -> str:
    """List free/government-backed courses (NPTEL, SWAYAM) for engineering students."""
    return get_courses_by_platform("NPTEL")

@tool(args_schema=JobRoleSearchInput)
def get_career_specific_courses(job_role: str) -> str:
    """Get courses specifically linked to achieving a certain job role."""
    return format_results(db_query('branch_courses', job_role, col='job_role'), "career_specific")

# ==========================================
# 5. INSTITUTION & RANKING TOOLS
# ==========================================

@tool(args_schema=CollegeSearchInput)
def get_college_info(college_name: str) -> str:
    """Basic search for a college to find its AISHE code and State."""
    return format_results(db_query('engineering_colleges', college_name, col='name'), "college_basic")

@tool(args_schema=CollegeSearchInput)
def get_college_details(college_name: str) -> str:
    """Get full details including website, rank, and location for a specific college."""
    coll = db_query('engineering_colleges', college_name, col='name')
    rank = db_query('nirf_rankings_engineering', college_name, col='name')
    return str({"college_data": coll, "rankings": rank})

@tool(args_schema=StateSearchInput)
def find_colleges_by_state(state: str) -> str:
    """List engineering colleges located in a specific Indian state."""
    return format_results(db_query('engineering_colleges', state, col='state'), "state_colleges")

@tool(args_schema=CitySearchInput)
def find_colleges_by_city(city: str) -> str:
    """List engineering colleges located in a specific city."""
    return format_results(db_query('nirf_rankings_engineering', city, col='city'), "city_colleges")

@tool(args_schema=RankingSearchInput)
def get_engineering_rankings(state: Optional[str] = None, limit: int = 15) -> str:
    """List top NIRF ranked engineering institutions. Can be filtered by state."""
    q = supabase.table('nirf_rankings_engineering').select('*').order('rank')
    if state: q = q.ilike('state', f'%{state}%')
    return format_results(q.limit(limit).execute().data, "eng_rankings")

@tool(args_schema=YearRangeInput)
def get_engineering_colleges_in_rank_range(start_rank: int, end_rank: int) -> str:
    """Filter engineering colleges by a specific NIRF rank range (e.g. top 100)."""
    res = supabase.table('nirf_rankings_engineering').select('*').execute()
    filtered = [r for r in res.data if r['rank'].isdigit() and start_rank <= int(r['rank']) <= end_rank]
    return format_results(sorted(filtered, key=lambda x: int(x['rank'])), "eng_rank_range")

@tool(args_schema=UniversitySearchInput)
def get_university_info(name: str) -> str:
    """Search for a university to find its AISHE code, type, and location."""
    return format_results(db_query('all_universities', name, col='name'), "uni_basic")

@tool(args_schema=StateSearchInput)
def get_universities_by_state(state: str) -> str:
    """List all universities in a specific state."""
    return format_results(db_query('all_universities', state, col='state'), "state_unis")

@tool(args_schema=DistrictSearchInput)
def get_universities_by_district(district: str, state: str) -> str:
    """Find universities within a specific District (e.g. Pune, Bangalore)."""
    res = supabase.table('all_universities').select('*').ilike('district', f'%{district}%').ilike('state', f'%{state}%').execute()
    return format_results(res.data, "district_unis")

@tool(args_schema=RankingSearchInput)
def get_university_rankings(state: Optional[str] = None, limit: int = 15) -> str:
    """List top NIRF ranked universities. Can be filtered by state."""
    q = supabase.table('nirf_rankings_university').select('*').order('ranking')
    if state: q = q.ilike('state', f'%{state}%')
    return format_results(q.limit(limit).execute().data, "uni_rankings")

@tool(args_schema=YearRangeInput)
def get_universities_by_establishment_year(start_year: int, end_year: int) -> str:
    """Find universities established during a specific time period (e.g. 1950 to 1960)."""
    res = supabase.table('all_universities').select('*').execute()
    filtered = []
    for r in res.data:
        y = str(r.get('year_of_establishment', ''))
        if y.isdigit() and start_year <= int(y) <= end_year:
            filtered.append(r)
    return format_results(sorted(filtered, key=lambda x: int(x['year_of_establishment'])), "uni_year")

@tool(args_schema=CollegeSearchInput)
def find_institution_by_aishe(aishe_code: str) -> str:
    """Lookup a college or university using its unique AISHE code."""
    c = supabase.table('engineering_colleges').select('*').eq('aishe_code', aishe_code).execute().data
    u = supabase.table('all_universities').select('*').eq('aishe_code', aishe_code).execute().data
    return str({"college": c, "university": u})

@tool
def list_available_categories() -> str:
    """List all broad engineering categories (Civil, CSE, Bio, etc.) from AICTE."""
    res = supabase.table('branch_categories').select('*').execute()
    return format_results([r['category_name'] for r in res.data], "categories")

@tool(args_schema=StateSearchInput)
def get_university_count_in_state(state: str) -> str:
    """Get the total number of universities registered in a given state."""
    res = supabase.table('all_universities').select('count', count='exact').ilike('state', f'%{state}%').execute()
    return f"Total universities in {state}: {res.count}"

@tool(args_schema=StateSearchInput)
def get_college_count_in_state(state: str) -> str:
    """Get the total number of engineering colleges registered in a given state."""
    res = supabase.table('engineering_colleges').select('count', count='exact').ilike('state', f'%{state}%').execute()
    return f"Total colleges in {state}: {res.count}"

# ==========================================
# 6. COMPARISON & GLOBAL SEARCH 
# ==========================================

@tool(args_schema=ComparisonInput)
def compare_two_branches(branch_a: str, branch_b: str) -> str:
    """Side-by-side comparison of two branches including Salary, Roadmaps, and Skills."""
    a = get_branch_career_summary(branch_a)
    b = get_branch_career_summary(branch_b)
    return f"Comparison between {branch_a} and {branch_b}:\n\n{branch_a} Details:\n{a}\n\n{branch_b} Details:\n{b}"

@tool(args_schema=ComparisonInput)
def compare_two_colleges(college_a: str, college_b: str) -> str:
    """Side-by-side comparison of two institutions based on NIRF rank, Location, and Website."""
    a = get_college_details(college_a)
    b = get_college_details(college_b)
    return f"Comparison:\nInstitution A: {a}\nInstitution B: {b}"

@tool(args_schema=CourseSearchInput)
def general_search(query: str) -> str:
    """Emergency cross-database search for any keyword. Use this if specific tools fail."""
    results = {
        "branches": db_query('aicte_branches', query, col='branch_name', limit=5),
        "colleges": db_query('engineering_colleges', query, col='name', limit=5),
        "rankings": db_query('nirf_rankings_engineering', query, col='name', limit=5)
    }
    return format_results(results, "general_search")

@tool(args_schema=BranchSearchInput)
def analyze_career_potential(branch_name: str) -> str:
    """Comprehensive analysis of a branch: Category + Technical Skills + Job Roles + Career Value."""
    return get_branch_career_summary(branch_name)

@tool(args_schema=CourseSearchInput)
def search_everything_by_keyword(keyword: str) -> str:
    """Global keyword search across all data (Colleges, Courses, Branches, Salaries)."""
    return general_search(keyword)

@tool(args_schema=CategorySearchInput)
def get_top_colleges_by_category(category: str) -> str:
    """Search for top institutions that specifically specialize in a category like 'Aeronautical'."""
    return format_results(db_query('nirf_rankings_engineering', category, col='name'), "category_to_college")

# ==========================================
# 7. SYSTEM & META TOOLS
# ==========================================

@tool
def list_database_tables() -> str:
    """List all available data tables for debugging or internal lookup."""
    return "Tables: branch_courses, aicte_branches, all_universities, engineering_colleges, branch_roadmaps, branch_salaries, nirf_rankings_engineering, nirf_rankings_university, branch_technical_domains, branch_job_roles"

@tool(args_schema=TableNameInput)
def explain_table_fields(table_name: str) -> str:
    """Get the list of columns available in a specific table to understand the data structure."""
    catalog = {
        "aicte_branches": "branch_name, category",
        "branch_courses": "branch_name, job_role, course_name, platform, course_link",
        "branch_salaries": "freshers_salary_lpa, experienced_salary_lpa, intl_salary_usd",
        "all_universities": "name, aishe_code, state, website, established_year",
        "nirf_rankings": "rank, institution_name, city, state"
    }
    return catalog.get(table_name, "Table details not found.")

@tool(args_schema=StateSearchInput)
def get_top_colleges_by_state(state: str) -> str:
    """Show the highest NIRF-ranked colleges in a given state."""
    res = supabase.table('nirf_rankings_engineering').select('*').ilike('state', f'%{state}%').order('rank').limit(10).execute()
    return format_results(res.data, "top_state_colleges")

@tool(args_schema=BranchSearchInput)
def get_high_growth_job_roles(branch_name: str) -> str:
    """List specialized job roles for a branch that typically offer high salary growth."""
    return format_results(db_query('branch_job_roles', branch_name, limit=10), "growth_roles")

@tool(args_schema=CategorySearchInput)
def list_branches_in_domain(category: str) -> str:
    """List all AICTE branches belonging to a domain like 'Management' or 'Technology'."""
    return get_branches_by_category(category)

@tool(args_schema=UniversitySearchInput)
def search_universities_globally(keyword: str) -> str:
    """Find universities by any part of their name or location."""
    return format_results(db_query('all_universities', keyword, col='name'), "uni_search")

@tool
def get_system_overview() -> str:
    """Provides a map of what data lives where (e.g. where to find salaries vs where to find rankings)."""
    return """
    DATA MAP:
    - Branch Info: 'aicte_branches', 'branch_categories'
    - Careers/Pay: 'branch_salaries', 'branch_job_roles'
    - Learning: 'branch_courses', 'branch_roadmaps'
    - Skills: 'branch_technical_domains'
    - Institutions: 'all_universities', 'engineering_colleges'
    - Rankings: 'nirf_rankings_engineering', 'nirf_rankings_university'
    """

@tool
def get_database_stats() -> str:
    """Get the total count of records across all major tables for quick verification."""
    try:
        stats = {
            "branches": supabase.table('aicte_branches').select('count', count='exact').execute().count,
            "courses": supabase.table('branch_courses').select('count', count='exact').execute().count,
            "colleges": supabase.table('engineering_colleges').select('count', count='exact').execute().count,
            "rankings": supabase.table('nirf_rankings_engineering').select('count', count='exact').execute().count
        }
        return f"Database Statistics: {stats}"
    except:
        return "Could not fetch stats."

# ==========================================
# TOOL LIST EXPORT (60+ Tools)
# ==========================================

tool_list = [
    # Top Priority
    web_search,
    
    # Branch & Meta
    list_all_branches, search_branch_details, get_branch_category, get_branches_by_category,
    get_branch_technical_domains, get_branches_by_technical_domain, list_available_categories,
    list_branches_in_domain,
    
    # Career & Economy
    search_job_roles_by_branch, find_branches_by_job_role, search_salary_insights,
    get_international_salary_insights, get_top_paying_branches, get_high_growth_job_roles,
    get_branch_career_summary, analyze_career_potential,
    
    # Roadmap & Skills
    search_career_roadmaps, get_project_ideas_by_branch, search_project_ideas_globally, 
    get_internship_prep_tips, 
    
    # Education & Learning
    get_courses, get_courses_by_job_role, get_courses_by_platform,
    search_courses_by_keyword, list_all_course_platforms, get_courses_filtered,
    get_course_count_by_branch, get_top_rated_courses, get_free_learning_resources,
    get_career_specific_courses,
    
    # Institutions (Colleges)
    get_college_info, get_college_details, find_colleges_by_state, find_colleges_by_city,
    get_top_colleges_by_state, get_college_count_in_state, find_institution_by_aishe,
    
    # Institutions (Universities)
    get_university_info, get_universities_by_state, get_universities_by_district,
    get_universities_by_establishment_year, get_university_count_in_state, search_universities_globally,
    
    # Rankings
    get_engineering_rankings, get_engineering_colleges_in_rank_range, get_university_rankings,
    get_top_colleges_by_category,
    
    # Global & Logic
    compare_two_branches, compare_two_colleges, general_search, 
    search_everything_by_keyword, list_database_tables, explain_table_fields,
    get_system_overview, get_database_stats
]