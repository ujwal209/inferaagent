from langchain_core.tools import tool
from typing import Optional, List, Dict, Any
from app.models.schemas import *
import os
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Initialize Supabase Client
url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_KEY")
if not url or not key:
    raise ValueError("SUPABASE_URL or SUPABASE_KEY environment variable is not set")

supabase: Client = create_client(url, key)

# Helper for consistent output format
def format_results(data: Any) -> str:
    if not data:
        return "No results found."
    return str(data)

# Relevant tables for reference
RELEVANT_TABLES = [
    'aicte_branches', 'all_universities', 'branch_categories', 
    'branch_courses', 'branch_job_roles', 'branch_roadmaps', 
    'branch_salaries', 'branch_technical_domains', 
    'engineering_colleges', 'nirf_rankings_engineering', 
    'nirf_rankings_university'
]

@tool(args_schema=TableNameInput)
def list_tables() -> str:
    """List all available tables in the database."""
    return f"Available tables: {', '.join(RELEVANT_TABLES)}"

@tool(args_schema=TableNameInput)
def get_table_schema(table_name: str) -> str:
    """Inspect columns and data types for a specific table."""
    # Since we can't easily reflect schema via REST, we use the catalog info
    catalog = {
        'aicte_branches': 'branch_name, category',
        'all_universities': 'aishe_code, name, state, district, website, year_of_establishment, location',
        'branch_categories': 'category_name',
        'branch_courses': 'branch_name, job_role, course_name, platform, course_link',
        'branch_job_roles': 'branch_name, job_role',
        'branch_roadmaps': 'branch_name, project_ideas, internship_preparation',
        'branch_salaries': 'branch_name, freshers_salary_lpa, experienced_salary_lpa, salary_source_link, international_salary_usd, international_salary_source',
        'branch_technical_domains': 'branch_name, technical_domains',
        'engineering_colleges': 'aishe_code, name, state, website',
        'nirf_rankings_engineering': 'name, city, state, rank',
        'nirf_rankings_university': 'name, city, state, ranking'
    }
    if table_name in catalog:
        return f"Table {table_name} columns: {catalog[table_name]}"
    return f"Error: Table {table_name} not found."

@tool(args_schema=BranchSearchInput)
def get_branch_details(branch_name: str) -> str:
    """Get the AICTE category and general info for an engineering branch."""
    try:
        res = supabase.table('aicte_branches').select('*').ilike('branch_name', f'%{branch_name}%').execute()
        return format_results(res.data)
    except Exception as e:
        return f"Error: {e}"

@tool(args_schema=BranchSearchInput)
def get_job_roles_by_branch(branch_name: str) -> str:
    """Find specific job roles associated with an engineering branch."""
    try:
        res = supabase.table('branch_job_roles').select('job_role').ilike('branch_name', f'%{branch_name}%').execute()
        return format_results(res.data)
    except Exception as e:
        return f"Error: {e}"

@tool(args_schema=BranchSearchInput)
def get_salary_insights(branch_name: str) -> str:
    """Compare freshers vs experienced salaries for a branch."""
    try:
        res = supabase.table('branch_salaries').select('*').ilike('branch_name', f'%{branch_name}%').execute()
        return format_results(res.data)
    except Exception as e:
        return f"Error: {e}"

@tool(args_schema=BranchSearchInput)
def get_technical_domains(branch_name: str) -> str:
    """List the important technical domains for a branch."""
    try:
        res = supabase.table('branch_technical_domains').select('technical_domains').ilike('branch_name', f'%{branch_name}%').execute()
        return format_results(res.data)
    except Exception as e:
        return f"Error: {e}"

@tool(args_schema=BranchSearchInput)
def get_career_roadmaps(branch_name: str) -> str:
    """Get project ideas and internship preparation tips for a branch."""
    try:
        res = supabase.table('branch_roadmaps').select('project_ideas, internship_preparation').ilike('branch_name', f'%{branch_name}%').execute()
        return format_results(res.data)
    except Exception as e:
        return f"Error: {e}"

@tool(args_schema=BranchSearchInput)
def get_courses_by_branch(branch_name: str) -> str:
    """Find recommended courses for a specific engineering branch."""
    try:
        res = supabase.table('branch_courses').select('course_name, platform, job_role, course_link').ilike('branch_name', f'%{branch_name}%').limit(50).execute()
        return format_results(res.data)
    except Exception as e:
        return f"Error: {e}"

@tool(args_schema=PlatformSearchInput)
def get_courses_by_platform(platform: str) -> str:
    """Find all courses available on a specific platform."""
    try:
        res = supabase.table('branch_courses').select('course_name, branch_name, job_role, course_link').ilike('platform', f'%{platform}%').limit(50).execute()
        return format_results(res.data)
    except Exception as e:
        return f"Error: {e}"

@tool(args_schema=JobRoleSearchInput)
def get_courses_by_job_role(job_role: str) -> str:
    """Find courses tailored for a specific career or job role."""
    try:
        res = supabase.table('branch_courses').select('course_name, platform, branch_name, course_link').ilike('job_role', f'%{job_role}%').limit(50).execute()
        return format_results(res.data)
    except Exception as e:
        return f"Error: {e}"

@tool(args_schema=CourseSearchInput)
def search_courses_by_keyword(keyword: str) -> str:
    """Search for courses by keyword."""
    try:
        res = supabase.table('branch_courses').select('course_name, platform, branch_name, job_role, course_link').ilike('course_name', f'%{keyword}%').limit(50).execute()
        return format_results(res.data)
    except Exception as e:
        return f"Error: {e}"

@tool
def list_all_course_platforms() -> str:
    """List all available course platforms (Coursera, Udemy, etc.)."""
    try:
        res = supabase.table('branch_courses').select('platform').execute()
        platforms = sorted(list(set(r['platform'] for r in res.data if r['platform'])))
        return f"Available platforms: {', '.join(platforms)}"
    except Exception as e:
        return f"Error: {e}"

@tool(args_schema=StateSearchInput)
def find_colleges_by_state(state: str) -> str:
    """Search for engineering colleges in a specific state."""
    try:
        res = supabase.table('engineering_colleges').select('name, website').ilike('state', f'%{state}%').limit(100).execute()
        return format_results(res.data)
    except Exception as e:
        return f"Error: {e}"

@tool(args_schema=CollegeSearchInput)
def get_college_info(college_name: str) -> str:
    """Get AISHE code, state, and website for a specific college."""
    try:
        res = supabase.table('engineering_colleges').select('*').ilike('name', f'%{college_name}%').execute()
        return format_results(res.data)
    except Exception as e:
        return f"Error: {e}"

@tool(args_schema=RankingSearchInput)
def get_engineering_rankings(state: Optional[str] = None, limit: int = 50) -> str:
    """Get the top NIRF-ranked engineering institutions. Optionally filter by state."""
    try:
        query = supabase.table('nirf_rankings_engineering').select('rank, name, city, state')
        if state:
            query = query.ilike('state', f'%{state}%')
        res = query.order('rank', desc=False).limit(limit).execute()
        return format_results(res.data)
    except Exception as e:
        return f"Error: {e}"

@tool(args_schema=RankingSearchInput)
def get_university_rankings(state: Optional[str] = None, limit: int = 50) -> str:
    """Get the top NIRF-ranked universities. Optionally filter by state or ranking number."""
    try:
        query = supabase.table('nirf_rankings_university').select('ranking, name, city, state')
        if state:
            query = query.ilike('state', f'%{state}%')
        res = query.order('ranking', desc=False).limit(limit).execute()
        return format_results(res.data)
    except Exception as e:
        return f"Error: {e}"

@tool(args_schema=UniversitySearchInput)
def get_university_info(name: str) -> str:
    """Get detailed info about a university including AISHE code, website, and location."""
    try:
        res = supabase.table('all_universities').select('*').ilike('name', f'%{name}%').limit(10).execute()
        return format_results(res.data)
    except Exception as e:
        return f"Error: {e}"

@tool(args_schema=CategorySearchInput)
def get_branches_by_category(category_name: str) -> str:
    """Find all engineering branches belonging to a specific category (e.g., 'Information Technology')."""
    try:
        res = supabase.table('aicte_branches').select('branch_name').ilike('category', f'%{category_name}%').limit(100).execute()
        return format_results(res.data)
    except Exception as e:
        return f"Error: {e}"

@tool
def get_full_database_info() -> str:
    """Provides a high-level summary of all tables and what data they contain. Use this if unsure which table to query."""
    return """
    Available Database Tables:
    1. aicte_branches: branch_name, category
    2. all_universities: aishe_code, name, state, district, website, year_of_establishment, location
    3. branch_categories: category_name
    4. branch_courses: branch_name, job_role, course_name, platform, course_link
    5. branch_job_roles: branch_name, job_role
    6. branch_roadmaps: branch_name, project_ideas, internship_preparation
    7. branch_salaries: branch_name, freshers_salary_lpa, experienced_salary_lpa, salary_source_link, international_salary_usd, international_salary_source
    8. branch_technical_domains: branch_name, technical_domains
    9. engineering_colleges: aishe_code, name, state, website
    10. nirf_rankings_engineering: name, city, state, rank
    11. nirf_rankings_university: name, city, state, ranking
    """

@tool(args_schema=SalaryFilterInput)
def get_top_paying_branches(min_salary_lpa: float = 8.0) -> str:
    """Find engineering branches with the highest starting salaries (> min_salary_lpa)."""
    try:
        res = supabase.table('branch_salaries').select('branch_name, freshers_salary_lpa, experienced_salary_lpa').execute()
        
        filtered_data = []
        import re
        for item in res.data:
            salary_str = item.get('freshers_salary_lpa', '0')
            # Extract first numeric part
            match = re.search(r'^[0-9.]+', str(salary_str))
            if match:
                salary_val = float(match.group())
                if salary_val >= min_salary_lpa:
                    item['salary_val'] = salary_val
                    filtered_data.append(item)
        
        # Sort by salary_val desc
        filtered_data.sort(key=lambda x: x['salary_val'], reverse=True)
        # Remove helper key and limit
        results = filtered_data[:20]
        for r in results: r.pop('salary_val', None)
        
        return format_results(results)
    except Exception as e:
        return f"Error: {e}"

@tool(args_schema=BranchSearchInput)
def get_international_salary_insights(branch_name: str) -> str:
    """Get international salary data (USD) and source links for a specific branch."""
    try:
        res = supabase.table('branch_salaries').select('branch_name, international_salary_usd, international_salary_source').ilike('branch_name', f'%{branch_name}%').not_.is_('international_salary_usd', 'null').execute()
        return format_results(res.data)
    except Exception as e:
        return f"Error: {e}"

@tool(args_schema=ComparisonInput)
def compare_two_branches(branch_a: str, branch_b: str) -> str:
    """Compare two engineering branches side-by-side (Salaries, Domains, Jobs)."""
    try:
        # We'll do multiple queries as joins are restricted in Supabase client REST
        results = {}
        for b in [branch_a, branch_b]:
            sal = supabase.table('branch_salaries').select('*').ilike('branch_name', f'%{b}%').execute()
            dom = supabase.table('branch_technical_domains').select('*').ilike('branch_name', f'%{b}%').execute()
            road = supabase.table('branch_roadmaps').select('*').ilike('branch_name', f'%{b}%').execute()
            results[b] = {
                "salaries": sal.data,
                "domains": dom.data,
                "roadmaps": road.data
            }
        return format_results(results)
    except Exception as e:
        return f"Error: {e}"

@tool(args_schema=CitySearchInput)
def find_colleges_by_city(city: str, state: Optional[str] = None) -> str:
    """Search for NIRF-ranked colleges and universities in a specific city."""
    try:
        query_eng = supabase.table('nirf_rankings_engineering').select('rank, name, city, state').ilike('city', f'%{city}%')
        if state: query_eng = query_eng.ilike('state', f'%{state}%')
        res_eng = query_eng.limit(50).execute()
        
        query_uni = supabase.table('nirf_rankings_university').select('ranking, name, city, state').ilike('city', f'%{city}%')
        if state: query_uni = query_uni.ilike('state', f'%{state}%')
        res_uni = query_uni.limit(50).execute()
        
        return f"Engineering: {res_eng.data}\n\nUniversities: {res_uni.data}"
    except Exception as e:
        return f"Error: {e}"

@tool(args_schema=DistrictSearchInput)
def get_universities_by_district(district: str, state: str) -> str:
    """List all universities in a specific district and state."""
    try:
        res = supabase.table('all_universities').select('name, website, location').ilike('district', f'%{district}%').ilike('state', f'%{state}%').limit(50).execute()
        return format_results(res.data)
    except Exception as e:
        return f"Error: {e}"

@tool(args_schema=CourseSearchInput)
def get_branches_by_technical_domain(keyword: str) -> str:
    """Search for engineering branches based on a technical domain keyword (e.g., 'VLSI', 'AI', 'Power')."""
    try:
        res = supabase.table('branch_technical_domains').select('branch_name, technical_domains').ilike('technical_domains', f'%{keyword}%').limit(50).execute()
        return format_results(res.data)
    except Exception as e:
        return f"Error: {e}"

@tool(args_schema=CourseSearchInput)
def search_project_ideas(keyword: str) -> str:
    """Find project ideas and preparation tips across all branches using a keyword."""
    try:
        res = supabase.table('branch_roadmaps').select('branch_name, project_ideas').ilike('project_ideas', f'%{keyword}%').limit(50).execute()
        return format_results(res.data)
    except Exception as e:
        return f"Error: {e}"

@tool(args_schema=CollegeComparisonInput)
def compare_colleges(college_a: str, college_b: str) -> str:
    """Compare two colleges or universities (Rankings, State, Website)."""
    try:
        # Union equivalent in REST is two queries
        e1 = supabase.table('nirf_rankings_engineering').select('name, city, state, rank').ilike('name', f'%{college_a}%').execute()
        e2 = supabase.table('nirf_rankings_engineering').select('name, city, state, rank').ilike('name', f'%{college_b}%').execute()
        u1 = supabase.table('nirf_rankings_university').select('name, city, state, ranking').ilike('name', f'%{college_a}%').execute()
        u2 = supabase.table('nirf_rankings_university').select('name, city, state, ranking').ilike('name', f'%{college_b}%').execute()
        
        return format_results({
            "engineering": e1.data + e2.data,
            "universities": u1.data + u2.data
        })
    except Exception as e:
        return f"Error: {e}"

@tool(args_schema=YearRangeInput)
def get_universities_by_establishment_year(start_year: int, end_year: int) -> str:
    """Find universities established within a specific year range."""
    try:
        res = supabase.table('all_universities').select('name, state, year_of_establishment, website').execute()
        # Filter in Python due to text-to-int cast requirement
        filtered = []
        for r in res.data:
            year_str = r.get('year_of_establishment')
            if year_str and year_str.isdigit():
                year = int(year_str)
                if start_year <= year <= end_year:
                    filtered.append(r)
        
        # Sort and limit
        filtered.sort(key=lambda x: int(x['year_of_establishment']))
        return format_results(filtered[:50])
    except Exception as e:
        return f"Error: {e}"

@tool(args_schema=QueryInput)
def execute_advanced_sql(query: str) -> str:
    """Execute raw SQL SELECT query. Note: Limited support via Supabase URL."""
    return "Error: Raw SQL execution is not supported via Supabase URL. Use specific tools instead."

tool_list = [
    list_tables, get_table_schema, get_branch_details, get_job_roles_by_branch,
    get_salary_insights, get_technical_domains, get_career_roadmaps,
    get_courses_by_branch, get_courses_by_platform, get_courses_by_job_role,
    search_courses_by_keyword, list_all_course_platforms,
    find_colleges_by_state, get_college_info,
    get_engineering_rankings, get_university_rankings, 
    get_university_info, get_branches_by_category,
    get_top_paying_branches, get_international_salary_insights,
    compare_two_branches, find_colleges_by_city,
    get_universities_by_district, get_branches_by_technical_domain,
    search_project_ideas, compare_colleges, get_universities_by_establishment_year,
    get_full_database_info, execute_advanced_sql
]
