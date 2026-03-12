from langchain_core.tools import tool
from typing import Optional
from app.models.schemas import *
import os
from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv

# Load environment
load_dotenv()

_db = None

def get_db():
    global _db
    if _db is None:
        DATABASE_URL = os.getenv("DATABASE_URL")
        if not DATABASE_URL:
            raise ValueError("DATABASE_URL environment variable is not set")
        
        # List specific tables to avoid expensive full database reflection
        RELEVANT_TABLES = [
            'aicte_branches', 'all_universities', 'branch_categories', 
            'branch_courses', 'branch_job_roles', 'branch_roadmaps', 
            'branch_salaries', 'branch_technical_domains', 
            'engineering_colleges', 'nirf_rankings_engineering', 
            'nirf_rankings_university'
        ]
        _db = SQLDatabase.from_uri(
            DATABASE_URL, 
            include_tables=RELEVANT_TABLES,
            engine_args={
                "connect_args": {
                    "sslmode": "require",
                    "connect_timeout": 10
                }
            }
        )
    return _db

@tool(args_schema=TableNameInput)
def list_tables() -> str:
    """List all available tables in the database."""
    try:
        tables = get_db().get_usable_table_names()
        return f"Available tables: {', '.join(tables)}"
    except Exception as e:
        return f"Error: {e}"

@tool(args_schema=TableNameInput)
def get_table_schema(table_name: str) -> str:
    """Inspect columns and data types for a specific table."""
    try:
        return get_db().get_table_info([table_name])
    except Exception as e:
        return f"Error: {e}"

@tool(args_schema=BranchSearchInput)
def get_branch_details(branch_name: str) -> str:
    """Get the AICTE category and general info for an engineering branch."""
    query = f"SELECT * FROM aicte_branches WHERE branch_name ILIKE '%{branch_name}%';"
    return str(get_db().run(query))

@tool(args_schema=BranchSearchInput)
def get_job_roles_by_branch(branch_name: str) -> str:
    """Find specific job roles associated with an engineering branch."""
    query = f"SELECT job_role FROM branch_job_roles WHERE branch_name ILIKE '%{branch_name}%';"
    return str(get_db().run(query))

@tool(args_schema=BranchSearchInput)
def get_salary_insights(branch_name: str) -> str:
    """Compare freshers vs experienced salaries for a branch."""
    query = f"SELECT * FROM branch_salaries WHERE branch_name ILIKE '%{branch_name}%';"
    return str(get_db().run(query))

@tool(args_schema=BranchSearchInput)
def get_technical_domains(branch_name: str) -> str:
    """List the important technical domains for a branch."""
    query = f"SELECT technical_domains FROM branch_technical_domains WHERE branch_name ILIKE '%{branch_name}%';"
    return str(get_db().run(query))

@tool(args_schema=BranchSearchInput)
def get_career_roadmaps(branch_name: str) -> str:
    """Get project ideas and internship preparation tips for a branch."""
    query = f"SELECT project_ideas, internship_preparation FROM branch_roadmaps WHERE branch_name ILIKE '%{branch_name}%';"
    return str(get_db().run(query))

@tool(args_schema=BranchSearchInput)
def get_courses_by_branch(branch_name: str) -> str:
    """Find recommended courses for a specific engineering branch."""
    query = f"SELECT course_name, platform, job_role, course_link FROM branch_courses WHERE branch_name ILIKE '%{branch_name}%' LIMIT 50;"
    return str(get_db().run(query))

@tool(args_schema=PlatformSearchInput)
def get_courses_by_platform(platform: str) -> str:
    """Find all courses available on a specific platform."""
    query = f"SELECT course_name, branch_name, job_role, course_link FROM branch_courses WHERE platform ILIKE '%{platform}%' LIMIT 50;"
    return str(get_db().run(query))

@tool(args_schema=JobRoleSearchInput)
def get_courses_by_job_role(job_role: str) -> str:
    """Find courses tailored for a specific career or job role."""
    query = f"SELECT course_name, platform, branch_name, course_link FROM branch_courses WHERE job_role ILIKE '%{job_role}%' LIMIT 50;"
    return str(get_db().run(query))

@tool(args_schema=CourseSearchInput)
def search_courses_by_keyword(keyword: str) -> str:
    """Search for courses by keyword."""
    query = f"SELECT course_name, platform, branch_name, job_role, course_link FROM branch_courses WHERE course_name ILIKE '%{keyword}%' LIMIT 50;"
    return str(get_db().run(query))

@tool
def list_all_course_platforms() -> str:
    """List all available course platforms (Coursera, Udemy, etc.)."""
    query = "SELECT DISTINCT platform FROM branch_courses WHERE platform IS NOT NULL;"
    return str(get_db().run(query))

@tool(args_schema=StateSearchInput)
def find_colleges_by_state(state: str) -> str:
    """Search for engineering colleges in a specific state."""
    query = f"SELECT name, website FROM engineering_colleges WHERE state ILIKE '%{state}%' LIMIT 100;"
    return str(get_db().run(query))

@tool(args_schema=CollegeSearchInput)
def get_college_info(college_name: str) -> str:
    """Get AISHE code, state, and website for a specific college."""
    query = f"SELECT * FROM engineering_colleges WHERE name ILIKE '%{college_name}%';"
    return str(get_db().run(query))

@tool(args_schema=RankingSearchInput)
def get_engineering_rankings(state: Optional[str] = None, limit: int = 50) -> str:
    """Get the top NIRF-ranked engineering institutions. Optionally filter by state."""
    try:
        state_clause = f"WHERE state ILIKE '%{state}%'" if state else ""
        query = f"SELECT rank, name, city, state FROM nirf_rankings_engineering {state_clause} ORDER BY CAST(rank AS INTEGER) ASC LIMIT {limit};"
        return str(get_db().run(query))
    except Exception as e:
        return f"Error: {e}"

@tool(args_schema=RankingSearchInput)
def get_university_rankings(state: Optional[str] = None, limit: int = 50) -> str:
    """Get the top NIRF-ranked universities. Optionally filter by state or ranking number."""
    try:
        state_clause = f"WHERE state ILIKE '%{state}%'" if state else ""
        query = f"SELECT ranking, name, city, state FROM nirf_rankings_university {state_clause} ORDER BY CAST(ranking AS INTEGER) ASC LIMIT {limit};"
        return str(get_db().run(query))
    except Exception as e:
        return f"Error: {e}"

@tool(args_schema=UniversitySearchInput)
def get_university_info(name: str) -> str:
    """Get detailed info about a university including AISHE code, website, and location."""
    try:
        query = f"SELECT * FROM all_universities WHERE name ILIKE '%{name}%' LIMIT 10;"
        return str(get_db().run(query))
    except Exception as e:
        return f"Error: {e}"

@tool(args_schema=CategorySearchInput)
def get_branches_by_category(category_name: str) -> str:
    """Find all engineering branches belonging to a specific category (e.g., 'Information Technology')."""
    try:
        query = f"SELECT branch_name FROM aicte_branches WHERE category ILIKE '%{category_name}%' LIMIT 100;"
        return str(get_db().run(query))
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
        # Salaries are stored as text (e.g., '10-12'), so we cast/parse. This uses a standard SQL trick.
        query = f"SELECT branch_name, freshers_salary_lpa, experienced_salary_lpa FROM branch_salaries WHERE CAST(SUBSTRING(freshers_salary_lpa FROM '^[0-9.]+') AS FLOAT) >= {min_salary_lpa} ORDER BY CAST(SUBSTRING(freshers_salary_lpa FROM '^[0-9.]+') AS FLOAT) DESC LIMIT 20;"
        return str(get_db().run(query))
    except Exception as e:
        return f"Error: {e}"

@tool(args_schema=BranchSearchInput)
def get_international_salary_insights(branch_name: str) -> str:
    """Get international salary data (USD) and source links for a specific branch."""
    try:
        query = f"SELECT branch_name, international_salary_usd, international_salary_source FROM branch_salaries WHERE branch_name ILIKE '%{branch_name}%' AND international_salary_usd IS NOT NULL;"
        return str(get_db().run(query))
    except Exception as e:
        return f"Error: {e}"

@tool(args_schema=ComparisonInput)
def compare_two_branches(branch_a: str, branch_b: str) -> str:
    """Compare two engineering branches side-by-side (Salaries, Domains, Jobs)."""
    try:
        query = f"""
        SELECT s.branch_name, s.freshers_salary_lpa, d.technical_domains, r.project_ideas
        FROM branch_salaries s
        LEFT JOIN branch_technical_domains d ON s.branch_name = d.branch_name
        LEFT JOIN branch_roadmaps r ON s.branch_name = r.branch_name
        WHERE s.branch_name ILIKE '%{branch_a}%' OR s.branch_name ILIKE '%{branch_b}%';
        """
        return str(get_db().run(query))
    except Exception as e:
        return f"Error: {e}"

@tool(args_schema=CitySearchInput)
def find_colleges_by_city(city: str, state: Optional[str] = None) -> str:
    """Search for NIRF-ranked colleges and universities in a specific city."""
    try:
        state_clause = f"AND state ILIKE '%{state}%'" if state else ""
        query = f"SELECT rank, name, city, state FROM nirf_rankings_engineering WHERE city ILIKE '%{city}%' {state_clause} LIMIT 50;"
        res_eng = get_db().run(query)
        query_uni = f"SELECT ranking, name, city, state FROM nirf_rankings_university WHERE city ILIKE '%{city}%' {state_clause} LIMIT 50;"
        res_uni = get_db().run(query_uni)
        return f"Engineering: {res_eng}\n\nUniversities: {res_uni}"
    except Exception as e:
        return f"Error: {e}"

@tool(args_schema=DistrictSearchInput)
def get_universities_by_district(district: str, state: str) -> str:
    """List all universities in a specific district and state."""
    try:
        query = f"SELECT name, website, location FROM all_universities WHERE district ILIKE '%{district}%' AND state ILIKE '%{state}%' LIMIT 50;"
        return str(get_db().run(query))
    except Exception as e:
        return f"Error: {e}"

@tool(args_schema=CourseSearchInput)
def get_branches_by_technical_domain(keyword: str) -> str:
    """Search for engineering branches based on a technical domain keyword (e.g., 'VLSI', 'AI', 'Power')."""
    try:
        query = f"SELECT branch_name, technical_domains FROM branch_technical_domains WHERE technical_domains ILIKE '%{keyword}%' LIMIT 50;"
        return str(get_db().run(query))
    except Exception as e:
        return f"Error: {e}"

@tool(args_schema=CourseSearchInput)
def search_project_ideas(keyword: str) -> str:
    """Find project ideas and preparation tips across all branches using a keyword."""
    try:
        query = f"SELECT branch_name, project_ideas FROM branch_roadmaps WHERE project_ideas ILIKE '%{keyword}%' LIMIT 50;"
        return str(db.run(query))
    except Exception as e:
        return f"Error: {e}"

@tool(args_schema=CollegeComparisonInput)
def compare_colleges(college_a: str, college_b: str) -> str:
    """Compare two colleges or universities (Rankings, State, Website)."""
    try:
        query = f"""
        SELECT 'Engineering' as type, name, city, state, rank as info
        FROM nirf_rankings_engineering 
        WHERE name ILIKE '%{college_a}%' OR name ILIKE '%{college_b}%'
        UNION ALL
        SELECT 'University' as type, name, city, state, ranking as info
        FROM nirf_rankings_university
        WHERE name ILIKE '%{college_a}%' OR name ILIKE '%{college_b}%';
        """
        return str(db.run(query))
    except Exception as e:
        return f"Error: {e}"

@tool(args_schema=YearRangeInput)
def get_universities_by_establishment_year(start_year: int, end_year: int) -> str:
    """Find universities established within a specific year range."""
    try:
        # year_of_establishment is text, so we cast it for comparison.
        query = f"""
        SELECT name, state, year_of_establishment, website 
        FROM all_universities 
        WHERE CAST(NULLIF(year_of_establishment, '') AS INTEGER) BETWEEN {start_year} AND {end_year}
        ORDER BY year_of_establishment ASC LIMIT 50;
        """
        return str(db.run(query))
    except Exception as e:
        return f"Error: {e}"

@tool(args_schema=QueryInput)
def execute_advanced_sql(query: str) -> str:
    """Execute raw SQL SELECT query. ALWAYS use LIMIT."""
    if not query.strip().lower().startswith("select"):
        return "Error: Only SELECT queries are allowed."
    return str(db.run(query))

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
