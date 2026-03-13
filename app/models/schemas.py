from pydantic import BaseModel, Field
from typing import Optional, List

class TableNameInput(BaseModel):
    table_name: str = Field(description="The name of the table to inspect")

class QueryInput(BaseModel):
    query: str = Field(description="The SQL SELECT query to execute. ALWAYS use LIMIT.")

class StateSearchInput(BaseModel):
    state: str = Field(description="The name of the state (e.g., 'Maharashtra', 'Tamil Nadu')")

class BranchSearchInput(BaseModel):
    branch_name: str = Field(description="The engineering branch name")

class CollegeSearchInput(BaseModel):
    college_name: str = Field(description="The name or partial name of the college")

class RankingSearchInput(BaseModel):
    state: Optional[str] = Field(None, description="Filter rankings by state")
    limit: int = Field(50, description="Number of top results to return")

class CourseSearchInput(BaseModel):
    keyword: str = Field(description="Keyword to search in course names")

class PlatformSearchInput(BaseModel):
    platform: str = Field(description="Name of the platform")

class JobRoleSearchInput(BaseModel):
    job_role: str = Field(description="The specific job role")

class ComparisonInput(BaseModel):
    branch_a: str = Field(description="First engineering branch name")
    branch_b: str = Field(description="Second engineering branch name")

class SalaryFilterInput(BaseModel):
    min_salary_lpa: float = Field(default=5.0, description="Minimum freshers salary in LPA")

class CitySearchInput(BaseModel):
    city: str = Field(description="The name of the city")
    state: Optional[str] = Field(None, description="Optional state to narrow down city search")

class DistrictSearchInput(BaseModel):
    district: str = Field(description="The name of the district")
    state: str = Field(description="The name of the state")

class UniversitySearchInput(BaseModel):
    name: str = Field(description="The name or partial name of the university")

class CollegeComparisonInput(BaseModel):
    college_a: str = Field(description="First college/university name")
    college_b: str = Field(description="Second college/university name")

class YearRangeInput(BaseModel):
    start_year: int = Field(description="Starting year of establishment")
    end_year: int = Field(description="Ending year of establishment")

class CategorySearchInput(BaseModel):
    category_name: str = Field(description="The name of the branch category (e.g., 'Core Engineering', 'IT & Computer')")

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[dict]] = []
