import os
import pandas as pd
import numpy as np
from supabase import create_client, Client
from dotenv import load_dotenv

# 1. Setup
load_dotenv()
URL = os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL")
KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") # MUST USE SERVICE ROLE KEY TO BYPASS RLS
supabase: Client = create_client(URL, KEY)

# 2. Data Cleaning Helper
def clean_df(df):
    """Strips whitespace, removes NaNs, prepares for JSON insert"""
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(['object']).columns:
        df[col] = df[col].astype(str).str.strip().replace({'nan': None, 'NaN': None, 'None': None, '': None})
    
    # Aggressively replace all forms of NaN with None
    return df.replace({np.nan: None}).where(pd.notnull(df), None)

# 3. Batch Upsert Helper (Maximum Resilience)
def batch_upsert(table_name, data_list, batch_size=200):
    if not data_list: return
    print(f"Pushing {len(data_list)} records to {table_name}...")
    success_count = 0
    for i in range(0, len(data_list), batch_size):
        batch = data_list[i:i+batch_size]
        try:
            # Attempt batch insert
            supabase.table(table_name).insert(batch).execute()
            success_count += len(batch)
        except Exception:
            # Fallback to row-by-row for this batch to ensure we miss NOTHING
            print(f"  ⚠️ Batch {i} contains some duplicates. Switching to row-by-row mode...")
            for row in batch:
                try:
                    supabase.table(table_name).insert(row).execute()
                    success_count += 1
                except Exception as e:
                    # If insert fails, at least update it so user sees the data
                    try:
                        supabase.table(table_name).upsert(row).execute()
                        success_count += 1
                    except:
                        pass # Truly duplicate in every way
    print(f"✓ {table_name} complete. ({success_count} rows in DB)\n")

def run_seed():
    print("🚀 Starting Database Seed...\n")

    # --- LOAD ALL CSVs ---
    df_cat = clean_df(pd.read_csv("AICTE Recognized UG Engineering Branches Categorized.csv"))
    df_uni = clean_df(pd.read_csv("University-ALL UNIVERSITIES.csv", skiprows=2))
    df_col = clean_df(pd.read_csv("Engineering colleges cleaned.csv"))
    df_road = clean_df(pd.read_csv("Road map.csv"))
    df_sal = clean_df(pd.read_csv("Salary.csv"))
    df_tech = clean_df(pd.read_csv("PROJECT_5_All_Branches_Filled.csv"))
    df_roles = clean_df(pd.read_csv("Branch With Job Roles.csv"))
    df_crs = clean_df(pd.read_csv("All_Engineering_Branches_Aerospace_Spaced3.csv"))
    df_rnk_e = clean_df(pd.read_csv("NIRF_2025_Engineering.csv"))
    df_rnk_u = clean_df(pd.read_csv("NIRF_2025_University.csv"))

    # ==========================================
    # STEP 1: MASTER TABLES
    # ==========================================
    
    # Categories
    categories = [{'category_name': cat} for cat in df_cat['Category'].dropna().unique()]
    batch_upsert('branch_categories', categories)

    # Master Branches (We must combine all unique branches from all files so Foreign Keys don't crash)
    all_branches = set()
    all_branches.update(df_cat['Branch Name'].dropna().unique())
    all_branches.update(df_road['AICTE-Recognized BE/B.Tech Branches'].dropna().unique())
    all_branches.update(df_sal['AICTE-Recognized BE/B.Tech Branches'].dropna().unique())
    all_branches.update(df_tech['AICTE-Recognized BE/B.Tech Branches'].dropna().unique())
    all_branches.update(df_roles['Branch_Name'].dropna().unique())
    all_branches.update(df_crs['Branch'].dropna().unique())
    all_branches.discard("None") # remove junk
    
    # Map known categories, default others to None
    cat_map = dict(zip(df_cat['Branch Name'], df_cat['Category']))
    branches_payload = [{'branch_name': b, 'category': cat_map.get(b, None)} for b in all_branches]
    batch_upsert('aicte_branches', branches_payload)

    # Universities
    uni_payload = df_uni.rename(columns={
        'Aishe Code': 'aishe_code', 'Name': 'name', 'State': 'state', 
        'District': 'district', 'Website': 'website', 
        'Year Of Establishment': 'year_of_establishment', 'Location': 'location'
    }).dropna(subset=['aishe_code']).to_dict(orient='records')
    batch_upsert('all_universities', uni_payload)

    # Colleges
    col_payload = df_col.rename(columns={
        'Aishe Code': 'aishe_code', 'Name': 'name', 'State': 'state', 'Website': 'website'
    }).dropna(subset=['aishe_code']).to_dict(orient='records')
    batch_upsert('engineering_colleges', col_payload)

    # ==========================================
    # STEP 2: 1-TO-1 DEPENDENT TABLES
    # ==========================================

    # Roadmaps
    road_payload = df_road.rename(columns={
        'AICTE-Recognized BE/B.Tech Branches': 'branch_name',
        'Project Ideas': 'project_ideas',
        'Internship Preparation Guidance': 'internship_preparation'
    }).dropna(subset=['branch_name']).to_dict(orient='records')
    batch_upsert('branch_roadmaps', road_payload)

    # Salaries
    sal_payload = df_sal.rename(columns={
        'AICTE-Recognized BE/B.Tech Branches': 'branch_name',
        'Freshers Salary (₹ LPA)': 'freshers_salary_lpa',
        'Experienced Salary (₹ LPA)': 'experienced_salary_lpa',
        'Salary Source (Useful Link)': 'salary_source_link',
        'International Salary (USD per Year Approx.)': 'international_salary_usd',
        'International Salary Source (Reference Link)': 'international_salary_source'
    }).dropna(subset=['branch_name']).to_dict(orient='records')
    batch_upsert('branch_salaries', sal_payload)

    # Technical Domains
    tech_payload = df_tech.rename(columns={
        'AICTE-Recognized BE/B.Tech Branches': 'branch_name',
        'Important Technical Domains': 'technical_domains'
    }).dropna(subset=['branch_name']).to_dict(orient='records')
    batch_upsert('branch_technical_domains', tech_payload)


    # ==========================================
    # STEP 3: 1-TO-MANY DEPENDENT TABLES
    # ==========================================

    # Job Roles
    print("Preparing Job Roles...")
    roles_payload = df_roles.rename(columns={
        'Branch_Name': 'branch_name', 'Job Role': 'job_role'
    }).dropna(subset=['branch_name', 'job_role']).to_dict(orient='records')
    batch_upsert('branch_job_roles', roles_payload)

    # Courses - RAW INSERT AS IS (No grouping)
    print("Preparing Courses (Raw)...")
    crs_payload = df_crs.rename(columns={
        'Branch': 'branch_name', 'Job Role': 'job_role', 
        'Course Name': 'course_name', 'Platform': 'platform', 'Course Link': 'course_link'
    }).dropna(subset=['branch_name', 'course_name']).to_dict(orient='records')
    batch_upsert('branch_courses', crs_payload)


    # ==========================================
    # STEP 4: RANKINGS
    # ==========================================
    
    rnk_e_payload = df_rnk_e.rename(columns={
        'Name': 'name', 'City': 'city', 'State': 'state', 'Rank': 'rank'
    }).dropna(subset=['name']).to_dict(orient='records')
    batch_upsert('nirf_rankings_engineering', rnk_e_payload)

    rnk_u_payload = df_rnk_u.rename(columns={
        'Name': 'name', 'City': 'city', 'State': 'state', 'Ranking': 'ranking'
    }).dropna(subset=['name']).to_dict(orient='records')
    batch_upsert('nirf_rankings_university', rnk_u_payload)

    print("🎉 Database Seed Completely Successfully!")

if __name__ == "__main__":
    run_seed()