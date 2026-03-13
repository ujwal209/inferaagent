import pandas as pd
import numpy as np
import os

def clean(val):
    if pd.isna(val) or str(val).lower() in ['nan', 'none', '']:
        return "NULL"
    # Escape single quotes for SQL
    return "'" + str(val).replace("'", "''").strip() + "'"

def generate():
    sql_file = "full_data_dump.sql"
    print(f"📂 Generating {sql_file}...")
    
    with open(sql_file, "w", encoding="utf-8") as f:
        f.write("-- RECREATE SCHEMA WITHOUT CONSTRAINTS\n")
        f.write("DROP TABLE IF EXISTS branch_courses CASCADE;\n")
        f.write("DROP TABLE IF EXISTS branch_job_roles CASCADE;\n")
        f.write("DROP TABLE IF EXISTS branch_roadmaps CASCADE;\n")
        f.write("DROP TABLE IF EXISTS branch_salaries CASCADE;\n")
        f.write("DROP TABLE IF EXISTS branch_technical_domains CASCADE;\n")
        f.write("DROP TABLE IF EXISTS aicte_branches CASCADE;\n")
        f.write("DROP TABLE IF EXISTS branch_categories CASCADE;\n")
        f.write("DROP TABLE IF EXISTS all_universities CASCADE;\n")
        f.write("DROP TABLE IF EXISTS engineering_colleges CASCADE;\n")
        f.write("DROP TABLE IF EXISTS nirf_rankings_engineering CASCADE;\n")
        f.write("DROP TABLE IF EXISTS nirf_rankings_university CASCADE;\n\n")

        f.write("CREATE TABLE branch_categories (id BIGSERIAL PRIMARY KEY, category_name TEXT);\n")
        f.write("CREATE TABLE aicte_branches (id BIGSERIAL PRIMARY KEY, branch_name TEXT, category TEXT);\n")
        f.write("CREATE TABLE all_universities (id BIGSERIAL PRIMARY KEY, name TEXT, aishe_code TEXT, state TEXT, district TEXT, website TEXT, year_of_establishment TEXT, location TEXT);\n")
        f.write("CREATE TABLE engineering_colleges (id BIGSERIAL PRIMARY KEY, name TEXT, aishe_code TEXT, state TEXT, website TEXT);\n")
        f.write("CREATE TABLE branch_roadmaps (id BIGSERIAL PRIMARY KEY, branch_name TEXT, roadmap_url TEXT, roadmap_image TEXT, project_ideas TEXT, internship_preparation TEXT);\n")
        f.write("CREATE TABLE branch_salaries (id BIGSERIAL PRIMARY KEY, branch_name TEXT, freshers_salary_lpa TEXT, experienced_salary_lpa TEXT, salary_source_link TEXT, international_salary_usd TEXT, international_salary_source TEXT);\n")
        f.write("CREATE TABLE branch_technical_domains (id BIGSERIAL PRIMARY KEY, branch_name TEXT, technical_domains TEXT);\n")
        f.write("CREATE TABLE branch_job_roles (id BIGSERIAL PRIMARY KEY, branch_name TEXT, job_role TEXT);\n")
        f.write("CREATE TABLE branch_courses (id BIGSERIAL PRIMARY KEY, branch_name TEXT, job_role TEXT, course_name TEXT, platform TEXT, course_link TEXT);\n")
        f.write("CREATE TABLE nirf_rankings_engineering (id BIGSERIAL PRIMARY KEY, name TEXT, city TEXT, state TEXT, rank TEXT);\n")
        f.write("CREATE TABLE nirf_rankings_university (id BIGSERIAL PRIMARY KEY, name TEXT, city TEXT, state TEXT, ranking TEXT);\n\n")
        
        f.write("BEGIN;\n\n")

        # 1. Categories & Master Branches
        df = pd.read_csv("AICTE Recognized UG Engineering Branches Categorized.csv")
        df.columns = df.columns.str.strip()
        cats = df['Category'].dropna().unique()
        for cat in cats:
            f.write(f"INSERT INTO branch_categories (category_name) VALUES ({clean(cat)});\n")
        for _, row in df.iterrows():
            f.write(f"INSERT INTO aicte_branches (branch_name, category) VALUES ({clean(row['Branch Name'])}, {clean(row['Category'])});\n")

        # 3. Universities
        df_uni = pd.read_csv("University-ALL UNIVERSITIES.csv", skiprows=2)
        df_uni.columns = df_uni.columns.str.strip()
        for _, row in df_uni.iterrows():
            f.write(f"INSERT INTO all_universities (name, aishe_code, state, district, website, year_of_establishment, location) VALUES ({clean(row['Name'])}, {clean(row['Aishe Code'])}, {clean(row['State'])}, {clean(row['District'])}, {clean(row['Website'])}, {clean(row['Year Of Establishment'])}, {clean(row['Location'])});\n")

        # 4. Colleges
        df_col = pd.read_csv("Engineering colleges cleaned.csv")
        df_col.columns = df_col.columns.str.strip()
        for _, row in df_col.iterrows():
            f.write(f"INSERT INTO engineering_colleges (name, aishe_code, state, website) VALUES ({clean(row['Name'])}, {clean(row['Aishe Code'])}, {clean(row['State'])}, {clean(row['Website'])});\n")

        # 5. Roadmaps
        df_road = pd.read_csv("Road map.csv")
        df_road.columns = df_road.columns.str.strip()
        for _, row in df_road.iterrows():
            f.write(f"INSERT INTO branch_roadmaps (branch_name, project_ideas, internship_preparation) VALUES ({clean(row['AICTE-Recognized BE/B.Tech Branches'])}, {clean(row['Project Ideas'])}, {clean(row['Internship Preparation Guidance'])});\n")

        # 6. Salaries
        df_sal = pd.read_csv("Salary.csv")
        df_sal.columns = df_sal.columns.str.strip()
        for _, row in df_sal.iterrows():
            f.write(f"INSERT INTO branch_salaries (branch_name, freshers_salary_lpa, experienced_salary_lpa, salary_source_link, international_salary_usd, international_salary_source) VALUES ({clean(row['AICTE-Recognized BE/B.Tech Branches'])}, {clean(row.get('Freshers Salary (₹ LPA)'))}, {clean(row.get('Experienced Salary (₹ LPA)'))}, {clean(row.get('Salary Source (Useful Link)'))}, {clean(row.get('International Salary (USD per Year Approx.)'))}, {clean(row.get('International Salary Source (Reference Link)'))});\n")

        # 7. Technical Domains
        df_tech = pd.read_csv("PROJECT_5_All_Branches_Filled.csv")
        df_tech.columns = df_tech.columns.str.strip()
        for _, row in df_tech.iterrows():
            f.write(f"INSERT INTO branch_technical_domains (branch_name, technical_domains) VALUES ({clean(row['AICTE-Recognized BE/B.Tech Branches'])}, {clean(row['Important Technical Domains'])});\n")

        # 8. Job Roles
        df_roles = pd.read_csv("Branch With Job Roles.csv")
        df_roles.columns = df_roles.columns.str.strip()
        for _, row in df_roles.iterrows():
            f.write(f"INSERT INTO branch_job_roles (branch_name, job_role) VALUES ({clean(row['Branch_Name'])}, {clean(row['Job Role'])});\n")

        # 9. Courses
        df_crs = pd.read_csv("All_Engineering_Branches_Aerospace_Spaced3.csv")
        df_crs.columns = df_crs.columns.str.strip()
        for _, row in df_crs.iterrows():
            f.write(f"INSERT INTO branch_courses (branch_name, job_role, course_name, platform, course_link) VALUES ({clean(row['Branch'])}, {clean(row['Job Role'])}, {clean(row['Course Name'])}, {clean(row['Platform'])}, {clean(row['Course Link'])});\n")

        # 10. Rankings
        df_rnk_e = pd.read_csv("NIRF_2025_Engineering.csv")
        df_rnk_e.columns = df_rnk_e.columns.str.strip()
        for _, row in df_rnk_e.iterrows():
            f.write(f"INSERT INTO nirf_rankings_engineering (name, city, state, rank) VALUES ({clean(row['Name'])}, {clean(row['City'])}, {clean(row['State'])}, {clean(row['Rank'])});\n")

        df_rnk_u = pd.read_csv("NIRF_2025_University.csv")
        df_rnk_u.columns = df_rnk_u.columns.str.strip()
        for _, row in df_rnk_u.iterrows():
            f.write(f"INSERT INTO nirf_rankings_university (name, city, state, ranking) VALUES ({clean(row['Name'])}, {clean(row['City'])}, {clean(row['State'])}, {clean(row['Ranking'])});\n")

        f.write("\nCOMMIT;")
    print("✅ SQL Dump Generated Successfully!")

    print("✅ SQL Dump Generated Successfully!")

if __name__ == "__main__":
    generate()
