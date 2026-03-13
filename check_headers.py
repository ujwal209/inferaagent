import pandas as pd
files = [
    'AICTE Recognized UG Engineering Branches Categorized.csv',
    'University-ALL UNIVERSITIES.csv',
    'Engineering colleges cleaned.csv',
    'Road map.csv',
    'Salary.csv',
    'PROJECT_5_All_Branches_Filled.csv',
    'Branch With Job Roles.csv',
    'All_Engineering_Branches_Aerospace_Spaced3.csv',
    'NIRF_2025_Engineering.csv',
    'NIRF_2025_University.csv'
]
for f in files:
    try:
        if 'University-ALL' in f:
            df = pd.read_csv(f, skiprows=2, nrows=0)
        else:
            df = pd.read_csv(f, nrows=0)
        print(f"{f}: {list(df.columns)}")
    except Exception as e:
        print(f"Error {f}: {e}")
