import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()
URL = os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL")
KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
if not KEY:
    print("❌ ERROR: SUPABASE_SERVICE_ROLE_KEY is required for deletion!")
    exit(1)

supabase: Client = create_client(URL, KEY)

TABLES = [
    'branch_courses',
    'branch_job_roles',
    'branch_roadmaps',
    'branch_salaries',
    'branch_technical_domains',
    'aicte_branches',
    'branch_categories',
    'all_universities',
    'engineering_colleges',
    'nirf_rankings_engineering',
    'nirf_rankings_university'
]

def cleanup():
    print("🧹 Starting Database Cleanup...")
    for table in TABLES:
        try:
            # delete().neq('id', -1) is a common way to 'delete all' in Supabase without a filter
            # Better: delete().gt('created_at', '1970-01-01') or similar if no pk is known
            # Actually, just delete().neq() or filtering by a column that always exists is safest.
            # Using filtering on 'created_at' as it's default in Supabase.
            print(f"Deleting all records from {table}...")
            supabase.table(table).delete().neq('created_at', '1900-01-01').execute()
        except Exception as e:
            print(f"Failed to clear {table}: {e}")
    print("✅ Cleanup complete.")

if __name__ == "__main__":
    cleanup()
