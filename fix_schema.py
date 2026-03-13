import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()
URL = os.getenv("SUPABASE_URL")
KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
supabase: Client = create_client(URL, KEY)

sql = """
-- Drop the restrictive primary key
ALTER TABLE branch_courses DROP CONSTRAINT IF EXISTS branch_courses_pkey;
-- Add a unique ID for every single row
ALTER TABLE branch_courses ADD COLUMN IF NOT EXISTS row_id BIGSERIAL PRIMARY KEY;
-- Do the same for Job Roles just in case
ALTER TABLE branch_job_roles DROP CONSTRAINT IF EXISTS branch_job_roles_pkey;
ALTER TABLE branch_job_roles ADD COLUMN IF NOT EXISTS row_id BIGSERIAL PRIMARY KEY;
"""

try:
    print("🚀 Attempting to remove DB constraints to allow 'as-is' insertion...")
    res = supabase.rpc('execute_advanced_sql', {'sql_query': sql}).execute()
    print("✅ Success:", res.data)
except Exception as e:
    print("❌ Failed:", e)
