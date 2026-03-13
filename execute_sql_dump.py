import psycopg2
import os

# Your direct PostgreSQL connection string
CONN_STRING = "postgresql://postgres:fpCI2k8tWBtapfx2@db.hneigfjjhopwlgloqfmn.supabase.co:5432/postgres"

def run_sql():
    sql_file = "full_data_dump.sql"
    if not os.path.exists(sql_file):
        print(f"❌ Error: {sql_file} not found!")
        return

    print(f"🚀 Connecting to database and executing {sql_file}...")
    try:
        # Connect to the database
        conn = psycopg2.connect(CONN_STRING)
        conn.autocommit = True  # We have BEGIN/COMMIT in the script
        cur = conn.cursor()

        # Read the SQL file
        with open(sql_file, "r", encoding="utf-8") as f:
            sql_script = f.read()

        # Execute the script
        cur.execute(sql_script)
        
        print("✅ SQL script executed successfully!")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")

if __name__ == "__main__":
    run_sql()
