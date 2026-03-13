import os
import pandas as pd
import time
from supabase import create_client, Client
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# Config
URL = os.getenv("SUPABASE_URL")
KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

supabase: Client = create_client(URL, KEY)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GOOGLE_API_KEY)

CSV_FILES = [
    "AICTE Recognized UG Engineering Branches Categorized.csv",
    "Branch With Job Roles.csv",
    "Road map.csv",
    "Salary.csv",
    "PROJECT_5_All_Branches_Filled.csv",
    "All_Engineering_Branches_Aerospace_Spaced3.csv",
    "Engineering colleges cleaned.csv",
    "NIRF_2025_Engineering.csv",
    "NIRF_2025_University.csv",
    "University-ALL UNIVERSITIES.csv"
]

from tenacity import retry, wait_exponential, stop_after_attempt

def vectorize_csv(file_path):
    if not os.path.exists(file_path):
        print(f"⚠️ Warning: {file_path} not found.")
        return []
        
    print(f"📄 Processing {file_path}...")
    df = pd.read_csv(file_path)
    
    rows = []
    for _, row in df.iterrows():
        content_parts = []
        for col in df.columns:
            val = str(row[col]).strip()
            if val and val.lower() != "nan" and val != "None":
                content_parts.append(f"{col}: {val}")
        
        content = ". ".join(content_parts)
        # Try to find a branch name in various possible columns
        branch_cols = ['Branch', 'Branch_Name', 'Branch Name', 'AICTE-Recognized BE/B.Tech Branches']
        branch_val = "Unknown"
        for bc in branch_cols:
            if bc in row:
                branch_val = str(row[bc])
                break

        metadata = {
            "source": file_path,
            "branch": branch_val
        }
        rows.append({"content": content, "metadata": metadata})
    
    return rows

@retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(5))
def embed_with_retry(texts):
    return embeddings.embed_documents(texts)

def push_to_supabase(data):
    print(f"🚀 Vectorizing and pushing {len(data)} records in batches...")
    
    batch_size = 20 # Smaller batch to avoid token limits
    for i in tqdm(range(0, len(data), batch_size)):
        batch = data[i:i+batch_size]
        texts = [doc['content'] for doc in batch]
        
        try:
            # Generate embeddings with backoff retry
            vector_embeddings = embed_with_retry(texts)
            
            payload = []
            for j, doc in enumerate(batch):
                payload.append({
                    "content": doc['content'],
                    "metadata": doc['metadata'],
                    "embedding": vector_embeddings[j]
                })
            
            supabase.table("knowledge_base").insert(payload).execute()
            
            # Rate limit cooling period for free tier (100 RPM limit encountered)
            time.sleep(2) 
            
        except Exception as e:
            print(f"❌ Permanent Error in batch {i}: {e}")

def run():
    all_data = []
    for f in CSV_FILES:
        all_data.extend(vectorize_csv(f))
    
    if all_data:
        push_to_supabase(all_data)
        print("✅ Vectorization complete!")

if __name__ == "__main__":
    run()
