from fastapi import FastAPI
from app.controllers import agent_controller
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="AI Engineering Career Agent API")

# 🚀 STEP 1: ENABLE CORS (Direct-to-Backend Refactor)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the router
app.include_router(agent_controller.router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"status": "online", "message": "Engineering Career API is running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": "ok"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
