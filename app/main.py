from fastapi import FastAPI
from app.controllers import agent_controller
import uvicorn
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="AI Engineering Career Agent API")

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
