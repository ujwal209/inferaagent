import logging
import PyPDF2
import json
from io import BytesIO
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Import the pre-compiled agents from the service layer (Roadmap removed)
from app.services.agent_service import (
    general_agent, 
    resume_agent, 
    study_agent  # Import the new Study Agent
)

router = APIRouter()

# --- SCHEMAS ---
class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []
    webSearch: Optional[bool] = False
    deepThink: Optional[bool] = False

# --- HELPER ---
def format_history(history: list[dict]):
    """Converts raw dictionary history into LangChain message objects."""
    messages = []
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    return messages


# --- ENDPOINTS ---

@router.post("/chat")
async def chat_with_agent(request: ChatRequest):
    """Original General Chat Endpoint"""
    try:
        messages = format_history(request.history)
        messages.append(HumanMessage(content=request.message))
        
        final_state = general_agent.invoke({"messages": messages}, config={"recursion_limit": 20})
        return {"response": final_state["messages"][-1].content}
        
    except Exception as e:
        logging.error(f"General chat error: {e}")
        return {
            "response": "I'm sorry, I encountered an internal error while processing that deeply. Could you try rephrasing?", 
            "error": str(e)
        }


@router.post("/study")
async def study_with_agent(request: ChatRequest):
    """Specialized Study Buddy Endpoint (Math, CS, Linear Algebra)"""
    try:
        messages = format_history(request.history)
        
        dynamic_prompts = []
        if request.webSearch:
            dynamic_prompts.append("USER ENABLED WEB SEARCH: You MUST proactively use your `web_search` tool over multiple iterations to fetch up-to-date facts if necessary.")
        if request.deepThink:
            dynamic_prompts.append("USER ENABLED DEEP THINK: You MUST deeply reason and 'deep think' through solutions step-by-step prior to returning your final output. Be extremely detailed and thoughtful in your reasoning.")
            
        if dynamic_prompts:
            messages.append(SystemMessage(content="\n\n".join(dynamic_prompts)))
            
        messages.append(HumanMessage(content=request.message))
        
        final_state = study_agent.invoke({"messages": messages}, config={"recursion_limit": 20})
        return {"response": final_state["messages"][-1].content}
        
    except Exception as e:
        logging.error(f"Study agent error: {e}")
        return {
            "response": "I'm sorry, I encountered an error trying to solve that. Could you provide a bit more detail?",
            "error": str(e)
        }


@router.post("/resume-analyze")
async def analyze_resume(
    target_role: Optional[str] = Form(None), 
    message: Optional[str] = Form(None),
    history: Optional[str] = Form(None),  # <--- NEW: Accepts stringified JSON history
    file: Optional[UploadFile] = File(None)
):
    """Extracts text from PDF or handles follow-up chat messages with Memory"""
    try:
        prompt = ""

        # SCENARIO 1: Initial Upload (Contains PDF File)
        if file:
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(status_code=400, detail="Only PDF files are supported.")
            
            pdf_bytes = await file.read()
            pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
            extracted_text = ""
            for page in pdf_reader.pages:
                extracted_text += page.extract_text() + "\n"
                
            if not extracted_text.strip():
                raise HTTPException(status_code=400, detail="Could not extract readable text from the PDF.")

            role_text = target_role or "General Optimization"
            prompt = (
                f"Target Role: {role_text}\n\n"
                f"Here is the extracted resume text:\n{extracted_text}\n\n"
                f"Please analyze this resume deeply against the target role. Provide rewritten bullet points and thorough explanations."
            )
        
        # SCENARIO 2: Follow-up Chat (No File, Contains Message)
        elif message:
            prompt = message
            
        else:
            raise HTTPException(status_code=400, detail="Must provide either a file upload or a chat message.")

        # --- MEMORY INJECTION ---
        parsed_history = []
        if history:
            try:
                parsed_history = json.loads(history)
            except Exception as e:
                logging.error(f"History parse error: {e}")

        messages = format_history(parsed_history)
        messages.append(HumanMessage(content=prompt))
        
        # Invoke agent with full conversational context
        final_state = resume_agent.invoke({"messages": messages}) 
        return {"response": final_state["messages"][-1].content}
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Resume analysis error: {e}")
        raise HTTPException(status_code=500, detail="Failed to process resume.")