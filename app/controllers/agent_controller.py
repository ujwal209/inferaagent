import logging
import PyPDF2
import json
import re
from io import BytesIO
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from app.services.rag_service import process_and_store_document, retrieve_relevant_context

from app.services.agent_service import (
    general_agent, 
    resume_agent, 
    study_agent  
)

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []
    webSearch: Optional[bool] = False
    deepThink: Optional[bool] = False
    sessionId: Optional[str] = None
    images: list[str] = [] 

def construct_user_message(text: str, images: list[str] = [], context: str = ""):
    """Helper to safely build Multi-Modal Human messages for Vision endpoints."""
    supported_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.gif')
    valid_images = [url for url in images if url.lower().split('?')[0].endswith(supported_extensions)]
    
    # 🚀 CRITICAL FIX: Extract BOTH Markdown images and [Attachment: URL] tags!
    md_urls = re.findall(r'!\[.*?\]\((https?://[^\)\s]+)\)', text)
    att_urls = re.findall(r'\[Attachment:\s*(https?://[^\]\s]+)\]', text, re.IGNORECASE)
    
    all_extracted = md_urls + att_urls
    for url in all_extracted:
        if url.lower().split('?')[0].endswith(supported_extensions):
            valid_images.append(url.strip())
            
    valid_images = list(set(valid_images)) # Deduplicate
    
    # Clean the text so the LLM doesn't try to "read" the raw URL strings
    clean_text = re.sub(r'!\[.*?\]\((https?://[^\)\s]+)\)', '', text)
    clean_text = re.sub(r'\[Attachment:\s*(https?://[^\]\s]+)\]', '', clean_text, flags=re.IGNORECASE).strip()

    if valid_images:
        content = [{"type": "text", "text": clean_text + "\n" + (context or "")}]
        for img_url in valid_images:
            content.append({
                "type": "image_url",
                "image_url": {"url": img_url}
            })
        return HumanMessage(content=content)
    else:
        return HumanMessage(content=text + (context or ""))


def format_history(history: list[dict]):
    messages = []
    for msg in history:
        content = msg.get("content", "")
        role = msg.get("role")
        
        if role == "user":
            messages.append(construct_user_message(content, msg.get("images", [])))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


@router.post("/upload-doc")
async def upload_document(session_id: str = Form(...), file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        file_url = process_and_store_document(file_bytes, session_id, file.filename)
        
        return {
            "status": "success", 
            "message": f"File {file.filename} processed successfully.",
            "url": file_url
        }
    except Exception as e:
        logging.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat")
async def chat_with_agent(request: ChatRequest):
    try:
        messages = format_history(request.history)
        session_id = request.sessionId or (request.history[0].get("session_id") if request.history else "DEFAULT_SESSION")
        
        try:
            document_context = retrieve_relevant_context(request.message, session_id)
        except Exception as e:
            logging.warn(f"RAG Retrieval failed or no docs found: {e}")
            document_context = ""
        
        dynamic_prompts = []
        if request.webSearch:
            dynamic_prompts.append("USER ENABLED WEB SEARCH...")
            
        if dynamic_prompts:
            messages.append(SystemMessage(content="\n\n".join(dynamic_prompts)))
        
        messages.append(construct_user_message(request.message, request.images, document_context))
        
        final_state = general_agent.invoke({"messages": messages}, config={"recursion_limit": 20})
        return {"response": final_state["messages"][-1].content}
        
    except Exception as e:
        logging.error(f"General chat error: {e}")
        return {"response": "Error processing request.", "error": str(e)}


@router.post("/study")
async def study_with_agent(request: ChatRequest):
    try:
        messages = format_history(request.history)
        session_id = request.sessionId or "default-study"
        
        try:
            document_context = retrieve_relevant_context(request.message, session_id)
        except Exception as e:
            logging.warn(f"RAG Retrieval failed for Study Session: {e}")
            document_context = ""

        dynamic_prompts = []
        if request.webSearch:
            dynamic_prompts.append("USER ENABLED WEB SEARCH: You MUST proactively use your `web_search` tool over multiple iterations to fetch up-to-date facts if necessary.")
        if request.deepThink:
            dynamic_prompts.append("USER ENABLED DEEP THINK: You MUST deeply reason and 'deep think' through solutions step-by-step prior to returning your final output. Be extremely detailed and thoughtful in your reasoning.")
            
        if dynamic_prompts:
            messages.append(SystemMessage(content="\n\n".join(dynamic_prompts)))
            
        messages.append(construct_user_message(request.message, request.images, document_context))
        
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
    history: Optional[str] = Form(None), 
    file: Optional[UploadFile] = File(None)
):
    try:
        prompt = ""
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
        elif message:
            prompt = message
        else:
            raise HTTPException(status_code=400, detail="Must provide either a file upload or a chat message.")

        parsed_history = []
        if history:
            try:
                parsed_history = json.loads(history)
            except Exception as e:
                logging.error(f"History parse error: {e}")

        messages = format_history(parsed_history)
        messages.append(HumanMessage(content=prompt))
        
        final_state = resume_agent.invoke({"messages": messages}) 
        return {"response": final_state["messages"][-1].content}
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Resume analysis error: {e}")
        raise HTTPException(status_code=500, detail="Failed to process resume.")