from fastapi import APIRouter, HTTPException
from app.models.schemas import ChatRequest
from app.services.agent_service import agent_engine
from langchain_core.messages import HumanMessage, AIMessage
import logging

router = APIRouter()

@router.post("/chat")
async def chat_with_agent(request: ChatRequest):
    try:
        messages = []
        for msg in request.history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        messages.append(HumanMessage(content=request.message))
        
        inputs = {"messages": messages}
        config = {"recursion_limit": 10}
        
        final_state = agent_engine.invoke(inputs, config=config)
        final_response = final_state["messages"][-1].content
        
        return {"response": final_response}
    except Exception as e:
        logging.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
