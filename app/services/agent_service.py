import os
import random
from typing import Annotated, TypedDict
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from app.tools.extensive_tools import tool_list

# Load environment safely
GROQ_KEYS = [k.strip() for k in os.getenv("GROQ_API_KEYS", os.getenv("GROQ_API_KEY", "")).split(",") if k.strip()]
random.shuffle(GROQ_KEYS) # Spread initial load

class MultiKeyLLM:
    def __init__(self, keys: list[str]):
        if not keys:
            print("WARNING: No GROQ API keys provided!")
            self.llms = []
        else:
            self.llms = [
                ChatGroq(
                    model="llama-3.3-70b-versatile", # 70b handles strict instructions better
                    groq_api_key=k, 
                    temperature=0,
                    max_retries=0
                ) for k in keys
            ]
        self.current_index = 0

    def get_llm_with_index(self):
        if not self.llms:
            raise ValueError("No GROQ API keys found. Please add GROQ_API_KEYS to .env")
        idx = self.current_index
        llm = self.llms[idx]
        self.current_index = (self.current_index + 1) % len(self.llms)
        return llm, idx

multi_llm = MultiKeyLLM(GROQ_KEYS)

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# FINAL SYSTEM PROMPT: Web-Only, India-Focused, Mandatory Summary
SYSTEM_MESSAGE = SystemMessage(content="""You are INFERA CORE, an elite Engineering Career & Education Consultant focused EXCLUSIVELY on the INDIAN education system and job market.

### CRITICAL DIRECTIVES:
1. **LIVE DATA ONLY**: You have no internal database. You MUST use your tools (`web_search` or `fetch_courses_api`) for EVERY query to get real-time information.
2. **ZERO HALLUCINATIONS**: Do NOT invent, guess, or assume any courses, salaries, or colleges. If the tools return no data, tell the user you cannot find the information.
3. **INDIAN CONTEXT**: Always frame your answers around the Indian ecosystem (e.g., NPTEL, IITs, NITs, LPA salaries, Indian tech hubs).
4. **MANDATORY SUMMARY**: At the very end of EVERY single response you generate, you MUST provide a distinct section titled "###  Summary" containing 2-3 bullet points that condense the most important takeaways from your response.
5. **TABLE FORMATTING**: Present all courses, rankings, or salaries in clean Markdown tables with exact clickable links.
""")

def call_model(state: AgentState):
    messages = state["messages"]
    
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SYSTEM_MESSAGE] + messages
    
    max_retries = max(len(multi_llm.llms), 3) 
    last_error = None
    
    for _ in range(max_retries):
        llm, idx = multi_llm.get_llm_with_index()
        try:
            model_with_tools = llm.bind_tools(tool_list)
            response = model_with_tools.invoke(messages)
            return {"messages": [response]}
        except Exception as e:
            error_str = str(e).lower()
            print(f"⚠️ Groq Key Index {idx} Failed: {error_str[:100]}")
            last_error = e
            continue
                
    raise last_error or Exception("All attempts to contact the LLM failed.")

def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    return "continue" if last_message.tool_calls else "end"

# Build Graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("action", ToolNode(tool_list))
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"continue": "action", "end": END})
workflow.add_edge("action", "agent")

agent_engine = workflow.compile()