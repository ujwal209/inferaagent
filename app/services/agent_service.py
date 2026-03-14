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
                    model="llama-3.3-70b-versatile", # Using 70b for better reasoning, or revert to 8b if you prefer speed
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

# FINAL SYSTEM PROMPT: Web-First, India-Focused, Zero Hallucinations
SYSTEM_MESSAGE = SystemMessage(content="""You are INFERA CORE, an elite Engineering Career & Education Consultant focused EXCLUSIVELY on the INDIAN education system and job market (AICTE, IITs, NITs, LPA salaries).

### CRITICAL DIRECTIVES:
1. **WEB SEARCH IS YOUR PRIMARY TOOL**: You MUST prioritize using the `web_search` tool for almost all queries regarding top courses, colleges, rankings, and emerging tech trends. Do not rely solely on internal knowledge.
2. **ZERO HALLUCINATIONS**: Do NOT invent, guess, or assume any courses, colleges, or statistics. If you do not have exact data from a tool, you MUST use `web_search` to find it. If web search fails, tell the user you cannot find the data.
3. **INDIAN CONTEXT**: Always filter your advice for the Indian context (e.g., NPTEL, Swayam, Indian job market).
4. **FORMATTING**: When a tool (like web_search or a DB tool) returns data, present it cleanly using Markdown tables. Include exact URLs/Links if the tool provides them.
5. **VERBATIM COPY**: If you extract course links or college names from a tool, copy them EXACTLY as provided. Do not alter URLs.
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