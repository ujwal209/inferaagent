import os
from typing import Annotated, TypedDict
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from app.tools.extensive_tools import tool_list

# Load environment safely
GROQ_KEYS = [k.strip() for k in os.getenv("GROQ_API_KEYS", os.getenv("GROQ_API_KEY", "")).split(",") if k.strip()]

class MultiKeyLLM:
    def __init__(self, keys: list[str]):
        if not keys:
            print("WARNING: No GROQ API keys provided!")
            self.llms = []
        else:
            self.llms = [
                ChatGroq(
                    model="llama-3.3-70b-versatile", 
                    groq_api_key=k, 
                    temperature=0,
                    max_retries=0
                ) for k in keys
            ]
        self.current_index = 0

    def get_llm(self):
        if not self.llms:
            raise ValueError("No GROQ API keys found in environment. Please add GROQ_API_KEY to Render.")
        llm = self.llms[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.llms)
        return llm

multi_llm = MultiKeyLLM(GROQ_KEYS)

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# FINAL SYSTEM PROMPT: Enforcing strict result limits and mandatory explanations
SYSTEM_MESSAGE = SystemMessage(content="""You are the 'Supabase Engineering Intelligence' agent.
You have access to 60+ specialized tools for engineering branches, careers, education, and rankings.

### OUTPUT RULES (STRICT):
1. **TOP 5 ONLY**: When listing courses, colleges, or any data, you MUST only show the top 5 most relevant results. Do NOT show 10, 15, or more.
2. **SUMMARY & JUSTIFICATION**: After every list of results, you MUST provide a separate section titled "### Why this data?". 
   - Explain how these specific 5 items are relevant to the user's branch (e.g., 'VLSI is the core of Integrated Circuit design in ECE').
   - Provide a 1-2 sentence summary of the career value of these results.
3. **TOOL USAGE**: Always use tools. Never imagine data.
4. **NO HALLUCINATIONS**: If the database data is generic, report it honestly but explain the context.

### RESPONSE STYLE:
- Professional, structured, and extremely helpful.
- For comparison requests, always use tables.
""")

def call_model(state: AgentState):
    messages = state["messages"]
    
    # Safely insert system message if not present
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SYSTEM_MESSAGE] + messages
    
    max_retries = max(1, len(multi_llm.llms))
    last_error = None
    
    for _ in range(max_retries):
        try:
            llm = multi_llm.get_llm()
            model_with_tools = llm.bind_tools(tool_list)
            response = model_with_tools.invoke(messages)
            return {"messages": [response]}
        except Exception as e:
            if "rate_limit_exceeded" in str(e).lower() or "429" in str(e):
                print(f"⚠️ Rate limit hit on key {multi_llm.current_index}. Switching to next key...")
                last_error = e
                continue
            else:
                print(f"CRITICAL LLM ERROR: {str(e)}")
                raise e 
                
    raise last_error or Exception("All Groq API keys failed.")

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