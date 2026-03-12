import os
from typing import Annotated, TypedDict
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from app.tools.extensive_tools import tool_list

# Load environment
GROQ_KEYS = [k.strip() for k in os.getenv("GROQ_API_KEYS", os.getenv("GROQ_API_KEY", "")).split(",") if k.strip()]

class MultiKeyLLM:
    def __init__(self, keys: list[str]):
        # Set max_retries=0 to prevent internal waiting and trigger our failover immediately
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
            raise ValueError("No GROQ API keys found in environment.")
        # Simple cycle - each call moves to next key to distribute load
        llm = self.llms[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.llms)
        return llm

multi_llm = MultiKeyLLM(GROQ_KEYS)

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

SYSTEM_MESSAGE = SystemMessage(content="""You are an elite Engineering Career & Education Consultant.
You MUST follow these strict operational rules:

1. **TOOL USAGE IS MANDATORY**: If you need to search for courses, branches, colleges, or rankings, you MUST use the provided tools. 
2. **NO INTERNAL CODE**: Never output tags like '<|python_tag|>' or try to write your own database queries. Use the specialized tool for the specific task.
3. **ONLY USE PROVIDED TOOLS**: Your only way to get data is through the tools in your 'tool_list'. 
4. **DATA SCHEMA**:
   - Courses -> use `get_courses_by_branch`
   - Branches -> use `get_branch_details`
   - Jobs -> use `get_job_roles_by_branch`
   - Salaries -> use `get_salary_insights`
5. **SEARCH VARIATIONS**: If 'ECE' fails, automatically retry with 'Electronics' using the appropriate tool.

RESPONSE FORMAT:
- If you call a tool, your output should be the tool call.
- Once you have the data, provide a premium, human-friendly summary with clickable links.
- DO NOT invent your own query language or tags.
""")

def call_model(state: AgentState):
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SYSTEM_MESSAGE] + messages
    
    # Try calling Groq with fallback logic
    max_retries = len(multi_llm.llms)
    last_error = None
    
    for _ in range(max_retries):
        try:
            llm = multi_llm.get_llm()
            model_with_tools = llm.bind_tools(tool_list)
            response = model_with_tools.invoke(messages)
            return {"messages": [response]}
        except Exception as e:
            # Check for Rate Limit (429) specifically
            if "rate_limit_exceeded" in str(e).lower() or "429" in str(e):
                print(f"⚠️ Rate limit hit on key {multi_llm.current_index}. Switching to next key...")
                last_error = e
                continue # Try next key
            else:
                raise e # Re-raise if it's a different kind of error
                
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
