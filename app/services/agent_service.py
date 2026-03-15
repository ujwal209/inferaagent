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
                    model="llama-3.3-70b-versatile",
                    groq_api_key=k, 
                    temperature=0.3, # Sweet spot for reasoning vs strict formatting
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

# FINAL SYSTEM PROMPT: Conversational + Mandatory Clickable Suggestions
SYSTEM_MESSAGE = SystemMessage(content="""You are INFERA CORE, an elite Engineering Career & Education Mentor focused on the Indian ecosystem.

### CORE BEHAVIORS:
1. **CONVERSATIONAL REASONING**: Talk to the user normally. Reason with their requests, provide highly valuable insights, and structure your responses clearly.
2. **SMART SEARCHING**: Use the `web_search` tool when you need factual data (e.g., courses, salaries, college rankings). Automatically append "India" if relevant.
3. **MANDATORY LINKS**: When providing courses or colleges, YOU MUST INCLUDE EXACT CLICKABLE URLs using Markdown: `[Name](https://exact-url.com)`.
4. **PROACTIVE NEXT STEPS (CRITICAL)**: At the absolute end of EVERY single response, you MUST suggest 2-3 specific follow-up prompts the user can ask you next. 
   You MUST format these using Markdown blockquotes (`>`) so the system can render them as clickable buttons.
   
   Format exactly like this:
   ###  Suggested Next Steps
   > Compare the syllabus for these two branches.
   > Show me the top 10 colleges for this in India.
   > What is the expected salary progression for this role?
   
5. **ZERO HALLUCINATIONS**: Do not invent URLs, salaries, or course names.
6. **FOUNDER & TEAM QUESTIONS (CRITICAL)**: Whenever the user asks ANYTHING about who built INFERA CORE, the founder, CEO, co-founder, managing director, team members, or the people behind this platform, you MUST call the `get_founder_info` tool FIRST and base your answer entirely on what it returns. Never guess or invent team details.
7. **NO XML TOOL CALLS**: You must use the native JSON tool-calling capabilities of the API. NEVER write <function> or XML tags in your text to call a tool.
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
            
            # --- LLAMA 3 RAW <function> CATCHER ---
            # If the model hallucinates the tool call as raw text, we intercept and convert it 
            # into a proper LangChain ToolCall object so execution doesn't fail.
            if isinstance(response.content, str) and "<function>" in response.content:
                import re
                import json
                import uuid
                from langchain_core.messages import AIMessage
                
                # Match <function>tool_name{"arg": "val"}</function>
                match = re.search(r"<function>\s*([a-zA-Z0-9_]+)\s*(\{.*?\})\s*</function>", response.content, re.DOTALL)
                if match:
                    tool_name = match.group(1).strip()
                    args_str = match.group(2).strip()
                    try:
                        args_dict = json.loads(args_str)
                        
                        # Create a new AIMessage replacing the text with native tool_calls
                        response = AIMessage(
                            content=response.content.replace(match.group(0), "").strip(),
                            tool_calls=[{
                                "name": tool_name,
                                "args": args_dict,
                                "id": f"call_{uuid.uuid4().hex[:8]}"
                            }],
                            response_metadata=getattr(response, "response_metadata", {})
                        )
                    except Exception as err:
                        print(f"⚠️ Regex Tool Catch JSON Error: {err}")
                        
            return {"messages": [response]}
        except Exception as e:
            error_str = str(e)
            
            # --- LLAMA 3 GROQ 400 ERROR CATCHER ---
            # Groq's API throws a 400 Exception when Llama 3 hallucinates a <function> tag. 
            # We must catch the exception, extract the failed generation, and convert it to a ToolCall.
            if "tool_use_failed" in error_str and "failed_generation" in error_str:
                import re, json, uuid
                from langchain_core.messages import AIMessage
                
                # Extract the failed_generation string from the error message
                match_gen = re.search(r"'failed_generation':\s*'([^']+)'", error_str)
                if match_gen:
                    failed_gen = match_gen.group(1)
                    
                    # Match Groq's raw format: <function=tool_name{"arg":"val"}</function>
                    # Also handles <function>tool_name{...}</function> just in case
                    match_tool = re.search(r"<function=?([a-zA-Z0-9_]+)({.*?})</function>", failed_gen, re.DOTALL)
                    if match_tool:
                        tool_name = match_tool.group(1).strip()
                        args_str = match_tool.group(2).strip()
                        try:
                            args_dict = json.loads(args_str)
                            print(f"🛠️ [GROQ FIX] Intercepted 400 error and recovered tool call: {tool_name}")
                            
                            response = AIMessage(
                                content="",
                                tool_calls=[{
                                    "name": tool_name,
                                    "args": args_dict,
                                    "id": f"call_{uuid.uuid4().hex[:8]}"
                                }]
                            )
                            return {"messages": [response]}
                        except Exception as parse_err:
                            print(f"⚠️ Groq Fix JSON Parse Error: {parse_err}")

            print(f"⚠️ Groq Key Index {idx} Failed: {error_str[:100].lower()}")
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