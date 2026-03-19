import os
import re
import json
import uuid
import random
import logging
from typing import Annotated, TypedDict

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

# Import the tools from our separate file
from app.tools.extensive_tools import tool_list 

# --- 1. MULTI-KEY LLM SETUP ---
GROQ_KEYS = [k.strip() for k in os.getenv("GROQ_API_KEYS", os.getenv("GROQ_API_KEY", "")).split(",") if k.strip()]
random.shuffle(GROQ_KEYS) 

class MultiKeyLLM:
    def __init__(self, keys: list[str]):
        if not keys:
            logging.warning("WARNING: No GROQ API keys provided!")
            self.llms = []
        else:
            self.llms = [
                ChatGroq(
                    model="llama-3.3-70b-versatile",
                    groq_api_key=k, 
                    temperature=0.3, # Lowered temperature for strict tool adherence
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

# --- 2. AGGRESSIVE TOOL INTERCEPTOR & CALLER ---
def invoke_model_with_fallbacks(messages, tools):
    """
    Handles Groq API rotations and aggressively intercepts Llama 3 tool text leaks.
    Forces leaked text into actual background tool executions.
    """
    max_retries = max(len(multi_llm.llms), 3) 
    last_error = None
    valid_tool_names = [t.name for t in tools] if tools else []
    
    for _ in range(max_retries):
        llm, idx = multi_llm.get_llm_with_index()
        try:
            # Bind the tools properly to the LLM
            model_to_use = llm.bind_tools(tools) if tools else llm
            response = model_to_use.invoke(messages)
            
            # 1. If native tool calling worked perfectly, return immediately
            if hasattr(response, "tool_calls") and len(response.tool_calls) > 0:
                return {"messages": [response]}
            
            # 2. THE JSON/XML INTERCEPTOR (Catching the Slop)
            content = str(response.content)
            tool_name = None
            args_dict = None
            
            # Scenario A: <web_search>{"keyword": "..."}</web_search>
            xml_match = re.search(r"<([a-zA-Z0-9_]+)>\s*(\{.*?\})\s*</\1>", content, re.DOTALL)
            # Scenario B: <function=web_search>{"keyword": "..."}</function>
            func_match = re.search(r"<function[=>\s]*([a-zA-Z0-9_]+)[>\s]*(\{.*?\})\s*</function>", content, re.DOTALL)
            # Scenario C: Just raw JSON dropped in the text like {"keyword": "..."}
            raw_match = re.search(r"(\{\s*\"(keyword|query)\"\s*:\s*\"[^\"]+\"\s*\})", content)

            if xml_match and xml_match.group(1).strip() in valid_tool_names:
                tool_name = xml_match.group(1).strip()
                try: args_dict = json.loads(xml_match.group(2).strip())
                except: pass

            elif func_match and func_match.group(1).strip() in valid_tool_names:
                tool_name = func_match.group(1).strip()
                try: args_dict = json.loads(func_match.group(2).strip())
                except: pass

            elif raw_match and tools:
                # If it's raw JSON with a "keyword" or "query", default to the first available tool
                tool_name = valid_tool_names[0] if valid_tool_names else None
                try: args_dict = json.loads(raw_match.group(1).strip())
                except: pass

            # 3. IF WE CAUGHT A LEAK: Scrub the text, force the tool call
            if tool_name and args_dict:
                # Create a clean AI message that executes the tool instead of showing text
                forced_response = AIMessage(
                    content="", # Erase the ugly leaked text
                    tool_calls=[{"name": tool_name, "args": args_dict, "id": f"call_{uuid.uuid4().hex[:8]}"}]
                )
                return {"messages": [forced_response]}
            
            # 4. If it's just normal conversation (e.g. "Hi, how are you?"), return standard text
            return {"messages": [response]}

        except Exception as e:
            error_str = str(e)
            
            # Catch Groq API internal "tool_use_failed" rejections and parse them manually
            if "tool_use_failed" in error_str and "failed_generation" in error_str:
                match_gen = re.search(r"'failed_generation':\s*'([^']+)'", error_str)
                if match_gen:
                    failed_gen = match_gen.group(1)
                    match_tool = re.search(r"<function[=>\s]*([a-zA-Z0-9_]+)>?({.*?})</function>", failed_gen, re.DOTALL)
                    if match_tool:
                        tool_name = match_tool.group(1).strip()
                        try:
                            args_dict = json.loads(match_tool.group(2).strip())
                            response = AIMessage(
                                content="",
                                tool_calls=[{"name": tool_name, "args": args_dict, "id": f"call_{uuid.uuid4().hex[:8]}"}]
                            )
                            return {"messages": [response]}
                        except Exception:
                            pass
            
            logging.warning(f"Groq Key Index {idx} Failed: {error_str[:100]}")
            last_error = e
            continue
            
    raise last_error or Exception("All LLM attempts failed.")


# --- 3. AGENT AVATAR FACTORY ---
def create_agent(system_prompt: str, tools: list = []):
    """Factory to create a specialized LangGraph agent."""
    sys_msg = SystemMessage(content=system_prompt)

    def call_node(state: AgentState):
        msgs = state["messages"]
        
        # CRITICAL FIX: MEMORY TRIMMER
        # Keeps only the last 5 messages to prevent the 24,000 Token Payload crash
        if len(msgs) > 5:
            msgs = msgs[-5:]
            
        if not any(isinstance(m, SystemMessage) for m in msgs):
            msgs = [sys_msg] + msgs
            
        return invoke_model_with_fallbacks(msgs, tools)

    def should_continue(state: AgentState):
        last_message = state["messages"][-1]
        # Route to the action node ONLY if a tool call was successfully generated/intercepted
        return "continue" if getattr(last_message, "tool_calls", None) else "end"

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_node)
    
    if tools:
        workflow.add_node("action", ToolNode(tools))
        workflow.add_conditional_edges("agent", should_continue, {"continue": "action", "end": END})
        # Loop back to agent after tool finishes to summarize results
        workflow.add_edge("action", "agent")
    else:
        workflow.add_edge("agent", END)

    workflow.set_entry_point("agent")
    return workflow.compile()


# --- 4. DETAILED SYSTEM PROMPTS ---

COMMON_RULES = """
You are a professional AI. 
If the user asks a general greeting like "hi" or "how are you", reply warmly and conversationally. DO NOT attempt to use tools for greetings.
For all technical questions, synthesize the data and write a comprehensive, highly detailed response using Markdown (Tables, Bold text, Lists).
Embed source links when applicable.
"""

GENERAL_PROMPT = f"""You are INFERA CORE, an elite Engineering Career & Education Mentor.
If the user asks for data, courses, salaries, or news, you MUST use the `web_search` tool to fetch live information.
Do not make up facts. Rely on your tools.

{COMMON_RULES}"""

ROADMAP_PROMPT = f"""You are the INFERA CORE Roadmap Architect.
You create professional, high-impact career roadmaps.
Use the `web_search` tool to find live, real-world links for the courses and resources you recommend.
Format your roadmap using clear Markdown tables showing: Stage, Concept, Resource Link, and Duration.

{COMMON_RULES}"""

RESUME_PROMPT = f"""You are the INFERA CORE Resume & ATS Specialist.
Critique the user's resume thoroughly. Explain why certain points fail ATS parsing and rewrite them using the Action-Benefit-Metric framework.

{COMMON_RULES}"""

STUDY_PROMPT = f"""You are the INFERA CORE Study Agent.
You specialize in STEM, Math, and Computer Science.
Solve problems step-by-step. You must format all mathematical formulas and equations using LaTeX.
Use single `$` for inline math and double `$$` for block equations.

{COMMON_RULES}"""

# --- 5. COMPILE THE AVATARS ---
general_agent = create_agent(GENERAL_PROMPT, tool_list)
roadmap_agent = create_agent(ROADMAP_PROMPT, tool_list)     
resume_agent = create_agent(RESUME_PROMPT, [])              
study_agent = create_agent(STUDY_PROMPT, tool_list)