import os
import re
import json
import uuid
import random
import logging
from typing import Annotated, TypedDict

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

# Make sure this points to your actual tools file
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
                    temperature=0.4, # Slightly higher for richer explanations
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

# --- 2. ROBUST MODEL CALLER ---
def invoke_model_with_fallbacks(messages, tools):
    """Handles Groq API rotations and Llama 3 tool hallucination catching."""
    max_retries = max(len(multi_llm.llms), 3) 
    last_error = None
    
    for _ in range(max_retries):
        llm, idx = multi_llm.get_llm_with_index()
        try:
            model_to_use = llm.bind_tools(tools) if tools else llm
            response = model_to_use.invoke(messages)
            
            # Catch raw <function> tags in text output
            if isinstance(response.content, str) and "<function>" in response.content:
                match = re.search(r"<function>\s*([a-zA-Z0-9_]+)\s*(\{.*?\})\s*</function>", response.content, re.DOTALL)
                if match:
                    tool_name, args_str = match.group(1).strip(), match.group(2).strip()
                    try:
                        args_dict = json.loads(args_str)
                        response = AIMessage(
                            content=response.content.replace(match.group(0), "").strip(),
                            tool_calls=[{"name": tool_name, "args": args_dict, "id": f"call_{uuid.uuid4().hex[:8]}"}]
                        )
                    except Exception as err:
                        logging.error(f"Regex Tool Catch Error: {err}")
            
            return {"messages": [response]}

        except Exception as e:
            error_str = str(e)
            
            # Catch Groq 400 Tool Use Failed errors
            if "tool_use_failed" in error_str and "failed_generation" in error_str:
                match_gen = re.search(r"'failed_generation':\s*'([^']+)'", error_str)
                if match_gen:
                    failed_gen = match_gen.group(1)
                    match_tool = re.search(r"<function=?([a-zA-Z0-9_]+)({.*?})</function>", failed_gen, re.DOTALL)
                    if match_tool:
                        tool_name, args_str = match_tool.group(1).strip(), match_tool.group(2).strip()
                        try:
                            args_dict = json.loads(args_str)
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
        if not any(isinstance(m, SystemMessage) for m in msgs):
            msgs = [sys_msg] + msgs
        return invoke_model_with_fallbacks(msgs, tools)

    def should_continue(state: AgentState):
        last_message = state["messages"][-1]
        return "continue" if getattr(last_message, "tool_calls", None) else "end"

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_node)
    
    if tools:
        workflow.add_node("action", ToolNode(tools))
        workflow.add_conditional_edges("agent", should_continue, {"continue": "action", "end": END})
        workflow.add_edge("action", "agent")
    else:
        workflow.add_edge("agent", END)

    workflow.set_entry_point("agent")
    return workflow.compile()


# --- 4. DETAILED SYSTEM PROMPTS ---

COMMON_RULES = """
### UNIVERSAL RULES:
1. NORMAL CHAT: If the user says "hi", "hello", or asks a casual question, respond naturally and politely. Do not force a deep analysis for a simple greeting.
2. EXACTLY ONE NEXT STEP: At the absolute end of your response, you must provide EXACTLY ONE suggested follow-up question the user can ask. Format it strictly as a blockquote starting with `>`. Do not provide multiple steps.
"""

GENERAL_PROMPT = f"""You are INFERA CORE, an elite Engineering Career & Education Mentor.
CRITICAL INSTRUCTION: Provide rich, detailed, and highly educational explanations. Talk to the user like a senior engineering mentor.
1. CONVERSATIONAL REASONING: Break down complex topics so they are easy to understand.
2. SMART TOOL USE — always pick the RIGHT tool:
   - For NPTEL courses, NPTEL links, or NPTEL content → use `search_nptel_courses`
   - For Coursera courses or Coursera links → use `search_coursera_courses`
   - For Udemy courses or Udemy links → use `search_udemy_courses`
   - For general web searches, news, salaries, job info → use `web_search`
   - For questions about INFERA CORE founders or team → use `get_founder_info`
3. MANDATORY LINKS: Format as [Name](https://exact-url.com).
4. ZERO HALLUCINATIONS.
{COMMON_RULES}"""

ROADMAP_PROMPT = f"""You are the INFERA CORE Roadmap Architect.
Your sole job is to create highly structured, week-by-week or month-by-month technical roadmaps.
CRITICAL INSTRUCTION: Provide a deep, comprehensive explanation for EVERY phase. Don't just list a technology, explain *why* it's important, *what* concepts to focus on, and *how* it connects to the role.
SMART TOOL USE — always pick the RIGHT tool:
   - For NPTEL courses, NPTEL links, or NPTEL content → use `search_nptel_courses`
   - For Coursera courses or Coursera links → use `search_coursera_courses`
   - For Udemy courses or Udemy links → use `search_udemy_courses`
   - For general web searches, salaries, latest trends, certifications → use `web_search`
{COMMON_RULES}"""

RESUME_PROMPT = f"""You are the INFERA CORE Resume & ATS Specialist.
You will be provided with the raw extracted text of a user's resume.
CRITICAL INSTRUCTION: Provide a deeply detailed, highly constructive critique. Explain *why* certain bullet points fail ATS systems and provide explicit, rewritten examples using the Action-Benefit-Metric framework.
Analyze it for: ATS Optimization, Impact metrics, and missing skills for the target role.
{COMMON_RULES}"""

STUDY_PROMPT = f"""You are the INFERA CORE Study Agent.
You are an elite academic tutor specializing in advanced mathematics (Linear Algebra, Calculus, Discrete Math), Computer Science, and core Engineering concepts.
CRITICAL INSTRUCTIONS:
1. Solve problems step-by-step. Do not skip steps. Explain the intuition behind the math.
2. LATEX MATH: You MUST format all mathematical formulas, equations, and variables using LaTeX. Use `$` for inline equations (e.g., $E = mc^2$) and `$$` for block equations. Do NOT put spaces between the $ and the formula.
3. SMART TOOL USE:
   - For NPTEL courses or NPTEL links → use `search_nptel_courses`
   - For Coursera courses or Coursera links → use `search_coursera_courses`
   - For Udemy courses or Udemy links → use `search_udemy_courses`
   - For academic definitions, formulas, or study resources → use `web_search`
{COMMON_RULES}"""

# --- 5. COMPILE THE AVATARS ---
general_agent = create_agent(GENERAL_PROMPT, tool_list)
roadmap_agent = create_agent(ROADMAP_PROMPT, tool_list)     
resume_agent = create_agent(RESUME_PROMPT, [])              
study_agent = create_agent(STUDY_PROMPT, tool_list) 