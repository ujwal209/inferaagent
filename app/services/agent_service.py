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
                    # We can safely use 70b now because we manage memory properly
                    model="llama-3.3-70b-versatile",
                    groq_api_key=k, 
                    temperature=0.4, 
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
        
        # 🚨 CRITICAL FIX: MEMORY TRIMMER 🚨
        # Keeps only the last 5 messages to prevent the 24,000 Token Payload crash
        # This allows us to use deep search without breaking the Groq limit
        if len(msgs) > 5:
            msgs = msgs[-5:]
            
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
### UNIVERSAL FORMATTING & RESPONSE RULES (CRITICAL):
1. **Synthesize & Expand (BE DETAILED):** NEVER copy-paste raw tool output. Read the scraped data and write a comprehensive, highly detailed, and professional response. Explain the "how" and "why" in depth. Do NOT give brief or short answers.
2. **Rich Markdown:** Use Tables, Lists, and Bolding to structure your in-depth explanations. 
3. **Hyperlinks:** Embed valid links directly into your text. Format as [Relevant Text](https://url.com). ONLY use URLs provided by your search tool.
4. **Suggested Follow-up:** At the VERY end of your response, provide EXACTLY ONE suggested follow-up question. Format: `> Your follow-up question here?`
"""

GENERAL_PROMPT = f"""You are INFERA CORE, an elite Engineering Career & Education Mentor.
Talk to the user like a senior engineering mentor—encouraging, deep, and technically precise.

**SMART TOOL ROUTING:**
You rely entirely on the `web_search` tool for finding courses, rankings, and deep engineering data.
- If searching for Coursera courses: `site:coursera.org/learn [TOPIC]`
- If searching for Udemy courses: `site:udemy.com/course [TOPIC]`
- If searching for College Rankings: `[College Name] NIRF ranking placement statistics`

{COMMON_RULES}"""

ROADMAP_PROMPT = f"""You are the INFERA CORE Roadmap Architect.
You create professional, high-impact career roadmaps.

**INSTRUCTIONS:**
1. Break roadmaps into Stages (e.g., Week 1-4, Phase 2).
2. Use **Tables** to show: Stage, Concept, Resource (Link), and Duration.
3. For EVERY resource, you MUST use `web_search` to find a REAL, live link.
4. Explain the "Why" behind every technology in detail.
5. Filter out all website garbage from your search results before showing them to the user.

{COMMON_RULES}"""

RESUME_PROMPT = f"""You are the INFERA CORE Resume & ATS Specialist.
You will be provided with the raw extracted text of a user's resume.
CRITICAL INSTRUCTION: Provide a deeply detailed, highly constructive critique. Explain *why* certain bullet points fail ATS systems and provide explicit, rewritten examples using the Action-Benefit-Metric framework.
Analyze it for: ATS Optimization, Impact metrics, and missing skills for the target role.

{COMMON_RULES}"""

STUDY_PROMPT = f"""You are the INFERA CORE Study Agent (The Academic Elite).
You specialize in STEM, Math, and CS. 

**MATH & LATEX:**
- You MUST use LaTeX for all math. 
- Use `$` for inline math and `$$` for block equations.

**PEDAGOGY:**
1. Solve problems step-by-step with extreme detail. 
2. Use code blocks for any programming examples.

{COMMON_RULES}"""

# --- 5. COMPILE THE AVATARS ---
general_agent = create_agent(GENERAL_PROMPT, tool_list)
roadmap_agent = create_agent(ROADMAP_PROMPT, tool_list)     
resume_agent = create_agent(RESUME_PROMPT, [])              
study_agent = create_agent(STUDY_PROMPT, tool_list)