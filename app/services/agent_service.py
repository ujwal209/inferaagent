import os
import re
import json
import uuid
import random
import logging
from typing import Annotated, TypedDict, List, Any

from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

# Import backend tools
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
                    temperature=0.1, # Extremely low temperature for strict schema adherence
                    max_retries=3    # FIX: Increased from 0 to survive micro rate-limiting
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


# --- 2. STRICT PYDANTIC UI WIDGET SCHEMAS ---

class QuizQuestion(BaseModel):
    """A single question inside the QuizWidget."""
    question: str = Field(description="A specific, thought-provoking question testing the concept just taught.")
    options: List[str] = Field(description="Exactly 4 distinct options for the user to choose from.", min_length=4, max_length=4)
    correctIndex: int = Field(description="The integer index (0, 1, 2, or 3) of the correct option.")
    explanation: str = Field(description="A detailed explanation of why the correct option is the right answer.")

class QuizWidget(BaseModel):
    """Generates an interactive Assessment Exam."""
    topic: str = Field(description="The main topic being tested.")
    questions: List[QuizQuestion] = Field(
        description="List of questions. You MUST generate EXACTLY 5 questions. NEVER generate 1 question. Provide 3 conceptual and 2 very hard numerical questions.", 
        min_length=5, 
        max_length=5
    )

class ProgressWidget(BaseModel):
    """Generates an interactive Progress Tracker."""
    topic: str = Field(description="The main subject currently being studied")
    masteryPercentage: int = Field(description="An integer from 0 to 100 estimating their completion of the topic")
    completedConcepts: List[str] = Field(description="List of micro-concepts the user has successfully learned so far")
    nextConcept: str = Field(description="The name of the very next concept to learn")


# --- 3. AGGRESSIVE TOOL INTERCEPTOR & FORMATTER ---
def process_intercepted_response(response: AIMessage) -> AIMessage:
    """
    Intercepts native tool calls meant for the UI (Quiz/Progress) 
    and converts them into the json?chameleon markdown string.
    Leaves backend tools (like web_search) intact for the ToolNode to execute.
    """
    if not hasattr(response, "tool_calls") or not response.tool_calls:
        return response

    final_content = response.content or ""
    backend_tool_calls = []

    for tc in response.tool_calls:
        if tc["name"] in ["QuizWidget", "ProgressWidget"]:
            args = tc.get("args", {})
            
            # BULLETPROOF FIX: Llama 3 often stringifies nested JSON arrays. 
            if tc["name"] == "QuizWidget" and "questions" in args and isinstance(args["questions"], str):
                try:
                    args["questions"] = json.loads(args["questions"])
                except Exception as e:
                    logging.error(f"Auto-fixer failed to parse stringified questions: {e}")
            
            # Format UI tools directly into the frontend markdown schema
            widget_json = {
                "component": tc["name"],
                "props": args
            }
            final_content += f"\n\n```json?chameleon\n{json.dumps(widget_json, indent=2)}\n```\n"
        else:
            # Preserve backend tools
            backend_tool_calls.append(tc)

    return AIMessage(
        content=final_content.strip(),
        tool_calls=backend_tool_calls,
        id=response.id or str(uuid.uuid4())
    )

def invoke_model_with_fallbacks(messages, tools):
    max_retries = max(len(multi_llm.llms), 3) 
    last_error = None
    
    valid_tool_names = []
    if tools:
        for t in tools:
            valid_tool_names.append(getattr(t, "name", getattr(t, "__name__", "")))
    
    for _ in range(max_retries):
        llm, idx = multi_llm.get_llm_with_index()
        try:
            model_to_use = llm.bind_tools(tools) if tools else llm
            response = model_to_use.invoke(messages)
            
            # Catch raw text leaks
            content = str(response.content)
            tool_name, args_dict = None, None
            
            xml_match = re.search(r"<([a-zA-Z0-9_]+)>\s*(\{.*?\})\s*</\1>", content, re.DOTALL)
            func_match = re.search(r"<function[=>\s]*([a-zA-Z0-9_]+)[>\s]*(\{.*?\})\s*</function>", content, re.DOTALL)
            raw_match = re.search(r"(\{\s*\"(topic|keyword|query)\"\s*:\s*\"[^\"]+\"\s*\})", content)

            if xml_match and xml_match.group(1).strip() in valid_tool_names:
                tool_name, args_dict = xml_match.group(1).strip(), json.loads(xml_match.group(2).strip())
            elif func_match and func_match.group(1).strip() in valid_tool_names:
                tool_name, args_dict = func_match.group(1).strip(), json.loads(func_match.group(2).strip())
            elif raw_match and tools and "QuizWidget" in valid_tool_names:
                tool_name, args_dict = "QuizWidget", json.loads(raw_match.group(1).strip())

            # Reconstruct leaked text into a proper ToolCall
            if tool_name and args_dict:
                response = AIMessage(
                    content="", 
                    tool_calls=[{"name": tool_name, "args": args_dict, "id": f"call_{uuid.uuid4().hex[:8]}"}]
                )
            
            final_response = process_intercepted_response(response)
            return {"messages": [final_response]}

        except Exception as e:
            error_str = str(e)
            
            # Catch Groq's internal API parsing failures
            if "tool_use_failed" in error_str and "failed_generation" in error_str:
                match_gen = re.search(r"'failed_generation':\s*'([^']+)'", error_str)
                if match_gen:
                    failed_gen = match_gen.group(1)
                    match_tool = re.search(r"<function[=>\s]*([a-zA-Z0-9_]+)>?({.*?})</function>", failed_gen, re.DOTALL)
                    if match_tool:
                        try:
                            t_name = match_tool.group(1).strip()
                            t_args = json.loads(match_tool.group(2).strip())
                            resp = AIMessage(
                                content="",
                                tool_calls=[{"name": t_name, "args": t_args, "id": f"call_{uuid.uuid4().hex[:8]}"}]
                            )
                            return {"messages": [process_intercepted_response(resp)]}
                        except Exception: pass
            
            logging.warning(f"Groq Key Index {idx} Failed: {error_str[:100]}")
            last_error = e
            continue
            
    raise last_error or Exception("All LLM attempts failed.")


# --- 4. AGENT AVATAR FACTORY ---
def create_agent(system_prompt: str, executable_tools: list = [], ui_tools: list = []):
    """Factory to create a specialized LangGraph agent."""
    sys_msg = SystemMessage(content=system_prompt)
    
    all_model_tools = executable_tools + ui_tools

    def call_node(state: AgentState):
        msgs = state["messages"]
        
        # Separate the system message from the chat history
        system_msgs = [m for m in msgs if isinstance(m, SystemMessage)]
        if not system_msgs:
            system_msgs = [sys_msg]
            
        chat_history = [m for m in msgs if not isinstance(m, SystemMessage)]
        
        # FIX: Restored memory capacity to 40 messages (20 full conversational turns).
        # Ensures deep context without blowing up the token payload limit.
        if len(chat_history) > 40:
            chat_history = chat_history[-40:]
            
            # CRITICAL SAFETIES FOR GROQ STRICT API:
            # 1. Never start the array with an orphaned ToolMessage
            while chat_history and isinstance(chat_history[0], ToolMessage):
                chat_history.pop(0)
                
            # 2. Never start the array with an AIMessage that made a tool call 
            #    if we just deleted its corresponding ToolMessage.
            while chat_history and getattr(chat_history[0], "tool_calls", None):
                if len(chat_history) > 1 and isinstance(chat_history[1], ToolMessage):
                    break # It has its pair, we're safe!
                else:
                    chat_history.pop(0) # Orphaned, remove it to prevent 400 Bad Request
                
        safe_msgs = system_msgs + chat_history
            
        return invoke_model_with_fallbacks(safe_msgs, all_model_tools)

    def should_continue(state: AgentState):
        last_message = state["messages"][-1]
        return "continue" if getattr(last_message, "tool_calls", None) else "end"

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_node)
    
    if executable_tools:
        workflow.add_node("action", ToolNode(executable_tools))
        workflow.add_conditional_edges("agent", should_continue, {"continue": "action", "end": END})
        workflow.add_edge("action", "agent")
    else:
        workflow.add_edge("agent", END)

    workflow.set_entry_point("agent")
    return workflow.compile()


# --- 5. DETAILED SYSTEM PROMPTS ---

COMMON_RULES = """
You are a professional AI named INFERA CORE. 
If the user asks a general greeting like "hi" or "how are you", reply warmly and conversationally. DO NOT attempt to use tools for greetings.
For all technical questions, synthesize the data and write a comprehensive, highly detailed response using Markdown (Tables, Bold text, Lists).
Embed source links when applicable.
"""

GENERAL_PROMPT = f"""You are INFERA CORE, an elite Engineering Career & Education Mentor.
If the user asks for data, courses, salaries, or news, you MUST use the `web_search` tool to fetch live information.
Do not make up facts. Rely on your tools.

{COMMON_RULES}"""

RESUME_PROMPT = f"""You are the INFERA CORE Resume & ATS Specialist.
Critique the user's resume thoroughly. Explain why certain points fail ATS parsing and rewrite them using the Action-Benefit-Metric framework.

{COMMON_RULES}"""

STUDY_PROMPT = f"""You are the INFERA CORE Neural Study Buddy, an elite technical tutor specializing in STEM, Math, and Computer Science.

YOUR PRIME DIRECTIVE: Explain concepts deeply and thoroughly before ever testing the user. You must use the Socratic method.
1. Break complex subjects into small, digestible micro-concepts.
2. Explain ONE micro-concept clearly, concisely, and with deep detail. Provide examples.
3. END EVERY EXPLANATION WITH A SUGGESTED NEXT STEP: At the very end of your text response, explicitly offer a concrete next step to guide the user. 
   - Example 1: "Shall we move on to [Next Concept Name]?"
   - Example 2: "Would you like me to explain [Specific part] in more detail?"

STRICT FORMATTING RULES (LATEX):
- You MUST use strict LaTeX for ALL mathematical symbols, variables, formulas, and equations in your response.
- Use double `$$` for block equations (e.g., $$ E = mc^2 $$). 
- Use single `$` for inline math, variables, and numbers related to equations (e.g., $x$, $v = 5 m/s$). 
- NEVER use plain text for math.

TESTING & EXAM PROTOCOL (CRITICAL - STRICT ADHERENCE REQUIRED):
- DO NOT test the user constantly. Explain things first.
- THE 5-QUESTION EXAM RULE: WHENEVER a quiz is triggered (either because 5 concepts have been taught, OR the user explicitly clicked 'Take Quiz'), you MUST generate an EXACTLY 5-QUESTION EXAM.
- EXAM FORMAT: You MUST call the `QuizWidget` tool and pass an array of EXACTLY 5 questions into the `questions` parameter. 
  - The 5 questions MUST be: 3 conceptual questions and 2 highly difficult numerical questions.
  - NEVER generate a 1-question quiz. Every single quiz must be exactly 5 questions.

TOOL BEHAVIOR & MULTITASKING (CRITICAL - DO NOT IGNORE):
- ALWAYS INCLUDE TEXT: NEVER output just a tool call or widget. Even when calling the `ProgressWidget` or `QuizWidget`, you MUST include your conversational text, explanation, and your suggested next step in the same response.
- When a user submits quiz answers and tells you their score:
  1. Acknowledge and praise them based on their score in standard text.
  2. Call the `ProgressWidget` to visually update their progress.
  3. In the SAME text response, IMMEDIATELY begin explaining the NEXT sub-concept in detail. DO NOT wait for them to ask.
  4. End with your Suggested Next Step.

{COMMON_RULES}"""

# --- 6. COMPILE THE AVATARS ---
general_agent = create_agent(GENERAL_PROMPT, executable_tools=tool_list, ui_tools=[])
resume_agent = create_agent(RESUME_PROMPT, executable_tools=[], ui_tools=[])              
study_agent = create_agent(STUDY_PROMPT, executable_tools=tool_list, ui_tools=[QuizWidget, ProgressWidget])