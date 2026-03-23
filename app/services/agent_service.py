import os
import json
import uuid
import random
import logging
from typing import Annotated, TypedDict, List, Any

from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

# Import backend tools
from app.tools.extensive_tools import tool_list 

# ==========================================
# 1. MULTI-KEY LLM SETUP (ZERO TEMPERATURE)
# ==========================================
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
                    temperature=0.0, # STRICT 0.0 TO KILL HALLUCINATIONS & ENSURE PURE JSON
                    max_retries=3
                ) for k in keys
            ]
        self.current_index = 0

    def get_llm_with_index(self):
        if not self.llms:
            raise ValueError("No GROQ API keys found.")
        idx = self.current_index
        llm = self.llms[idx]
        self.current_index = (self.current_index + 1) % len(self.llms)
        return llm, idx

multi_llm = MultiKeyLLM(GROQ_KEYS)

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ==========================================
# 2. EXTREMELY STRICT PYDANTIC UI SCHEMAS
# ==========================================
class QuizQuestion(BaseModel):
    """A single question inside the QuizWidget."""
    question: str = Field(description="A specific, thought-provoking question testing the concept just taught.")
    options: List[str] = Field(description="Exactly 4 distinct options. NO JOKE OPTIONS.", min_length=4, max_length=4)
    correctIndex: int = Field(description="The integer index (0, 1, 2, or 3) of the correct option.")
    explanation: str = Field(description="A detailed, step-by-step explanation of why the correct option is right.")

class QuizWidget(BaseModel):
    """Generates an interactive Assessment Exam."""
    topic: str = Field(description="The main topic being tested.")
    difficulty: str = Field(description="Strictly one of: 'Foundational' (0-30%), 'Intermediate' (35-70%), or 'Expert' (75-100%).")
    questions: List[QuizQuestion] = Field(
        description="MUST generate EXACTLY 10 questions. Provide conceptual and analytical questions.", 
        min_length=10, 
        max_length=10
    )

class ProgressWidget(BaseModel):
    """Generates an interactive Progress Tracker."""
    topic: str = Field(description="The overarching subject currently being studied.")
    masteryPercentage: int = Field(description="OVERALL course completion. MUST INCREASE BY EXACTLY 5% for every new concept learned. (e.g., 5, 10, 15...). Max 100.")
    completedConcepts: List[str] = Field(description="List of specific micro-concepts the user has successfully learned so far")
    nextConcept: str = Field(description="The name of the very next concept to learn")


# ==========================================
# 3. PURE NATIVE INTERCEPTOR (NO REGEX)
# ==========================================
def process_intercepted_response(response: AIMessage) -> AIMessage:
    """Takes native Pydantic tool calls and formats UI widgets for the frontend."""
    if not hasattr(response, "tool_calls") or not response.tool_calls:
        return response

    final_content = response.content or ""
    backend_tool_calls = []

    for tc in response.tool_calls:
        if tc["name"] in ["QuizWidget", "ProgressWidget"]:
            # Pure Pydantic Output - No manual parsing required
            widget_json = {
                "component": tc["name"],
                "props": tc.get("args", {})
            }
            final_content += f"\n\n```json?chameleon\n{json.dumps(widget_json, indent=2)}\n```\n"
        else:
            backend_tool_calls.append(tc)

    return AIMessage(
        content=final_content.strip(),
        tool_calls=backend_tool_calls,
        id=response.id or str(uuid.uuid4())
    )

def invoke_model_with_retries(messages, tools):
    """Clean execution using native `.bind_tools()`. No manual fallback parsing."""
    max_retries = max(len(multi_llm.llms), 3) 
    last_error = None
    
    for _ in range(max_retries):
        llm, idx = multi_llm.get_llm_with_index()
        try:
            model_to_use = llm.bind_tools(tools) if tools else llm
            response = model_to_use.invoke(messages)
            
            # Process response natively
            final_response = process_intercepted_response(response)
            return {"messages": [final_response]}

        except Exception as e:
            logging.warning(f"Groq Key Index {idx} Failed: {str(e)[:100]}")
            last_error = e
            continue
            
    # fallback to just text if tools fail completely
    # return {"messages": [AIMessage(content="Sorry, I am having trouble compiling widgets. Let's continue textually.")]}
    if last_error is not None:
        raise last_error
    raise Exception("All LLM attempts failed. Please check API constraints.")


# ==========================================
# 4. AGENT AVATAR FACTORY
# ==========================================
def create_agent(system_prompt: str, executable_tools: list = [], ui_tools: list = []):
    sys_msg = SystemMessage(content=system_prompt)
    all_model_tools = executable_tools + ui_tools

    def call_node(state: AgentState):
        msgs = state["messages"]
        system_msgs = [m for m in msgs if isinstance(m, SystemMessage)]
        if not system_msgs:
            system_msgs = [sys_msg]
            
        chat_history = [m for m in msgs if not isinstance(m, SystemMessage)]
        
        if len(chat_history) > 40:
            chat_history = chat_history[-40:]
            while chat_history and getattr(chat_history[0], "type", "") == "tool":
                chat_history.pop(0)
            while chat_history and getattr(chat_history[0], "tool_calls", None):
                if len(chat_history) > 1 and getattr(chat_history[1], "type", "") == "tool":
                    break 
                else:
                    chat_history.pop(0) 
                
        safe_msgs = system_msgs + chat_history
        return invoke_model_with_retries(safe_msgs, all_model_tools)

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


def create_study_agent(system_prompt: str, executable_tools: list = []):
    """Dedicated Study Agent with smart widget orchestration via LangGraph."""
    sys_msg = SystemMessage(content=system_prompt)
    # Study agent has access to: web tools + UI widgets
    ui_tools = [QuizWidget, ProgressWidget]
    all_tools = executable_tools + ui_tools

    def call_node(state: AgentState):
        msgs = state["messages"]
        system_msgs = [m for m in msgs if isinstance(m, SystemMessage)]
        if not system_msgs:
            system_msgs = [sys_msg]
            
        chat_history = [m for m in msgs if not isinstance(m, SystemMessage)]
        
        if len(chat_history) > 40:
            chat_history = chat_history[-40:]
            while chat_history and getattr(chat_history[0], "type", "") == "tool":
                chat_history.pop(0)
            while chat_history and getattr(chat_history[0], "tool_calls", None):
                if len(chat_history) > 1 and getattr(chat_history[1], "type", "") == "tool":
                    break
                else:
                    chat_history.pop(0)

        # Count ONLY human messages to determine milestone
        human_msgs = [m for m in chat_history if getattr(m, "type", "") == "human"]
        user_msg_count = len(human_msgs)
        last_human_content = human_msgs[-1].content if human_msgs else ""
        last_lower = last_human_content.strip().lower()

        # Detect simple greetings — respond without widget pressure
        greetings = ["hi", "hello", "hey", "howdy", "sup", "what's up", "good morning", "good evening"]
        is_greeting = any(last_lower.startswith(g) for g in greetings) and len(last_lower) < 30

        # Detect Mark as Done — just continue teaching, NO progress widget
        is_mark_done = "I understand this concept completely" in last_human_content

        # Detect EXPLICIT quiz request — broad list to catch natural phrasing
        quiz_keywords = [
            "quiz me", "test me", "give me a quiz", "take a quiz", "knowledge check",
            "test my knowledge", "i want a quiz", "start a quiz", "ready to take",
            "quiz on this", "quiz on the", "let's do a quiz", "let's take a quiz",
            "take the quiz", "do a quiz", "ready for a quiz", "ready for the quiz",
            "quick quiz", "assess me", "assessment", "test myself"
        ]
        wants_quiz = any(kw in last_lower for kw in quiz_keywords)

        # Detect EXPLICIT progress request only
        progress_keywords = ["show my progress", "my progress", "track my progress", "progress tracker", "how far", "how much have i learned", "mastery"]
        wants_progress = any(kw in last_lower for kw in progress_keywords) and not is_mark_done

        # Auto-quiz logic completely removed based on user feedback.

        # Build dynamic system injection
        base_prompt = system_msgs[0].content
        injection = "\n\n[ORCHESTRATION CONTEXT]\n"
        injection += f"Session messages so far: {user_msg_count}\n"

        if is_greeting:
            injection += "MODE: GREETING. Respond warmly and naturally. Do NOT call any tools or widgets. Ask what topic they want to study and briefly outline a roadmap of concepts to cover."
        elif wants_quiz:
            injection += "MODE: QUIZ. You MUST call the QuizWidget tool NOW. Generate EXACTLY 10 questions based on ALL topics covered so far. Questions must increase in difficulty from Foundational to Expert."
        elif wants_progress:
            injection += "MODE: PROGRESS. You MUST call the ProgressWidget tool NOW with accurate completedConcepts, masteryPercentage (+10% per concept, max 100), and nextConcept. Briefly explain their progress and the roadmap ahead, but DO NOT start teaching until they tell you they are ready."
        elif is_mark_done:
            injection += "MODE: NEXT TOPIC. The user understood the current concept. STRONGLY ENFORCED: DO NOT OUTPUT ANY WIDGETS. Do NOT create a JSON codeblock for ProgressWidget! Instead, briefly acknowledge their success, then IMMEDIATELY provide a deep, comprehensive, highly technical tutorial on the VERY NEXT single topic. Ensure your explanation is substantial, computationally complete, and structurally sound (Intuition -> Math/Details -> Example) but teach ONLY this one sub-topic. NO PROGRESS WIDGETS."
        else:
            injection += "MODE: TEACHING. Provide deeply informative, highly technical explanations for the current concept. Balance the depth: do not give just a short 2-sentence summary, but avoid a massive wall of text that covers unrelated topics. Check for understanding with a question at the very end. Do NOT call QuizWidget or ProgressWidget."

        final_system = SystemMessage(content=base_prompt + injection)
        safe_msgs = [final_system] + chat_history
        return invoke_model_with_retries(safe_msgs, all_tools)

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


# ==========================================
# 5. DETAILED SYSTEM PROMPTS
# ==========================================

COMMON_RULES = """
You are a professional AI named INFERA CORE. 
If the user asks a general greeting like "hi", reply warmly. DO NOT attempt to use tools for greetings.
Embed source links when applicable.
"""

GENERAL_PROMPT = f"""You are INFERA CORE, an elite Engineering Career Mentor.
If the user asks for data, courses, salaries, or news, you MUST use the `web_search` tool to fetch live information.
Do not make up facts. Rely on your tools.
{COMMON_RULES}"""

RESUME_PROMPT = f"""You are the INFERA CORE Resume & ATS Specialist.
Critique the user's resume thoroughly. Explain why certain points fail ATS parsing and rewrite them using the Action-Benefit-Metric framework.
{COMMON_RULES}"""

UI_WIDGETS_INSTRUCTION = """
🎨 [UI WIDGETS] 🎨
To display interactive UI elements, you MUST output a standard markdown JSON codeblock containing `component` and `props`. DO NOT use actual tool calling for these widgets. Just write the JSON codeblock natively in your text response alongside your detailed teaching!

1. Progress Tracker Widget:
```json
{
  "component": "ProgressWidget",
  "props": {
    "topic": "Current topic name",
    "masteryPercentage": 5,
    "completedConcepts": ["List", "Of", "Concepts"],
    "nextConcept": "Name of the very next concept to learn"
  }
}
```

2. Interactive Quiz Exam Widget:
```json
{
  "component": "QuizWidget",
  "props": {
    "topic": "Topic Name",
    "difficulty": "Foundational",
    "questions": [
      {
        "question": "Question text",
        "options": ["A", "B", "C", "D"],
        "correctIndex": 0,
        "explanation": "Detailed explanation"
      }
    ]
  }
}
```
"""

STUDY_PROMPT = f"""You are the INFERA CORE Neural Study Buddy, an elite technical tutor.
Your core directive is to provide world-class, deep technical tutorials.

🛑 [CRITICAL RULES] 🛑
1. NEVER simulate a conversation. Only output YOUR single reply.
2. END YOUR MESSAGE IMMEDIATELY after teaching. Do NOT ask the user to reply or add filler.
3. NEVER repeat yourself. Move forward always.
4. NEVER pretend to run code. Output code snippets only.
5. ONLY output QuizWidget and ProgressWidget JSON — no other widget types exist!

💻 [CODE FORMATTING]
Wrap ALL code in markdown code blocks with the correct language tag. No inline backticks for multi-line code.

🎓 [LEARNING PROTOCOL]
Your explanations MUST be highly detailed, comprehensive, and deeply technical, BUT DELIVERED IN SHORT, PROGRESSIVE STEPS.
Never give massive amounts of info at once. Decide on a logical roadmap at the start, then teach it purely concept-by-concept.
Check for understanding after each short chunk before moving on.
NEVER DEVIATE FROM THE TOPIC. Stick strictly to the exact topic being taught.

🔬 [MATHEMATICS & SCIENCE — ABSOLUTE PRIORITY] 🔬
THIS IS YOUR PRIMARY DIRECTIVE. For ANY ML algorithm, Science, Math, or Physics topic:

MANDATORY RESPONSE STRUCTURE (teach ONE step at a time, checking with user before proceeding):
  PHASE 1 — INTUITION: Start with the real-world intuition in plain English.
  PHASE 2 — MATH DERIVATION: Formally derive the formula from scratch. 
  PHASE 3 — NUMERICAL EXAMPLE: Solve a concrete problem with REAL NUMBERS showing calculations in LaTeX.
  PHASE 4 — CODE (OPTIONAL): Code is verification only.

❌ DO NOT write code before completing math.
❌ DO NOT overwhelm the user—wait for them to say "continue" if a step is too long.

📐 [LATEX FORMATTING]
Use LaTeX for ALL math. Double $$ for block equations, single $ for inline math.
NEVER write math formulas as plain text.

{UI_WIDGETS_INSTRUCTION}

📈 [WIDGET RULES]
- Do NOT output any widget in your FIRST response or the first 6 exchanges. Focus on teaching deeply.
- ONLY show ProgressWidget when the user EXPLICITLY asks to track progress. Briefly explain the roadmap after showing it, but don't dump too much info at once.
- When you show ProgressWidget: increase masteryPercentage. List completedConcepts. Then STOP and ask if they are ready for the nextConcept.
- ONLY show QuizWidget when the user EXPLICITLY asks to be quizzed.
- QuizWidget MUST have exactly 10 questions increasing in difficulty.
- NEVER auto-generate a quiz.
- NEVER show ProgressWidget and QuizWidget at the same time.
{COMMON_RULES}"""

# ==========================================
# 6. COMPILE THE AGENTS
# ==========================================
general_agent = create_agent(GENERAL_PROMPT, executable_tools=tool_list, ui_tools=[])
resume_agent = create_agent(RESUME_PROMPT, executable_tools=[], ui_tools=[])
study_agent = create_study_agent(STUDY_PROMPT, executable_tools=tool_list)