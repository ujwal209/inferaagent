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
                    model="llama-3.1-8b-instant",
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
            
            # Clean up orphaned tool calls to prevent 400 API errors
            while chat_history and getattr(chat_history[0], "type", "") == "tool":
                chat_history.pop(0)
            while chat_history and getattr(chat_history[0], "tool_calls", None):
                if len(chat_history) > 1 and getattr(chat_history[1], "type", "") == "tool":
                    break 
                else:
                    chat_history.pop(0) 

        # Calculate exact number of user prompts
        user_msg_count = len([m for m in chat_history if getattr(m, "type", "") == "user"])

        # Dynamically inject prompt enforcement for the Study Agent
        if system_msgs and "ProgressWidget" in system_msgs[0].content:
            can_show_widgets = (user_msg_count > 0 and user_msg_count % 7 == 0)
            enforcement = f"\n\n[SYSTEM PROTOCOL INJECTION]\nCurrent session length: {user_msg_count} user messages.\n"
            if can_show_widgets:
                enforcement += "STATUS: MILESTONE REACHED. You MUST output the ProgressWidget (and/or QuizWidget) to track progress in this exact response."
            else:
                enforcement += "STATUS: GRINDING. You are ABSOLUTELY BANNED from outputting ANY ProgressWidget or QuizWidget JSON blocks in this response. Proceed solely with educational deep-dives."
            
            system_msgs = [SystemMessage(content=system_msgs[0].content + enforcement)]
                
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

🛑 [CRITICAL ANTI-HALLUCINATION PROTOCOL] 🛑
1. NEVER simulate a conversation. Only output YOUR single reply.
2. END YOUR MESSAGE IMMEDIATELY after teaching or asking a question. Do NOT add filler urging the user to reply.
3. NEVER repeat yourself. Move forward.
4. YOU MUST NEVER PRETEND TO RUN CODE. Output code snippets only.

💻 [STRICT CODE FORMATTING] 💻
Wrap ALL code in markdown code blocks with the correct language tag. No inline backticks for multi-line code.

🎓 [LEARNING EXPERIENCE PROTOCOL] 🎓
Your explanations MUST be highly detailed, comprehensive, and deeply technical.
Naturally progress from basic to advanced. Never give short or skipped explanations.
🔥 PRIORITIZE EXPLANATION OVER CODE. Only provide code when it directly demonstrates the concept.
🔥 NEVER DEVIATE FROM THE TOPIC. Stick strictly to the exact topic being taught. Never jump to unrelated areas.

🔬 [MATHEMATICS & SCIENCE DEEP-DIVE PROTOCOL] 🔬
For ANY Machine Learning algorithm, Physics, Chemistry, or Math topic, you are REQUIRED to:
- Explain the deep MATHEMATICAL INTUITION from scratch, not just theory.
- Derive the formulas and explain WHY each term exists.
- Solve a concrete numerical worked example STEP BY STEP with actual numbers.
- Show intermediate calculation steps using LaTeX.
Do NOT skip the math. If you skip the derivation or worked example, you have FAILED your task.

📐 [STRICT LATEX FORMATTING] 📐
Use strict LaTeX for ALL math. Double $$ for block equations, single $ for inline math.

{UI_WIDGETS_INSTRUCTION}

🚫🚫🚫 [IRON-CLAD WIDGET SUPPRESSION RULE] 🚫🚫🚫
You are ABSOLUTELY FORBIDDEN from outputting ANY ProgressWidget or QuizWidget JSON blocks unless BOTH of the following conditions are true:
  CONDITION 1: The conversation has had AT LEAST 7 user messages (count them in the chat history).
  CONDITION 2: The user has clicked "Mark as Done" (their message literally contains "I understand this concept completely") OR the user has explicitly asked for a quiz.

If EITHER condition is not met, you MUST NOT output any widget JSON. Teach deeply instead.

📈 [PROGRESS RULES - ONLY when widgets are allowed] 📈
When you are ALLOWED to output ProgressWidget (both conditions met above):
- Increase masteryPercentage by a MAXIMUM of 10% per concept. NEVER jump by more than 10%.
- masteryPercentage must never exceed 100.
- Always include an accurate list of completedConcepts and nextConcept.
- After outputting the ProgressWidget, immediately give a full in-depth tutorial of the next concept.

🧠 [QUIZ RULES - ONLY when widgets are allowed] 🧠
- Only output QuizWidget when BOTH conditions above are met AND the user explicitly says "quiz me" or presses Mark as Done at a 25%, 50%, 75%, or 100% milestone.
- Quiz MUST contain exactly 10 questions that gradually increase in difficulty.
- After the quiz is shown, wait for the user to answer before continuing.
{COMMON_RULES}"""

# ==========================================
# 6. COMPILE THE AGENTS
# ==========================================
general_agent = create_agent(GENERAL_PROMPT, executable_tools=tool_list, ui_tools=[])
resume_agent = create_agent(RESUME_PROMPT, executable_tools=[], ui_tools=[])
study_agent = create_agent(STUDY_PROMPT, executable_tools=tool_list, ui_tools=[])