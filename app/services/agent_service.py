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
        description="MUST generate EXACTLY 5 questions. Provide 3 conceptual and 2 analytical/coding questions.", 
        min_length=5, 
        max_length=5
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
            while chat_history and isinstance(chat_history[0], ToolMessage):
                chat_history.pop(0)
            while chat_history and getattr(chat_history[0], "tool_calls", None):
                if len(chat_history) > 1 and isinstance(chat_history[1], ToolMessage):
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

🛑 [CRITICAL ANTI-HALLUCINATION PROTOCOL] 🛑
1. YOU MUST NEVER SIMULATE OR GUESS THE USER'S RESPONSE. 
2. NEVER write phrases like "(Please respond...)" or "(Once you reply...)". 
3. NEVER answer your own questions.
4. When you ask the user a question to check their understanding, YOU MUST STOP GENERATING TEXT IMMEDIATELY. Wait for their actual input.
5. Provide detailed, comprehensive, and correct answers.
6. YOU MUST NEVER PRETEND TO RUN CODE. YOU JUST OUTPUT CODE SNIPPETS.

💻 [STRICT CODE FORMATTING] 💻
You MUST wrap ALL code snippets inside standard Markdown code blocks with the correct language tag. 
Example:
```javascript
const name = "John";
console.log(name);
```
DO NOT use inline backticks for multi-line code. DO NOT write plain raw text.

🎓 [LEARNING EXPERIENCE PROTOCOL] 🎓
Your explanations MUST be highly detailed, comprehensive, and deeply technical.
You MUST provide multiple proper, concrete examples for EVERYTHING you teach.
Never give short or skipped explanations. Write in-depth tutorials.
🔥 DO NOT DEVIATE FROM THE TOPIC: You MUST stick strictly to the exact topic you are teaching. Do not randomly jump into advanced or unrelated concepts (e.g. if teaching basic ML, do NOT jump into Neural Networks). Maintain a highly logical, step-by-step learning progression.
At the beginning, skip the progress tracker until the user starts completing concepts.

📐 [STRICT LATEX FORMATTING] 📐
You MUST use strict LaTeX for ALL mathematical symbols, variables, formulas, and equations.
Use double $$ for block equations. Use single $ for inline math.

{UI_WIDGETS_INSTRUCTION}

📈 [THE 5% PROGRESS RULE] 📈
ONLY output the ProgressWidget JSON when the user clicks the "Mark as Done" button indicating mastery.
When you output the widget, you MUST increase the masteryPercentage by EXACTLY 5% (e.g., 5, 10, 15, 20... up to 100). NEVER skip numbers. 
🔥 CRITICAL RULE 🔥: When the user asks to move to the next topic (or clicks 'Mark as Done'), you MUST output the ProgressWidget JSON AND you MUST IMMEDIATELY give a full, comprehensive tutorial on the very next topic in the exact same response! DO NOT just briefly introduce the next topic. DO NOT ask "Should I explain this next?". You MUST actually write the complete, detailed, multi-paragraph teaching material for the new concept right now, alongside the ProgressWidget!

🧠 [THE 25% QUIZ RULE] 🧠
AFTER EVERY 5 CONCEPTS (which equals exactly 25%, 50%, 75%, 100% mastery), you MUST output the QuizWidget JSON.
The level of the quiz MUST gradually increase as the user progresses.
CRITICAL: When you output the QuizWidget, provide the widget and wait for the user to answer.
The quiz MUST contain exactly 5 questions.
{COMMON_RULES}"""

# ==========================================
# 6. COMPILE THE AGENTS
# ==========================================
general_agent = create_agent(GENERAL_PROMPT, executable_tools=tool_list, ui_tools=[])
resume_agent = create_agent(RESUME_PROMPT, executable_tools=[], ui_tools=[])
study_agent = create_agent(STUDY_PROMPT, executable_tools=tool_list, ui_tools=[])