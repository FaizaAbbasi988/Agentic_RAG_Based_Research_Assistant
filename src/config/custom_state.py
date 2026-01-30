from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing import List, Annotated, Literal

def accumulate_or_reset(existing: List[dict], new: List[dict])->List[dict]:
    if new and any(item.get('__reset__') for item in new):
        return []
    return existing + new

class State(MessagesState):
    "State for main agent graph"
    user_question : str = ""
    context : str = ""
    conversation_summary : str = ""
    guardrail_check : Literal["safe", "unsafe"]
    question_isClear : bool = False
    rewrittenQuestions : List[str] = []
    agent_answers : Annotated[List[dict], accumulate_or_reset] = []
    selector : Literal["DIRECT_ANSWER", "WEB_SEARCH", "ARXIV_SEARCH"]

class AgentState(MessagesState):
    question : str = ""
    question_index :int = 0
    final_answer : str = ""
    agent_answers : Annotated[List[dict], accumulate_or_reset] = []