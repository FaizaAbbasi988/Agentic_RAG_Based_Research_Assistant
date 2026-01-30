from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain.messages import HumanMessage, AIMessage, SystemMessage, RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.runnables import RunnableConfig
from langchain.tools import tool 
from langgraph.types import Send
from datasets import load_dataset
from pymilvus import settings
from tavily import TavilyClient
from IPython.display import display, Image

from src.config.custom_state import State, AgentState
from src.config.QueryAnalysis import QueryAnalysis
from src.config.selector_response import selector_response
from src.config.settings import settings

from src.prompts.get_aggregation_prompt import get_aggregation_prompt
from src.prompts.get_query_analysis_prompt import get_query_analysis_prompt
from src.prompts.get_rag_agent_prompt import get_rag_agent_prompt
from src.prompts.respond_or_select_tools_prompt import respond_or_select_tools_prompt
from src.prompts.safety_guardrail_prompt import safety_guardrail_prompt
from src.prompts.summarize_prompt import summarize_prompt
from src.prompts.WEB_GENERATE_PROMPT import WEB_GENERATE_PROMPT
from src.prompts.WEB_SEARCH_PROMPT import WEB_SEARCH_PROMPT 

from src.tools.milvus_retrieval_tool import milvus_retrieval
from src.tools.web_search_tool import web_search_tool

checkpointer = InMemorySaver()

class ResearchAssistantRAG():
    def __init__(self, checkpointer = checkpointer):
        self.model = ChatGoogleGenerativeAI(model=settings.generation_model, api_key = settings.google_api_key)
        self.llm_with_tools = self.model.bind_tools([milvus_retrieval])
        self.checkpointer = checkpointer
        self.builder = self.assemble_nodes()

    def invoke(self, query, user_id, on_update=None):
        config = {"configurable": {"thread_id" : user_id, "on_update": on_update}}
        ans = self.builder.invoke({"messages" : [{"role" : "user", "content" : query}]}, config = config)
        return self._process_content(ans['messages'][-1].content)

    def _send_update(self, config, message: str):
        if not config:
            return
        on_update = config.get("configurable", {}).get("on_update")
        if on_update:
            on_update(message)


    def safety_guardrail_node(self, state: State, config: RunnableConfig):
        self._send_update(config, "Analyzing query by passing through safety guardrails")
        user_input = state['messages'][-1].content
        check = self.model.invoke(safety_guardrail_prompt().format(user_input = user_input))
        decision = check.content.strip().lower()
        if "unsafe" in decision:
            return {"guardrail_check" : "unsafe"}
        return {"guardrail_check" : "safe"}

    def guadraill_router(self, state: State):
        guadrail_route = state.get("guardrail_check")
        return guadrail_route

    def safety_block_node(self, state: State):
        # This node provides the graceful exit message
        return {"messages": [AIMessage(content="I am a specialized Research AI Assistant. I can only assist with research-related inquiries. Please let me know if you have a relevant question.")]}

    def selector(self, state: State, config : RunnableConfig):
        self._send_update(config, "Selecting the appropriate route to process the query")
        questions = state["messages"][-1].content
        selector = self.model.with_config(temperature = 0.2).with_structured_output(selector_response)
        result = selector.invoke(respond_or_select_tools_prompt(question = questions))
        selector = result.selection_category.strip().upper()
        return {"selector" : selector}

    def direct_answer_node(self, state: State, config : RunnableConfig):
        self._send_update(config, "Preparing to answer your query")
        question = state["messages"][-1].content

        response = self.model.invoke([
            {"role": "system", "content": "Answer concisely and accurately."},
            {"role": "user", "content": question}
        ])

        return {"messages": [response]}

    def web_search_node(self, state: State, config : RunnableConfig):
        self._send_update(config, "Searching on the web for the query")
        question = state["messages"][-1].content

        prompt = WEB_SEARCH_PROMPT.format(question=question)
        web_query = self.model.invoke(
            [{"role": "user", "content": prompt}]
        ).content

        web_content = web_search_tool.invoke(web_query)

        return {
            "context": web_content,          # ← store retrieval
            "user_question": question,
            "next_step": "generate_answer"
        }

    def generate_answer_node(self, state: State, config : RunnableConfig):
        self._send_update(config, "generating your answer")
        question = state.get("user_question")
        context = state.get("context")

        prompt = WEB_GENERATE_PROMPT.format(
            question=question,
            context=context
        )

        response = self.model.invoke(
            [{"role": "user", "content": prompt}]
        )

        return {
            "messages": [response]
        }

    def analyze_chat_and_summarize_node(self, state: State, config : RunnableConfig):
        "analyzes the chat history and summarizes key points for context"
        self._send_update(config, "Summarizing the previous chats")
        if len(state['messages']) < 4:
            return {"conversation_summary" : ""}
        
        relevant_msgs = [
            msg for msg in state['messages'][:-1]
            if isinstance(msg , (HumanMessage, AIMessage))
            and not getattr(msg, "tool_calls", None)
        ]

        if not relevant_msgs:
            return {"conversation_summary": ""}
        
        conversation = "Conversation_history :"
        for msg in relevant_msgs[-6:]:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            conversation += f" {role} : {msg.content}\n"

        summary_response = self.model.with_config(temperature = 0.2).invoke([SystemMessage(content = summarize_prompt())] + [HumanMessage(content = conversation)])

        return {"conversation_summary": summary_response.content, "agent_answers": [{"__reset__": True}]}

    def analyze_and_rewrite_query(self, state: State, config : RunnableConfig):
        """
        Analyzes user query and rewrites it for clarity, optionally using conversation context.
        """
        self._send_update(config, "Rewriting the query into smaller subqueries")
        last_message = state["messages"][-1]
        conversation_summary = state.get("conversation_summary", "")

        context_section = (f"Conversation Context:\n{conversation_summary}\n" if conversation_summary.strip() else "") + f"User Query:\n{last_message.content}\n"

        llm_with_structure = self.model.with_config(temperature=0.1).with_structured_output(QueryAnalysis)
        response = llm_with_structure.invoke([SystemMessage(content=get_query_analysis_prompt())] + [HumanMessage(content=context_section)])

        if len(response.questions) > 0 and response.is_clear:
            # Remove all non-system messages
            delete_all = [
                RemoveMessage(id=m.id)
                for m in state["messages"]
                if not isinstance(m, SystemMessage)
            ]
            return {
                "question_isClear": True,
                "messages": delete_all,
                "originalQuery": last_message.content,
                "rewrittenQuestions": response.questions
            }
        else:
            clarification = response.clarification_needed if (response.clarification_needed and len(response.clarification_needed.strip()) > 10) else "I need more information to understand your question."
            return {
                "question_isClear": False,
                "messages": [AIMessage(content=clarification)]
            }

    def human_input_node(self, state: State):
        """Placeholder node for human-in-the-loop interruption"""
        return {}

    def route_after_rewrite(self, state: State):
        """Route to agent if question is clear, otherwise wait for human input"""
        if not state.get("question_isClear", False):
            return "human_input"
        else:
            return [
                    Send("process_question", {"question": query, "question_index": idx, "messages": []})
                    for idx, query in enumerate(state["rewrittenQuestions"])
                ]

    def agent_node(self, state: AgentState, config : RunnableConfig):
        "Main agent node that processes queries using tools"
        # Use state["question"] which is passed via Send()
        self._send_update(config, f"Processing sub-query: {state['question']}") 
        
        sys_msg = SystemMessage(content = get_rag_agent_prompt())
        if not state.get("messages"):
            human_msg = HumanMessage(content = state["question"])
            response = self.llm_with_tools.invoke([sys_msg]+[human_msg])
            return {"messages": [human_msg, response]}
        return {"messages": [self.llm_with_tools.invoke([sys_msg] + state["messages"])]}


    def extract_final_answer(self, state: AgentState):
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                res = {
                    "final_answer": msg.content,
                    "agent_answers": [{
                        "index": state["question_index"],
                        "question": state["question"],
                        "answer": msg.content
                    }]
                }
                return res
        return {
            "final_answer": "Unable to generate an answer.",
            "agent_answers": [{
                "index": state["question_index"],
                "question": state["question"],
                "answer": "Unable to generate an answer."
            }]
        }

    def aggregate_responses(self, state: State, config : RunnableConfig):
        # 🔹 CASE 1: Direct answer → return last AI message
        self._send_update(config, "Aggregating smaller responses to generate final response")
        if state.get("selector") == "DIRECT_ANSWER":
            for msg in reversed(state["messages"]):
                if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                    return {
                        "messages": [AIMessage(content=msg.content)],
                        "final_answer": msg.content
                    }

            return {
                "messages": [AIMessage(content="Unable to generate a direct answer.")],
                "final_answer": "Unable to generate a direct answer."
            }

        # 🔹 CASE 2: Tool-based answers → aggregate
        if not state.get("agent_answers"):
            return {
                "messages": [AIMessage(content="No answers were generated.")],
                "final_answer": "No answers were generated."
            }

        sorted_answers = sorted(
            state["agent_answers"],
            key=lambda x: x["index"]
        )

        formatted_answers = ""
        for i, ans in enumerate(sorted_answers, start=1):
            formatted_answers += f"\nAnswer {i}:\n{ans['answer']}\n"

        user_message = HumanMessage(
            content=f"""Retrieved answers:{formatted_answers}"""
        )

        synthesis_response = self.model.invoke([
            SystemMessage(content=get_aggregation_prompt()),
            user_message
        ])

        return {
            "messages": [AIMessage(content=synthesis_response.content)],
            "final_answer": synthesis_response.content
        }
    
    def assemble_nodes(self):
        """
        Orchestrates the creation of the subgraph (AgentState) 
        and the main graph (State).
        """
        
        # --- SUBGRAPH: AGENT BUILDER ---
        # This handles individual sub-queries in parallel
        agent_builder = StateGraph(AgentState)
        
        agent_builder.add_node("agent", self.agent_node)
        agent_builder.add_node("tools", ToolNode([milvus_retrieval]))
        agent_builder.add_node("extract_answer", self.extract_final_answer)

        # 1. Start at the agent node
        agent_builder.add_edge(START, "agent")

        # 2. Use tools_condition to decide:
        #    - If LLM makes a tool_call -> go to "tools"
        #    - If LLM provides text/finishes -> go to "extract_answer"
        agent_builder.add_conditional_edges(
            "agent",
            tools_condition,
            {
                "tools": "tools",
                "__end__": "extract_answer" 
            }
        )

        # 3. Loop back to agent after tool execution so it can read the results
        agent_builder.add_edge("tools", "agent")

        # 4. Final step of subgraph
        agent_builder.add_edge("extract_answer", END)

        # Compile the subgraph to be used as a node in the main graph
        agent_subgraph = agent_builder.compile()

        # --- MAIN GRAPH: WORKFLOW BUILDER ---
        graph_builder = StateGraph(State)

        # Add all orchestration nodes
        graph_builder.add_node("safety_guardrail", self.safety_guardrail_node)
        graph_builder.add_node("safety_block", self.safety_block_node)
        graph_builder.add_node("selector", self.selector)
        graph_builder.add_node("direct_answer", self.direct_answer_node)
        graph_builder.add_node("web_search", self.web_search_node)
        graph_builder.add_node("generate_web_answer", self.generate_answer_node)
        graph_builder.add_node("analyze_chat_and_summarize", self.analyze_chat_and_summarize_node)
        graph_builder.add_node("analyze_and_rewrite_query", self.analyze_and_rewrite_query)
        graph_builder.add_node("human_input", self.human_input_node)
        graph_builder.add_node("process_question", agent_subgraph)
        graph_builder.add_node("aggregate", self.aggregate_responses)

        # --- DEFINE EDGES & ROUTING ---
        graph_builder.add_edge(START, "safety_guardrail")

        # Guardrail routing
        graph_builder.add_conditional_edges(
            "safety_guardrail", 
            self.guadraill_router, 
            {"unsafe": "safety_block", "safe": "selector"}
        )
        graph_builder.add_edge("safety_block", END)

        # Main Selector routing
        graph_builder.add_conditional_edges(
            "selector", 
            lambda state: state['selector'], 
            {
                "DIRECT_ANSWER": "direct_answer", 
                "WEB_SEARCH": "web_search", 
                "ARXIV_SEARCH": "analyze_chat_and_summarize"
            }
        )

        # Web Search Path
        graph_builder.add_edge("web_search", "generate_web_answer")
        graph_builder.add_edge("generate_web_answer", END)

        # Arxiv/Research Path (Query Rewriting & Subgraph)
        graph_builder.add_edge("analyze_chat_and_summarize", "analyze_and_rewrite_query")
        graph_builder.add_conditional_edges(
            "analyze_and_rewrite_query", 
            self.route_after_rewrite,
            {
                "human_input": "human_input",
                "process_question": "process_question"
            }
        )
        graph_builder.add_edge("human_input", "analyze_and_rewrite_query")

        # Aggregation Path
        graph_builder.add_edge("direct_answer", "aggregate")
        graph_builder.add_edge("process_question", "aggregate")
        graph_builder.add_edge("aggregate", END)

        # Compile the final graph with memory for conversation history
        builder = graph_builder.compile(
            checkpointer=self.checkpointer,
            interrupt_before=["human_input"]
        )

        return builder

    def show_graph(self):
        display(Image(self.builder.get_graph().draw_mermaid_png()))


    def _process_content(self, ans):
        if isinstance(ans, str):
            return ans

        if isinstance(ans, list) and len(ans) > 0:
            return ans[-1].get('text', "")

