"""
LangGraph application for the DiffMicromechanics agent.

Graph structure:
    agent_node  →  (has tool calls?) → YES → tool_executor → checker → agent_node
                                      → NO  → END

    checker appends a PASS/FAIL quality report to inverse solver results
    before the agent reads them. Non-inverse tool results pass through untouched.

Call build_app() to get a compiled graph ready for invoke().
"""

from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from agent.state import AgentState
from agent.agent_tools import TOOLS
import core.services.service_checklist as _sc

QUALITY_CHECKED_TOOLS = {
    "run_elastic_inverse",
    "run_thermoelastic_inverse",
    "run_thermal_inverse",
    "run_full_pipeline",
}


def build_app(llm: BaseChatModel, system_prompt: str):
    """
    Build and compile the agent graph.

    Args:
        llm:           ChatOllama (or any BaseChatModel) instance.
        system_prompt: Full system prompt string injected at the start of every turn.

    Returns:
        Compiled LangGraph app with MemorySaver checkpointer.
    """
    llm_with_tools = llm.bind_tools(TOOLS)

    # ── Nodes ─────────────────────────────────────────────────────────────────

    def agent_node(state: AgentState):
        context = (
            f"\nCurrent material card: {state.get('current_card_id') or 'none loaded'}"
            f"\nCompleted stages: {state.get('completed_stages') or 'none'}"
        )
        messages = [SystemMessage(system_prompt + context)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    tool_executor = ToolNode(TOOLS)

    def checker_node(state: AgentState) -> dict:
        """Append a quality report to inverse solver results before the agent reads them."""
        last = state["messages"][-1]
        if not isinstance(last, ToolMessage) or last.name not in QUALITY_CHECKED_TOOLS:
            return {}
        report = _sc.evaluate(last.name, last.content)
        updated = ToolMessage(
            content=last.content + report,
            tool_call_id=last.tool_call_id,
            name=last.name,
        )
        return {"messages": [updated]}

    # ── Routing ───────────────────────────────────────────────────────────────

    def should_use_tool(state: AgentState) -> str:
        """Route to tool_executor if the last message has tool calls, else end."""
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tool_executor"
        return END

    # ── Graph ─────────────────────────────────────────────────────────────────

    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tool_executor", tool_executor)
    graph.add_node("checker", checker_node)

    graph.set_entry_point("agent")

    graph.add_conditional_edges("agent", should_use_tool)
    graph.add_edge("tool_executor", "checker")
    graph.add_edge("checker", "agent")

    return graph.compile(checkpointer=MemorySaver())
