"""
Stack validation script — run before building the agent.
Checks that Ollama is reachable, tool calling works, and LangGraph
wires correctly with MemorySaver.

Run from final/:
    python test_agent_stack.py
"""

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, TypedDict



llm = ChatOllama(
    model="qwen2.5:7b-instruct",
    temperature=0,
)



def test_basic():
    print("Test 1: basic LLM reachability ... ", end="", flush=True)
    response = llm.invoke("Reply with the single word: ok")
    assert response.content, "empty response"
    print(f"PASS  (got: '{response.content.strip()}')")


@tool
def add_numbers(a: float, b: float) -> str:
    """Add two numbers and return the result."""
    sum = a+b
    return f"Result: {sum}"

@tool
def subtract_numbers(a: float, b: float) -> str:
    """Subtract two numbers and return the result."""
    res = a-b
    return f"Result: {res}"

def test_tool_calling():
    print("Test 2: tool calling ... ", end="", flush=True)
    bound    = llm.bind_tools([add_numbers])
    response = bound.invoke("What is 3.5 plus 7.2?")
    assert response.tool_calls, (
        "No tool call returned — model answered in prose instead of calling "
        "the tool. Try a larger model or check the Ollama version."
    )
    call = response.tool_calls[0]
    assert call["name"] == "add_numbers", f"wrong tool called: {call['name']}"
    assert "a" in call["args"] and "b" in call["args"], f"missing args: {call['args']}"
    print(f"PASS  (called {call['name']} with {call['args']})")



class State(TypedDict):
    messages: Annotated[list, add_messages]

tools        = [add_numbers, subtract_numbers]
tool_node    = ToolNode(tools)
bound_llm    = llm.bind_tools(tools)

def agent_node(state: State):
    response = bound_llm.invoke(state["messages"])
    return {"messages": [response]}

def route(state: State):
    return "tools" if state["messages"][-1].tool_calls else END

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", route)
graph.add_edge("tools", "agent")
app = graph.compile(checkpointer=MemorySaver())

def test_langgraph_loop():
    print("Test 3: LangGraph tool loop ... ", end="", flush=True)
    config = {"configurable": {"thread_id": "test"}}
    result = app.invoke(
        {"messages": [HumanMessage("WFind 10.0 plus 25.5? And subract 10 from the result")]},
        config,
    )
    last = result["messages"][-1].content
    assert "25.5" in last or "25" in last, f"unexpected final answer: '{last}'"
    print(f"PASS  (final answer: '{last.strip()}')")



if __name__ == "__main__":
    print("\nDiffMicromechanics agent stack validation\n")
    try:
        test_basic()
        test_tool_calling()
        test_langgraph_loop()
        print("\nAll tests passed. Stack is ready.\n")
    except AssertionError as e:
        print(f"FAIL\n\nReason: {e}\n")
    except Exception as e:
        print(f"ERROR\n\n{type(e).__name__}: {e}\n")
