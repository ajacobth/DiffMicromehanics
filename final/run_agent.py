"""
DiffMicromechanics conversational agent.

Run from final/:
    python run_agent.py
    python run_agent.py --card 3        # resume an existing card by ID
    python run_agent.py --card "Carbon" # resume by name (partial match)
"""

import argparse
from pathlib import Path
from typing import Annotated, Optional, TypedDict

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from agent.state import AgentState

# ── Paths ──────────────────────────────────────────────────────────────────────

PROMPTS_DIR = Path(__file__).parent / "agent" / "prompts"

# ── Prompt loading ─────────────────────────────────────────────────────────────

def load_system_prompt() -> str:
    """Combine system_prompt.md and vocabulary.md into one system message."""
    system = (PROMPTS_DIR / "system_prompt.md").read_text()
    vocab  = (PROMPTS_DIR / "vocabulary.md").read_text()
    return f"{system}\n\n---\n\n{vocab}"

# ── LLM ───────────────────────────────────────────────────────────────────────

llm = ChatOllama(
    model="qwen2.5:7b-instruct",
    temperature=0,
    streaming=True,
)

# ── Graph ─────────────────────────────────────────────────────────────────────
# agent_tools.py and graph.py are not yet built.
# Until they are, this builds a minimal graph: LLM only, no tools.
# When graph.py is ready, replace everything between the dashes with:
#     from agent.graph import build_app
#     app = build_app(llm, SYSTEM_PROMPT)

SYSTEM_PROMPT = load_system_prompt()

def agent_node(state: AgentState):
    context = (
        f"\nCurrent material card: {state.get('current_card_id') or 'none loaded'}"
        f"\nCompleted stages: {state.get('completed_stages') or 'none'}"
    )
    messages = [SystemMessage(SYSTEM_PROMPT + context)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.set_entry_point("agent")
graph.add_conditional_edges(
    "agent",
    lambda s: END,          # no tools yet — always end after agent responds
)
app = graph.compile(checkpointer=MemorySaver())

# ── Streaming ─────────────────────────────────────────────────────────────────

def run_turn(user_input: str, config: dict) -> None:
    """Run one conversation turn and print the response."""
    result = app.invoke(
        {"messages": [HumanMessage(user_input)]},
        config,
    )
    response = result["messages"][-1].content
    print(f"\nAgent: {response}\n")

# ── Session helpers ────────────────────────────────────────────────────────────
# These will move to agent/session.py once that file is built.

def resolve_card(identifier: Optional[str]) -> Optional[int]:
    """Resolve a card name or ID string to a card_id integer."""
    if identifier is None:
        return None
    if identifier.isdigit():
        return int(identifier)
    # Name-based lookup — placeholder until session.py is built
    # from agent.session import resolve_card_from_message
    # return resolve_card_from_message(identifier)
    print(f"[name-based card lookup not yet implemented — use --card <id>]")
    return None

def restore_state(card_id: Optional[int]) -> dict:
    """Build initial state, restoring completed_stages from DB if card known."""
    if card_id is None:
        return {
            "messages":        [],
            "current_card_id": None,
            "completed_stages": [],
        }
    # Full restore from DB — placeholder until session.py is built
    # from agent.session import restore_state_from_card
    # return restore_state_from_card(card_id)
    return {
        "messages":         [],
        "current_card_id":  card_id,
        "completed_stages": [],
    }

# ── Main conversation loop ────────────────────────────────────────────────────

def chat_loop(card_id: Optional[int]) -> None:
    state  = restore_state(card_id)
    thread = f"card_{card_id}" if card_id else "new_session"
    config = {"configurable": {"thread_id": thread}}

    print("\nDiffMicromechanics Agent")
    print(f"Card: {card_id or 'none'} | Stages: {state['completed_stages'] or 'none'}")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break

        run_turn(user_input, config)

# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DiffMicromechanics agent")
    parser.add_argument(
        "--card", default=None,
        help="Card ID or name to resume (e.g. --card 3 or --card 'Carbon')"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args    = parse_args()
    card_id = resolve_card(args.card)
    chat_loop(card_id)
