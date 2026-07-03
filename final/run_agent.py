"""
DiffMicromechanics conversational agent.

Run from final/:
    python run_agent.py                         # default: qwen2.5 14B
    python run_agent.py --model 32b             # qwen2.5 32B (needs ~20GB RAM)
    python run_agent.py --model qwen2.5:14b-instruct-q4_K_M  # explicit Ollama tag
    python run_agent.py --card 3                # resume an existing card by ID
    python run_agent.py --card "Carbon"         # resume by name (partial match)
    python run_agent.py --model 32b --card 3   # combine flags
"""

import argparse
from pathlib import Path
from typing import Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from agent.state import AgentState
from agent.graph import build_app

# ── Paths ──────────────────────────────────────────────────────────────────────

PROMPTS_DIR = Path(__file__).parent / "agent" / "prompts"

# ── Model registry ─────────────────────────────────────────────────────────────

_MODEL_ALIASES = {
    "14b": "qwen2.5:14b-instruct-q4_K_M",
    "32b": "qwen2.5:32b-instruct-q4_K_M",
}

def resolve_model(alias: str) -> str:
    """Resolve a short alias (14b, 32b) or pass through a full Ollama model tag."""
    return _MODEL_ALIASES.get(alias.lower(), alias)

# ── Prompt loading ─────────────────────────────────────────────────────────────

def load_system_prompt() -> str:
    """Combine system_prompt.md and vocabulary.md into one system message."""
    system = (PROMPTS_DIR / "system_prompt.md").read_text()
    vocab  = (PROMPTS_DIR / "vocabulary.md").read_text()
    return f"{system}\n\n---\n\n{vocab}"

# ── LLM + Graph (built after args are parsed) ─────────────────────────────────

def build_agent(model_tag: str):
    llm = ChatOllama(
        model=model_tag,
        temperature=0,
        streaming=True,
    )
    return build_app(llm, load_system_prompt())

# ── Streaming ─────────────────────────────────────────────────────────────────

def run_turn(app, user_input: str, config: dict) -> None:
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

def chat_loop(app, card_id: Optional[int], model_tag: str) -> None:
    state  = restore_state(card_id)
    thread = f"card_{card_id}" if card_id else "new_session"
    config = {"configurable": {"thread_id": thread}}

    print("\nDiffMicromechanics Agent")
    print(f"Model: {model_tag} | Card: {card_id or 'none'} | Stages: {state['completed_stages'] or 'none'}")
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

        run_turn(app, user_input, config)

# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DiffMicromechanics agent")
    parser.add_argument(
        "--model", default="14b",
        help="Model alias (14b, 32b) or full Ollama tag (default: 14b)"
    )
    parser.add_argument(
        "--card", default=None,
        help="Card ID or name to resume (e.g. --card 3 or --card 'Carbon')"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args      = parse_args()
    model_tag = resolve_model(args.model)
    card_id   = resolve_card(args.card)
    app       = build_agent(model_tag)
    chat_loop(app, card_id, model_tag)
