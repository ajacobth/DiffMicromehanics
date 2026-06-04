"""
Intent router for the DiffMicromechanics agent.

Two-stage classification:
  1. Keyword short-circuit — deterministic, no LLM call
  2. LLM classifier fallback — for ambiguous messages

Valid intents: forward | inverse | knowledge | out_of_scope
"""

from typing import Literal
from pydantic import BaseModel
from langchain_core.messages import SystemMessage
from langchain_core.language_models import BaseChatModel


# ── Keyword short-circuit ─────────────────────────────────────────────────────
# Checked in order — first match wins. Keys are lowercase substrings.

KEYWORD_MAP = [
    # greetings / meta — route to knowledge so agent can respond naturally
    ("hello",   "knowledge"),
    ("hi ",     "knowledge"),
    ("hey",     "knowledge"),
    ("thanks",  "knowledge"),
    ("thank you","knowledge"),
    ("help",    "knowledge"),
    # card queries — forward agent has list_cards
    ("what cards",  "forward"),
    ("list cards",  "forward"),
    ("show cards",  "forward"),
    ("my cards",    "forward"),
    # forward
    ("predict",      "forward"),
    ("forward model","forward"),
    ("run forward",  "forward"),
    # inverse
    ("invert",       "inverse"),
    ("inverse",      "inverse"),
    ("characterize", "inverse"),
    ("characterise", "inverse"),
    ("infer",        "inverse"),
    ("elastic inverse",    "inverse"),
    ("thermoelastic",      "inverse"),
    ("thermal inverse",    "inverse"),
    ("i have measurements","inverse"),
    ("i have data",        "inverse"),
    # knowledge
    ("list materials",  "knowledge"),
    ("what materials",  "knowledge"),
    ("add fiber",       "knowledge"),
    ("add polymer",     "knowledge"),
    ("what is the modulus",   "knowledge"),
    ("what is the cte",       "knowledge"),
    ("what is the conductivity", "knowledge"),
    ("tell me about",   "knowledge"),
]


# ── LLM classifier ────────────────────────────────────────────────────────────

ROUTER_PROMPT = """
Classify the user's message into exactly one intent:

- forward       : user wants to PREDICT composite properties given microstructure or a card
- inverse       : user wants to INFER/CHARACTERIZE material properties from measurements
- knowledge     : user asks about material properties, theory, or wants to add/look up materials
- out_of_scope  : question is not about composite micromechanics

Reply with the intent label only. No explanation.
""".strip()


class Intent(BaseModel):
    intent: Literal["forward", "inverse", "knowledge", "out_of_scope"]


# ── Router node factory ───────────────────────────────────────────────────────

def build_router_node(llm: BaseChatModel):
    classifier = llm.with_structured_output(Intent)

    def router_node(state):
        last = state["messages"][-1]
        text = last.content.lower() if hasattr(last, "content") else ""

        # Stage 1 — keyword short-circuit (no LLM call)
        for keyword, intent in KEYWORD_MAP:
            if keyword in text:
                print(f"[ROUTER] keyword match '{keyword}' → {intent}")
                return {"intent": intent}

        # Stage 2 — LLM classifier
        try:
            result = classifier.invoke([SystemMessage(ROUTER_PROMPT), last])
            print(f"[ROUTER] LLM classified → {result.intent}")
            return {"intent": result.intent}
        except Exception:
            print("[ROUTER] classifier failed → fallback: forward")
            return {"intent": "forward"}   # safe fallback

    return router_node
