# Intent Routing — Implementation Plan

## Why

The current agent uses a single ReAct loop where one `agent_node` handles all user intents
(forward prediction, inverse stages, material lookup). When users mix intents in the same
message, the LLM disambiguates from the system prompt alone — fragile with local models.
A router node makes intent classification structural, and binding each agent to a focused
tool subset prevents hallucinated tool selection.

---

## Graph Structure

```
                    ┌─────────┐
          START ───▶│ router  │
                    └────┬────┘
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼              ▼
   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌────────┐
   │forward_agent │ │inverse_agent │ │knowledge_agnt│ │ reject │
   └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └───┬────┘
          │                │                 │             │
          └────────┬────────────────────────┘            END
                   ▼ (tool calls present?)
           ┌───────────────┐
           │ tool_executor │  ← shared, all tools registered
           └───────┬───────┘
                   │ (route back via state["intent"])
          ┌────────┴──────────────┐
          ▼              ▼        ▼
   forward_agent   inverse_agent  knowledge_agent
```

**No `card_agent`.** Card operations (`save_to_card`, `list_cards`, `get_card_status`) always
happen *within* a forward or inverse workflow — never in isolation. A separate card agent
would break mid-workflow by routing away from `inverse_agent` right when the user says
"save to card". Card tools are included directly in `FORWARD_TOOLS` and `INVERSE_TOOLS`.

---

## Intents

| Intent | When | Agent |
|---|---|---|
| `forward` | user wants to predict composite properties given microstructure or a card | `forward_agent` |
| `inverse` | user wants to infer/characterize material properties from measurements | `inverse_agent` |
| `knowledge` | user asks about material properties, theory, or wants to look up / add materials | `knowledge_agent` |
| `out_of_scope` | question not about composite micromechanics | `reject` → END |

---

## Files to change

| File | Change |
|---|---|
| `agent/state.py` | Add `intent: Optional[str]` field |
| `agent/agent_tools.py` | Add 3 tool group lists, keep `TOOLS` |
| `agent/router.py` | **New file** — Pydantic schema + classifier node |
| `agent/graph.py` | Full rewrite — 6 nodes, conditional routing |

`app_chat.py`, `rag.py`, `ingest.py`, all tool implementations — untouched.

---

## Step 1 — `agent/state.py`

```python
from typing import Annotated, Optional, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages:         Annotated[list[BaseMessage], add_messages]
    current_card_id:  Optional[int]
    completed_stages: list[str]
    intent:           Optional[str]  # set by router each turn
```

`intent` has no reducer — last-write-wins. Router sets it once per user turn; persists in
state so `after_tool` conditional edge knows which agent node to return to.

Valid values: `"forward"` | `"inverse"` | `"knowledge"` | `"out_of_scope"`

---

## Step 2 — `agent/agent_tools.py`

Add three named groups at the bottom of the file, after all `@tool` definitions.
Keep `TOOLS` as the deduplicated union — used only by the shared `ToolNode`.

```python
# ── Intent-scoped tool groups ─────────────────────────────────────────────────

FORWARD_TOOLS = [
    predict_properties,
    predict_thermal_conductivity,
    get_model_inputs_outputs,
    inspect_card_inputs,
    convert_fraction,
    list_cards,
    get_card_status,
    save_to_card,
    save_processing_conditions,
]

INVERSE_TOOLS = [
    run_elastic_inverse,
    run_thermoelastic_inverse,
    run_thermal_inverse,
    check_identifiability,
    convert_fraction,
    inspect_card_inputs,
    list_cards,
    get_card_status,
    save_to_card,
    save_processing_conditions,
]

KNOWLEDGE_TOOLS = [
    list_materials,
    get_material_details,
    add_fiber,
    add_polymer,
    search_knowledge_base,
]

# Full deduplicated union — registered on ToolNode only, not bound to any LLM
TOOLS = list({t.name: t for t in
    FORWARD_TOOLS + INVERSE_TOOLS + KNOWLEDGE_TOOLS
}.values())
```

---

## Step 3 — `agent/router.py` (new file)

```python
from typing import Literal
from pydantic import BaseModel
from langchain_core.messages import SystemMessage
from langchain_core.language_models import BaseChatModel

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


def build_router_node(llm: BaseChatModel):
    classifier = llm.with_structured_output(Intent)

    def router_node(state):
        last = state["messages"][-1]
        try:
            result = classifier.invoke([SystemMessage(ROUTER_PROMPT), last])
            return {"intent": result.intent}
        except Exception:
            return {"intent": "forward"}   # safe fallback if local model misbehaves

    return router_node
```

**Why `Literal` on the field:** Pydantic raises `ValidationError` if the LLM returns anything
outside the 4 labels (e.g. `"FORWARD"`, `"prediction"`). The `except` block catches this and
defaults to `"forward"` so the graph never crashes.

---

## Step 4 — `agent/graph.py` (full rewrite)

```python
from langchain_core.messages import SystemMessage
from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from agent.state import AgentState
from agent.router import build_router_node
from agent.agent_tools import TOOLS, FORWARD_TOOLS, INVERSE_TOOLS, KNOWLEDGE_TOOLS


def build_app(llm: BaseChatModel, system_prompt: str):

    # Each intent gets its own LLM bound to a focused tool subset
    llm_forward   = llm.bind_tools(FORWARD_TOOLS)
    llm_inverse   = llm.bind_tools(INVERSE_TOOLS)
    llm_knowledge = llm.bind_tools(KNOWLEDGE_TOOLS)

    def _context(state):
        return (
            f"\nCurrent card: {state.get('current_card_id') or 'none'}"
            f"\nCompleted stages: {state.get('completed_stages') or 'none'}"
        )

    def make_agent(bound_llm):
        def node(state):
            messages = [SystemMessage(system_prompt + _context(state))] + state["messages"]
            return {"messages": [bound_llm.invoke(messages)]}
        return node

    def reject_node(state):
        return {"messages": [{"role": "assistant", "content":
            "I'm only able to help with composite micromechanics and material characterization topics."}]}

    # Shared executor — registered with ALL tools, executes whatever the agent called
    tool_executor = ToolNode(TOOLS)

    # ── Routing functions ─────────────────────────────────────────────────────

    def route_intent(state) -> str:
        return {
            "forward":      "forward_agent",
            "inverse":      "inverse_agent",
            "knowledge":    "knowledge_agent",
            "out_of_scope": "reject",
        }.get(state.get("intent"), "forward_agent")

    def should_use_tool(state) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tool_executor"
        return END

    def after_tool(state) -> str:
        return {
            "forward":   "forward_agent",
            "inverse":   "inverse_agent",
            "knowledge": "knowledge_agent",
        }.get(state.get("intent"), "forward_agent")

    # ── Build graph ───────────────────────────────────────────────────────────

    graph = StateGraph(AgentState)

    graph.add_node("router",          build_router_node(llm))
    graph.add_node("forward_agent",   make_agent(llm_forward))
    graph.add_node("inverse_agent",   make_agent(llm_inverse))
    graph.add_node("knowledge_agent", make_agent(llm_knowledge))
    graph.add_node("reject",          reject_node)
    graph.add_node("tool_executor",   tool_executor)

    graph.set_entry_point("router")

    # router → one of 4 branches
    graph.add_conditional_edges("router", route_intent)

    # each agent → tool_executor or END
    for agent in ["forward_agent", "inverse_agent", "knowledge_agent"]:
        graph.add_conditional_edges(agent, should_use_tool)

    # tool_executor → back to correct agent (based on intent in state)
    graph.add_conditional_edges("tool_executor", after_tool)

    graph.add_edge("reject", END)

    return graph.compile(checkpointer=MemorySaver())
```

---

## What does NOT change

- `app_chat.py` — calls `get_app()` and `app.stream()` identically, no changes needed
- `agent/prompts/system_prompt.md` — keep as-is; tool set restriction already limits scope
- All tool implementations in `agent/agent_tools.py` — untouched
- `agent/rag.py`, `agent/ingest.py` — untouched

---

## Verification

```bash
# From final/
conda activate jax_trial
streamlit run app_chat.py
```

Test each route:

| Intent | Test message |
|---|---|
| `forward` | "predict for T300/PESU with a11=0.7, a22=0.15, mf=0.3, ar=20" |
| `inverse` | "I have E1=15 GPa, E2=8 GPa for T300/PESU, run elastic inverse" |
| `knowledge` | "what is the modulus of carbon fiber T300?" |
| `out_of_scope` | "what is the weather today" |
| **mixed** | "predict for card 1, also I have new measurements to invert" |

Also verify: after a successful inverse solve, saying "save to card" stays within
`inverse_agent` and calls `save_to_card` correctly without re-routing through router.
