---
name: Agent Implementation Progress
description: Current state of the agentic layer build — what's done, what's next, key design decisions made
type: project
---

## What has been built

### Files created
- `agent/__init__.py` — empty, makes agent a package
- `agent/state.py` — AgentState TypedDict (3 fields: messages, current_card_id, completed_stages)
- `agent/prompts/system_prompt.md` — full physics knowledge prompt (role, 3-layer model, 4-stage workflow, identifiability rules, unit handling, conversation/save behavior)
- `agent/prompts/vocabulary.md` — field name/synonym mapping with units, conversions, typical ranges
- `run_agent.py` — entry point in final/ alongside GUI files. Currently has an inline minimal graph (no tools). Working: LLM responds to user messages with system prompt context injected. NOT yet working: streaming (deferred), tools, session restore.
- `test_agent_stack.py` — stack validation: tests basic reachability, tool calling, LangGraph loop
- `requirements.txt` — project deps including langgraph, langchain-core, langchain-ollama

### AGENT_ORCHESTRATION.md
Fully updated to reflect all design decisions made in this session (Sections 12 and 13 added/revised).

---

## Key design decisions made

**Framework**: LangGraph (not plain while loop) because multi-agent future (simulation agent, surrogate fitting agent) needs supervisor pattern. Plain loop can't coordinate agents cleanly.

**Graph**: 3 nodes — agent_node (LLM, all logic), tool_executor (ToolNode), stage_updater (pure Python state update). See Section 12 of AGENT_ORCHESTRATION.md for full diagram and code.

**State**: Only 3 fields — messages (add_messages reducer), current_card_id (Optional[int]), completed_stages (list[str]). locked_inputs removed — service layer reads prior stage results from DB via card_id directly (same as GUI). User must save after each stage before proceeding.

**Persistence**: MemorySaver only (in-memory, no SqliteSaver). On resume, restore_state_from_card() infers completed_stages from constituent_property_values table (matrix_E present → elastic done, f_cte1 → thermoelastic, k_p1 → thermal). No new DB tables needed.

**Save behavior**: Agent asks after every successful solve. If user explicitly says save → call save_to_card immediately, no re-confirmation. Unsaved results lost on restart (same contract as GUI).

**Re-runs**: stage_updater uses DOWNSTREAM dict to invalidate later stages when an earlier stage is re-run (elastic re-run invalidates thermoelastic + thermal).

**Hardware**: M2 Max, 32GB. Use Llama 3.3 32B Q4 or Qwen2.5 32B Q4 (~20GB). 70B won't fit. Ollama uses Metal automatically.

**LLM**: Currently qwen2.5:7b-instruct for development/testing.

**Streaming**: Deferred. astream_events approach documented but not yet working. User wants to get tools working first.

---

## What still needs to be built (in order)

1. `agent/agent_tools.py` — @tool wrappers over service layer (run_elastic_inverse, run_thermoelastic_inverse, run_thermal_inverse, run_transfer, run_forward, load_material_card, save_to_card, list_materials, list_cards, get_card_status, get_model_inputs, get_model_outputs)
2. `agent/graph.py` — full 3-node graph with ToolNode and stage_updater, replacing inline graph in run_agent.py
3. `agent/session.py` — restore_state_from_card(), resolve_card_from_message()
4. Update run_agent.py — replace inline graph with `from agent.graph import build_app`, plug in session.py helpers
5. Streaming — revisit after tools are working

---

## Current run_agent.py state

Works: LLM responds, system prompt loaded, conversation loop, --card CLI arg parsed.
Placeholder stubs in run_agent.py:
- resolve_card() — name lookup not implemented, only int card IDs work
- restore_state() — doesn't read DB yet, returns empty state
- graph is inline minimal (no tools), will be replaced by agent/graph.py import

## Python version note
Environment is Python 3.9. Use Optional[X] not X | None for type hints.
