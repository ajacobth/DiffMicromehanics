---
name: Agent Implementation Progress
description: Current state of the agentic layer build — what's done, what's next, key design decisions made
type: project
---

## What has been built

### Files created / in place
- `agent/__init__.py` — empty, makes agent a package
- `agent/state.py` — AgentState TypedDict (3 fields: messages, current_card_id, completed_stages)
- `agent/prompts/system_prompt.md` — full physics prompt (role, 3-layer model, 4-stage workflow, identifiability rules, unit handling, conversation/save behavior, knowledge-base rules)
- `agent/prompts/vocabulary.md` — field name/synonym mapping with units, conversions, typical ranges
- `agent/rag.py` — parent-document RAG: 300-char child chunks for retrieval, full parent pages returned to LLM. PDFs go in `agent/knowledge/`
- `agent/graph.py` — full LangGraph 3-node graph (agent_node → tool_executor → agent_node). MemorySaver checkpointer. DEBUG prints on tool_calls and response type still present (user may want to remove later).
- `agent/agent_tools.py` — all discovery tools (see below)
- `agent/ingest.py` — PDF ingestion script for the knowledge base
- `run_agent.py` — entry point. Model: `qwen2.5:14b-instruct-q4_K_M`. Conversation loop with --card CLI arg. Session restore stubs (resolve_card, restore_state) still placeholder.
- `requirements.txt` — langgraph, langchain-core, langchain-ollama, langchain-community, etc.
- `core/services/service_material.py` — added `get_completed_stages(card_id, fiber_id, polymer_id) -> list[str]`
- `core/services/service_cards.py` — `load_card(card_id)` returns full card dict used by get_card_status

### Tools in agent/agent_tools.py (TOOLS list)
All tools are thin wrappers over core/services — zero db.py imports in agent_tools.py.

| Tool | Description |
|---|---|
| `search_knowledge_base(query)` | RAG over agent/knowledge/ PDFs |
| `list_materials()` | All fibers, polymers, printers with datasheet props |
| `get_material_details(material_name)` | Full datasheet for ONE fiber or polymer by name (partial, case-insensitive). Returns E1, E2, G12, nu12, nu23, density, CTE1, CTE2, k1, k2. Use this for single-material questions — not list_materials |
| `list_cards()` | All material cards with completed stages |
| `get_card_status(card_id)` | Full card detail: microstructure, constituent properties with [inferred/inputted] tags, experimental measurements |

### Stage detection logic (service_material.get_completed_stages)
- elastic → polymer has inferred `matrix_modulus`
- thermoelastic → fiber has inferred `f_cte1`
- thermal → fiber has inferred `k_f1`

---

## Key design decisions

**Framework**: LangGraph (not plain while loop) — supervisor pattern needed for future multi-agent.

**Graph**: agent_node → (has tool_calls?) → tool_executor → agent_node, else END. MemorySaver only.

**LLM**: `qwen2.5:14b-instruct-q4_K_M` via Ollama. Reliable tool selection. DO NOT add language instructions — they cause Thai/Chinese responses. Without them qwen2.5:14b responds correctly in English.

**llama3.1:8b is NOT suitable** — unreliable tool selection, hallucinates from training knowledge.

**State**: messages (add_messages), current_card_id (Optional[int]), completed_stages (list[str]).

**Persistence**: MemorySaver (in-memory). restore_state_from_card() not yet implemented.

**Save behavior**: Agent asks after solve. User explicitly says save → call save_to_card immediately, no re-confirmation.

**Service layer**: agent_tools.py → core/services/ → db/db.py. Never import db.py directly in tools.

---

## Known issues / prompt tuning history

### Language issue (RESOLVED — no language instructions needed)
qwen2.5:14b responds in Thai/Chinese when ANY language instruction is in the system prompt or user messages. Removing all language instructions results in correct English. Do NOT add "Respond in English" or similar.

### Tool selection issues resolved
- "what material cards do we have" → was calling `list_materials`. Fixed by adding trigger phrases and negative trigger ("Do NOT call this when asking about cards") to docstrings.
- Single-material questions → was calling `list_materials` (got all, then summarized, dropped fields). Fixed by adding `get_material_details` tool for focused single-material output.
- Follow-up questions like "what is nu23 of AF" → model skipped tool call and hallucinated. Mitigated by `get_material_details` (focused output = less summarization) and system prompt strengthening.

### Hallucination of abbreviations (RESOLVED)
qwen2.5:14b expanded "CMSC" to "Continuous Melt Spun Carbon" (fabricated). Fixed in system_prompt.md: "report only what the tool returned — do not expand abbreviations or supplement with training knowledge."

### nu23 missing from list_materials output (RESOLVED differently)
nu23 was in tool output but dropped in model summary when filtering all-materials list to one material. Root fix: `get_material_details` returns focused single-material output so model doesn't need to filter/truncate.

---

## What still needs to be built (in order)

1. **Solver tools** in `agent/agent_tools.py`:
   - `run_elastic_inverse(card_id, measurements)` → calls core/services/service_inverse.py (wraps core/inverse.py)
   - `run_thermoelastic_inverse(card_id, measurements)` → same pattern
   - `run_thermal_inverse(card_id, csv_path)` → wraps core/inverse_thermal.py
   - `run_forward(card_id, microstructure_override)` → wraps core/forward.py
   - `save_to_card(card_id, stage, results)` → writes to DB via service layer
   - `load_material_card(card_id)` → alias of get_card_status but returns raw dict for agent use

2. **`agent/session.py`**:
   - `restore_state_from_card(card_id)` — reads completed_stages from DB
   - `resolve_card_from_message(name_str)` — fuzzy name→card_id lookup

3. **Update `run_agent.py`** — replace stub helpers with session.py imports

4. **Streaming** — deferred, revisit after solver tools work

5. **Remove DEBUG prints** from `agent/graph.py` (lines that print tool_calls and response type) — user hasn't asked yet but they're still there

---

## Python version note
Environment is Python 3.9. Use `Optional[X]` not `X | None` for type hints.
Conda env: `jax_trial` (not diffmech). JAX 0.4.26.
