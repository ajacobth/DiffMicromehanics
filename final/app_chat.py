"""
DiffMicromechanics chat interface.

Run from final/:
    streamlit run app_chat.py
"""

import random
import sys
import uuid
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessageChunk, ToolMessage

_FINAL = Path(__file__).parent
if str(_FINAL) not in sys.path:
    sys.path.insert(0, str(_FINAL))

from dotenv import load_dotenv
load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from agent.graph import build_app

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MateriAl",
    page_icon="🧪",
    layout="wide",
)

PROMPTS_DIR = _FINAL / "agent" / "prompts"

# ── Model selection — change ACTIVE_MODEL to switch ──────────────────────────
MODELS = {
    "haiku":    ("anthropic", "claude-haiku-4-5-20251001"),
    "sonnet":   ("anthropic", "claude-sonnet-4-6"),
    "local":    ("ollama",    "qwen2.5:14b-instruct-q4_K_M"),
    "local_q8": ("ollama",    "qwen2.5:14b-instruct-q8_0"),
    "local32":  ("ollama",    "qwen2.5:32b-instruct-q3_K_M"),
    "local2":   ("ollama",    "qwen3:8b"),
    "local3":   ("ollama",    "qwen3:14b"),
}
# Context window per model — balances speed (prefill ∝ num_ctx) vs capacity
MODEL_CTX = {
    "haiku":    None,   # cloud — no num_ctx
    "sonnet":   None,
    "local":    8192,   # 14B Q4 — fast prefill, fits system prompt (~7500 tok)
    "local_q8": 16384,  # 14B Q8 — larger ctx, still fast on 32GB
    "local32":  16384,
    "local2":   None,   # 8B — Ollama default (40960); explicit ctx made it slower
    "local3":   None,    # 14B — Ollama default; explicit ctx was slower
}
ACTIVE_MODEL = "local2"  # ← change this: "haiku" | "sonnet" | "local" | "local_q8" | "local32" | "local2" | "local3"

_THINKING_MSGS = [
    "Consulting the orientation tensor…",
    "Homogenizing the microstructure…",
    "Warming up the surrogate…",
    "Exploring Jeffrey's orbit"
    "Checking the stiffness matrix…",
    "Running Mori-Tanaka in my head…",
    "Crunching fiber orientations…",
    "Asking the neural network nicely…",
    "Computing effective properties…",
    "Browsing the material card library…",
    "Untangling the CTE coupling…",
    "Aligning the fibers…",
    "Ensuring FAA complaince…"
]


def _build_llm():
    provider, model_id = MODELS[ACTIVE_MODEL]
    if provider == "anthropic":
        return ChatAnthropic(model=model_id, temperature=0)
    # ChatOllama 0.3.x uses `reasoning` (maps to Ollama's `think` field); `think` kwarg is ignored
    kwargs = {"reasoning": True} if model_id.startswith("qwen3") else {}
    ctx = MODEL_CTX.get(ACTIVE_MODEL)
    if ctx is not None:
        kwargs["num_ctx"] = ctx
    return ChatOllama(
        model=model_id,
        temperature=0,
        streaming=True,
        keep_alive=-1,     # keep model loaded between turns (default: unload after 5 min)
        **kwargs,
    )


# ── Cached resources (survive Streamlit reruns) ───────────────────────────────

@st.cache_resource
def _prewarm_models():
    """Trigger JAX JIT compilation for all three surrogate models at startup."""
    import core.services.service_forward as _sfw
    _dummy = {
        "e1": 230000.0, "e2": 15000.0, "g12": 15000.0,
        "f_nu12": 0.2, "f_nu23": 0.25,
        "ar": 20.0, "fiber_massfrac": 0.20,
        "fiber_density": 1760.0, "matrix_modulus": 3100.0,
        "matrix_poisson": 0.37, "matrix_density": 1280.0,
        "a11": 0.60, "a22": 0.15, "a12": 0.0, "a13": 0.0, "a23": 0.0,
        "f_cte1": -0.5e-6, "f_cte2": 15.0e-6, "m_cte": 60.0e-6,
    }
    _thermal_dummy = {
        "k_f1": 8.0, "k_f2": 1.0, "k_m": 0.2,
        "ar_f": 20.0, "w_f": 0.20, "rho_f": 1760.0, "rho_m": 1280.0,
        "a11": 0.60, "a22": 0.15, "a12": 0.0, "a13": 0.0, "a23": 0.0,
    }
    try:
        _sfw.run_forward("elastic", _dummy)
        _sfw.run_forward("thermoelastic", _dummy)
        _sfw.run_forward("thermal", _thermal_dummy)
    except Exception:
        pass  # warmup failure is non-fatal


@st.cache_resource
def get_app():
    _prewarm_models()
    system = (PROMPTS_DIR / "system_prompt.md").read_text()
    vocab  = (PROMPTS_DIR / "vocabulary.md").read_text()
    return build_app(_build_llm(), f"{system}\n\n---\n\n{vocab}")


# ── Session state initialisation ─────────────────────────────────────────────

if "thread_id"       not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]
if "display_msgs"    not in st.session_state:
    st.session_state.display_msgs = []   # list of {role, content, tools}


def get_config() -> dict:
    return {"configurable": {"thread_id": st.session_state.thread_id}}


def get_graph_state() -> dict:
    """Read current card and completed stages from the graph's MemorySaver."""
    try:
        state = get_app().get_state(get_config())
        return state.values if state else {}
    except Exception:
        return {}


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🧪 MateriAl")
    st.caption(f"Model: `{ACTIVE_MODEL}` ({MODELS[ACTIVE_MODEL][1]})")
    st.divider()

    graph_state       = get_graph_state()
    current_card_id   = graph_state.get("current_card_id")
    completed_stages  = graph_state.get("completed_stages") or []
    st.subheader("Session")
    st.caption(f"Thread: `{st.session_state.thread_id}`")

    st.subheader("Active Card")
    if current_card_id:
        st.success(f"Card #{current_card_id}")
        if completed_stages:
            for s in completed_stages:
                st.markdown(f"- ✅ {s}")
        else:
            st.caption("No stages complete yet")
    else:
        st.info("No card loaded")

    st.divider()
    if st.button("🗑️ New Session", use_container_width=True):
        st.session_state.thread_id  = str(uuid.uuid4())[:8]
        st.session_state.display_msgs = []
        st.rerun()



# ── Chat history ──────────────────────────────────────────────────────────────

st.title("MateriAl")

for msg in st.session_state.display_msgs:
    with st.chat_message(msg["role"]):
        if msg.get("tools"):
            for tool_name, tool_output in msg["tools"]:
                with st.expander(f"🔧 `{tool_name}`", expanded=False):
                    st.text(tool_output)
        if msg["content"]:
            st.markdown(msg["content"])


# ── Input and streaming response ──────────────────────────────────────────────

if prompt := st.chat_input("Ask about your composite materials…"):

    # Show user message immediately
    st.session_state.display_msgs.append({"role": "user", "content": prompt, "tools": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream agent response
    app = get_app()
    config = get_config()

    with st.chat_message("assistant"):
        text_box    = st.empty()          # live-updating token display
        full_text   = ""
        tools_shown = []                  # (tool_name, output) pairs for this turn

        # Show a fun waiting message until the first token arrives
        text_box.markdown(f"*{random.choice(_THINKING_MSGS)}*")

        # Track which tool is currently "in flight"
        active_tool_name   = None
        active_tool_status = None

        # Track thinking (reasoning) token stream
        thinking_status = None
        thinking_box    = None
        thinking_text   = ""

        try:
            for chunk, metadata in app.stream(
                {"messages": [HumanMessage(content=prompt)]},
                config,
                stream_mode="messages",
            ):
                # ── LLM output ────────────────────────────────────────────
                if isinstance(chunk, AIMessageChunk):

                    # Surface Qwen3 reasoning/thinking tokens
                    raw_thinking = (
                        chunk.additional_kwargs.get("reasoning_content") or
                        chunk.additional_kwargs.get("thinking") or
                        ""
                    )
                    if raw_thinking:
                        if thinking_status is None:
                            text_box.empty()   # remove the waiting message
                            thinking_status = st.status("Reasoning…", expanded=True)
                            thinking_box    = thinking_status.empty()
                        thinking_text += raw_thinking
                        thinking_box.markdown(thinking_text)

                    # LLM is deciding to call a tool
                    for tc in (chunk.tool_call_chunks or []):
                        name = tc.get("name", "")
                        if name and name != active_tool_name:
                            active_tool_name   = name
                            active_tool_status = st.status(
                                f"Calling `{name}`…", expanded=False
                            )

                    # LLM is generating text (final response, not a tool call)
                    if chunk.content and not chunk.tool_call_chunks:
                        # Collapse thinking expander when real text starts
                        if thinking_status is not None:
                            thinking_status.update(
                                label="Reasoning", state="complete", expanded=False
                            )
                            thinking_status = None
                        content = chunk.content
                        if isinstance(content, list):
                            content = "".join(
                                b["text"] if isinstance(b, dict) and b.get("type") == "text" else ""
                                for b in content
                            )
                        full_text += content
                        text_box.markdown(full_text + "▌")

                # ── Tool result ───────────────────────────────────────────
                elif isinstance(chunk, ToolMessage):
                    output = chunk.content or ""
                    tools_shown.append((chunk.name, output))

                    if active_tool_status:
                        active_tool_status.update(
                            label=f"🔧 `{chunk.name}` done",
                            state="complete",
                            expanded=False,
                        )
                        with active_tool_status:
                            # Truncate very long tool outputs for display
                            display = output if len(output) <= 1000 else output[:1000] + "\n…(truncated)"
                            st.text(display)
                        active_tool_name   = None
                        active_tool_status = None

        except Exception as e:
            st.error(f"Agent error: {e}")
            full_text = f"*(error: {e})*"

        # Ensure thinking expander is closed if turn ended without text
        if thinking_status is not None:
            thinking_status.update(label="Reasoning", state="complete", expanded=False)

        # Remove streaming cursor, show final text
        text_box.markdown(full_text)

    # Persist to display history
    st.session_state.display_msgs.append({
        "role":    "assistant",
        "content": full_text,
        "tools":   tools_shown,
    })

    # Sidebar card state refreshes naturally on the next user interaction
