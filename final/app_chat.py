"""
DiffMicromechanics chat interface.

Run from final/:
    streamlit run app_chat.py
"""

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
    "haiku":  ("anthropic", "claude-haiku-4-5-20251001"),
    "sonnet": ("anthropic", "claude-sonnet-4-6"),
    "local":  ("ollama",    "qwen2.5:14b-instruct-q4_K_M"),
    "local2": ("ollama",    "qwen3:8b"),
    "local3": ("ollama",    "qwen3:14b"),
}
ACTIVE_MODEL = "local"   # ← change this: "haiku" | "sonnet" | "local" | "local2" | "local3"


def _build_llm():
    provider, model_id = MODELS[ACTIVE_MODEL]
    if provider == "anthropic":
        return ChatAnthropic(model=model_id, temperature=0)
    return ChatOllama(model=model_id, temperature=0, streaming=True)


# ── Cached resources (survive Streamlit reruns) ───────────────────────────────

@st.cache_resource
def get_app():
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

    st.divider()
    st.caption("Tips")
    st.markdown(
        "**Prefix your message with:**\n\n"
        "- `PREDICT` — forward prediction\n"
        "- `INVERSE` — characterize from measurements\n"
        "- `SEARCH` — material lookup / theory\n\n"
        "**Examples:**\n\n"
        "- *\"PREDICT card 1\"*\n"
        "- *\"INVERSE elastic stage for T300/PESU\"*\n"
        "- *\"SEARCH what is the modulus of carbon fiber\"*\n"
        "- *\"what cards do we have\"*\n"
        "- *\"can I infer a11 from E1?\"*"
    )


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

if prompt := st.chat_input("PREDICT / INVERSE / SEARCH — ask about your composite materials…"):

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

        # Track which tool is currently "in flight"
        active_tool_name   = None
        active_tool_status = None

        try:
            for chunk, metadata in app.stream(
                {"messages": [HumanMessage(content=prompt)]},
                config,
                stream_mode="messages",
            ):
                # ── LLM output ────────────────────────────────────────────
                if isinstance(chunk, AIMessageChunk):

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

        # Remove streaming cursor, show final text
        text_box.markdown(full_text)

    # Persist to display history
    st.session_state.display_msgs.append({
        "role":    "assistant",
        "content": full_text,
        "tools":   tools_shown,
    })

    # Refresh sidebar card state after the turn
    st.rerun()
