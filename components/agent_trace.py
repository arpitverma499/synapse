"""
Agent trace panel — shows exactly what Gemini did on the most recent
call: which attempt, whether it called the expected tool, and with
what arguments. This is the visible evidence that Synapse uses real
function-calling rather than a plain-text chatbot response.
"""

from __future__ import annotations

import streamlit as st

from utils import session


def render_agent_trace() -> None:
    trace = session.get_last_trace()
    if not trace:
        return
    with st.expander(f"🔍 Agent trace ({len(trace)} attempt{'s' if len(trace) != 1 else ''})", expanded=False):
        for entry in trace:
            attempt = entry.get("attempt", "?")
            if "error" in entry:
                st.markdown(f"**Attempt {attempt}:** API error — `{entry['error']}`")
            elif entry.get("tool_call_args") is None:
                st.markdown(f"**Attempt {attempt}:** model responded without calling the expected tool")
            else:
                st.markdown(f"**Attempt {attempt}:** tool called")
                st.json(entry["tool_call_args"])
