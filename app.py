"""
Synapse — AI Meeting Action-Item Extractor & Assignment Orchestrator.

This file is a composition layer only: it wires together session
state, the input panel, the dashboard components, and the AI
recommendation panel. It contains no business logic, no Gemini calls,
and no Pandas math of its own — all of that lives in services/,
utils/, and components/, per the project's architecture rules.
"""

from __future__ import annotations

import os

import streamlit as st

from components import agent_trace, ai_panel, charts, input_panel, kpi_cards, task_board
from utils import data_pipeline, session

st.set_page_config(
    page_title="Synapse — Meeting Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _load_css() -> None:
    css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def _render_header() -> None:
    meeting_count = session.get_meeting_count()
    status = f"{meeting_count} meeting{'s' if meeting_count != 1 else ''} processed" if meeting_count else "no meetings yet"
    st.markdown(
        f"""
        <div class="synapse-header">
            <div>
                <span class="synapse-title">SYNAPSE</span><br/>
                <span class="synapse-tagline">Turn meeting chaos into an accountable action plan.</span>
            </div>
            <div class="synapse-status-pill">{status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Meeting History")
        meetings = session.get_meetings()
        if not meetings:
            st.caption("Processed meetings will appear here.")
        else:
            for m in reversed(meetings):
                with st.expander(f"{m.meeting_title} ({len(m.tasks)} task{'s' if len(m.tasks) != 1 else ''})"):
                    st.caption(m.meeting_date.isoformat())
                    if m.summary:
                        st.write(m.summary)

        st.divider()
        if st.button("🔄 Start Fresh Demo", use_container_width=True):
            session.reset_session()
            st.rerun()


def main() -> None:
    session.init_session_state()
    _load_css()
    _render_header()
    _render_sidebar()

    meetings = session.get_meetings()
    df = data_pipeline.tasks_to_dataframe(meetings)

    # New-meeting input always available, regardless of empty state.
    input_panel.render_input_panel()
    agent_trace.render_agent_trace()

    if df.empty:
        st.info(
            "👋 No tasks yet. Paste a transcript or record a voice memo above "
            "to get started — Synapse will extract action items automatically."
        )
        return

    st.divider()
    kpi_cards.render_kpi_strip(df)

    st.divider()
    col_main, col_ai = st.columns([2, 1])
    with col_main:
        st.subheader("Task Board")
        task_board.render_task_board()

        st.subheader("Workload")
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.caption("Tasks per owner")
            charts.render_workload_chart(df)
        with chart_col2:
            st.caption("Priority distribution")
            charts.render_priority_chart(df)

        st.subheader("Needs Attention")
        charts.render_deadline_risk_table(df)

    with col_ai:
        ai_panel.render_ai_panel()


if __name__ == "__main__":
    main()
