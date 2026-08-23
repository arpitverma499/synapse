"""
AI assignment recommendations + human approval.

This is the human-in-the-loop centerpiece: Gemini only ever produces
a *recommendation* (services.gemini_service.recommend_assignments).
Nothing about a task's ownership changes until a human clicks Approve
— rejecting or ignoring a recommendation leaves the task exactly as
it was. The mutation itself happens only inside _approve(), gated by
session.approve_recommendation() returning True (which itself enforces
the stale-guard: a recommendation generated against an older task
list cannot be approved after the task list has since changed).
"""

from __future__ import annotations

from datetime import date

import streamlit as st

from utils import data_pipeline, errors, session
from services import gemini_service


def _generate_recommendations() -> None:
    meetings = session.get_meetings()
    df = data_pipeline.tasks_to_dataframe(meetings)
    unassigned = data_pipeline.unassigned_tasks_for_prompt(df)
    workload = data_pipeline.workload_context_for_prompt(df)
    roster = data_pipeline.known_team_members(meetings)

    recs, error, _trace = gemini_service.recommend_assignments(
        unassigned_tasks=unassigned,
        workload_context=workload,
        known_team_members=roster,
        as_of=date.today(),
    )

    if error is not None:
        st.error(f"Couldn't generate recommendations: {error.message}")
        hint = errors.friendly_hint(error)
        if hint:
            st.caption(hint)
        return

    session.set_pending_recommendations([r.model_dump() for r in recs])
    st.rerun()


def _approve(task_id: str, recommended_owner: str) -> None:
    approved = session.approve_recommendation(task_id)
    if not approved:
        # Stale-guard tripped, or already decided — the task list
        # changed since this recommendation was generated. Nothing is
        # applied; the UI will simply no longer show this card.
        st.warning("This recommendation is no longer valid (the task list has changed since it was generated).")
        return
    try:
        ok = data_pipeline.apply_task_update(session.get_meetings(), task_id, {"owner": recommended_owner})
        if not ok:
            st.warning("That task no longer exists.")
    except Exception as e:
        st.error(f"Couldn't apply the assignment: {e}")
        return
    st.rerun()


def _reject(task_id: str) -> None:
    session.reject_recommendation(task_id)
    st.rerun()


def render_ai_panel() -> None:
    meetings = session.get_meetings()
    if not meetings:
        return

    df = data_pipeline.tasks_to_dataframe(meetings)
    unassigned_count = int((df["owner"] == "Unassigned").sum()) if not df.empty else 0

    st.subheader("AI Assignment Recommendations")

    pending = session.get_valid_pending_recommendations()

    if not pending:
        if unassigned_count == 0:
            st.caption("No unassigned tasks right now.")
            return
        st.caption(f"{unassigned_count} task(s) are unassigned.")
        if st.button("Generate Assignment Recommendations", type="primary"):
            with st.spinner("Synapse is analyzing workload and recommending owners..."):
                _generate_recommendations()
        return

    st.caption(f"{len(pending)} recommendation(s) awaiting your review.")
    for rec in pending:
        task = data_pipeline.find_task(meetings, rec["task_id"])
        task_desc = task.description if task else "(task no longer available)"
        with st.container(border=True):
            st.markdown(f"**{task_desc}**")
            st.markdown(
                f"→ Recommend: **{rec['recommended_owner']}** "
                f"(confidence: {rec['confidence']:.0%})"
            )
            st.caption(rec["rationale"])
            if rec.get("alternative_candidate"):
                st.caption(f"Alternative: {rec['alternative_candidate']}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Approve", key=f"approve_{rec['task_id']}", use_container_width=True):
                    _approve(rec["task_id"], rec["recommended_owner"])
            with col2:
                if st.button("❌ Reject", key=f"reject_{rec['task_id']}", use_container_width=True):
                    _reject(rec["task_id"])
