"""
Editable task board (st.data_editor).

Split per the architecture:
  compute_edits()     — pure function: diffs an edited DataFrame
                         against the original and returns only the
                         fields that actually changed, per task id.
                         No st.* calls — testable without Streamlit.
  render_task_board()  — the st.data_editor widget itself and applying
                         confirmed diffs via data_pipeline.apply_task_update.

Editable columns are deliberately limited to owner, priority, status,
and deadline — the fields a human plausibly needs to correct after
AI extraction. id, description, confidence, and source_excerpt stay
read-only: they're either identifiers or the AI's own evidence trail,
not things a user should silently overwrite.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from utils import data_pipeline, session

EDITABLE_COLUMNS = ["owner", "priority", "status", "deadline"]


def compute_edits(original_df: pd.DataFrame, edited_df: pd.DataFrame) -> dict[str, dict]:
    """
    Compares edited_df (data_editor's current output) against
    original_df (what was loaded into the widget this render) and
    returns {task_id: {field: new_value, ...}} for every row where at
    least one editable field actually changed. Rows/columns outside
    EDITABLE_COLUMNS are ignored even if somehow different.
    """
    if original_df.empty or edited_df.empty:
        return {}

    orig_indexed = original_df.set_index("id")
    edited_indexed = edited_df.set_index("id")

    changes: dict[str, dict] = {}
    for task_id in edited_indexed.index:
        if task_id not in orig_indexed.index:
            continue  # a row that no longer exists in the source of truth
        row_changes = {}
        for col in EDITABLE_COLUMNS:
            old_val = orig_indexed.loc[task_id, col]
            new_val = edited_indexed.loc[task_id, col]
            if _normalize(old_val) != _normalize(new_val):
                row_changes[col] = new_val
        if row_changes:
            changes[task_id] = row_changes
    return changes


def _normalize(value):
    """Deadline columns come back from data_editor as pandas Timestamp
    or NaT; the underlying Task field is a plain date or None. Compare
    on a common representation so a no-op round-trip isn't flagged as
    a change."""
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.date()
    return value


def render_task_board() -> None:
    meetings = session.get_meetings()
    df = data_pipeline.tasks_to_dataframe(meetings)

    if df.empty:
        st.info("No tasks yet — process a meeting above to populate the task board.")
        return

    st.caption("Edit owner, priority, status, or deadline directly — changes are validated and saved immediately.")

    display_df = df[["id", "description", "owner", "priority", "status", "deadline", "meeting_title"]].copy()

    edited_df = st.data_editor(
        display_df,
        key="synapse_task_board_editor",
        hide_index=True,
        use_container_width=True,
        disabled=["id", "description", "meeting_title"],
        column_config={
            "id": st.column_config.TextColumn("ID", width="small"),
            "description": st.column_config.TextColumn("Task", width="large"),
            "owner": st.column_config.TextColumn("Owner"),
            "priority": st.column_config.SelectboxColumn("Priority", options=["High", "Medium", "Low"]),
            "status": st.column_config.SelectboxColumn("Status", options=["Open", "In Progress", "Done"]),
            "deadline": st.column_config.DateColumn("Deadline"),
            "meeting_title": st.column_config.TextColumn("Meeting", width="medium"),
        },
    )

    changes = compute_edits(display_df, edited_df)
    if not changes:
        return

    applied, failed = 0, []
    for task_id, updates in changes.items():
        # deadline arrives from data_editor as a python date already
        # (DateColumn), or None — matches Task.deadline's type, no
        # coercion needed here.
        try:
            ok = data_pipeline.apply_task_update(meetings, task_id, updates)
            if ok:
                applied += 1
            else:
                failed.append(task_id)
        except Exception as e:
            # Task has validate_assignment=True (see models/schemas.py) —
            # an invalid edit (e.g. a malformed value) raises here rather
            # than silently corrupting state. Surface it, don't crash.
            failed.append(f"{task_id} ({e})")

    if failed:
        st.error(f"Some edits couldn't be applied: {failed}")
    if applied:
        st.rerun()
