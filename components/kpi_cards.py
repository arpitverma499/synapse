"""
KPI strip — the dashboard's top-level metric cards. Pure rendering:
all numbers come from utils/data_pipeline.compute_kpis, which is the
only place this math happens. This module just lays out st.metric
cards and formats deltas.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from utils import data_pipeline, session


def _delta_str(value) -> str | None:
    """None means 'no prior snapshot' -> st.metric shows no delta at
    all, rather than a fabricated '+0'."""
    if value is None:
        return None
    if value == 0:
        return "0"
    return f"{value:+d}"


def render_kpi_strip(df: pd.DataFrame) -> None:
    prev_snapshot = session.get_prev_kpi_snapshot()
    kpis = data_pipeline.compute_kpis(df, previous_df=None, as_of=date.today())
    # Deltas come from the session's stored pre-meeting snapshot (captured
    # in input_panel._handle_submission), not recomputed here — this is
    # "since your last processed meeting," an honest comparison, not a
    # fabricated one.
    delta_total = None
    delta_unassigned = None
    delta_overdue = None
    if prev_snapshot is not None:
        delta_total = kpis["total_tasks"] - prev_snapshot.get("total_tasks", kpis["total_tasks"])
        delta_unassigned = kpis["unassigned_tasks"] - prev_snapshot.get("unassigned_tasks", kpis["unassigned_tasks"])
        delta_overdue = kpis["overdue_tasks"] - prev_snapshot.get("overdue_tasks", kpis["overdue_tasks"])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Tasks", kpis["total_tasks"], delta=_delta_str(delta_total))
    with col2:
        st.metric("Unassigned", kpis["unassigned_tasks"], delta=_delta_str(delta_unassigned),
                   delta_color="inverse")
    with col3:
        st.metric("Overdue", kpis["overdue_tasks"], delta=_delta_str(delta_overdue),
                   delta_color="inverse")
    with col4:
        st.metric("Active Owners", kpis["active_owners"])
