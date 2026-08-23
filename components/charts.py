"""
Dashboard charts. Every chart here renders a DataFrame that
utils/data_pipeline.py already computed — no math happens in this
module, only st.bar_chart/st.dataframe layout.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from utils import data_pipeline


def render_workload_chart(df: pd.DataFrame) -> None:
    workload = data_pipeline.workload_by_owner(df)
    if workload.empty:
        st.caption("No assigned tasks yet — workload will appear here once tasks have owners.")
        return
    st.bar_chart(workload.set_index("owner")["task_count"], color="#5B8DEF")


def render_priority_chart(df: pd.DataFrame) -> None:
    priority = data_pipeline.priority_breakdown(df)
    if priority.empty:
        st.caption("No tasks yet.")
        return
    st.bar_chart(priority.set_index("priority")["count"], color="#E67E5A")


def render_deadline_risk_table(df: pd.DataFrame) -> None:
    risk = data_pipeline.deadline_risk_table(df, horizon_days=5)
    if risk.empty:
        st.caption("Nothing overdue or due in the next 5 days. ✅")
        return
    display = risk[["description", "owner", "priority", "risk"]].rename(columns={
        "description": "Task", "owner": "Owner", "priority": "Priority", "risk": "Status",
    })
    st.dataframe(display, hide_index=True, use_container_width=True)
