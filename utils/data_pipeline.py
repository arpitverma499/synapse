"""
Deterministic data pipeline for Synapse.

Everything in this module is plain Pandas/Python — no Gemini calls.
Per project rules, anything that can be reliably computed with math
(totals, aggregations, overdue checks) belongs here, not in a prompt.
This keeps Gemini's job honest: reasoning and extraction, not
arithmetic it might get subtly wrong.

Every public function accepts either a list[Task]/list[MeetingExtraction]
or an already-built DataFrame, and every function tolerates empty input
(returns an empty-but-correctly-shaped DataFrame/dict rather than
raising) since a brand-new session has no tasks yet.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd

from models.schemas import MeetingExtraction, Task

TASK_COLUMNS = [
    "id", "description", "owner", "deadline", "priority",
    "status", "confidence", "source_excerpt", "meeting_id", "meeting_title",
]


def _empty_task_df() -> pd.DataFrame:
    """A correctly-typed empty frame so downstream groupby/merge calls
    don't blow up on a fresh session with zero meetings processed."""
    df = pd.DataFrame(columns=TASK_COLUMNS)
    df["deadline"] = pd.to_datetime(df["deadline"])
    df["confidence"] = df["confidence"].astype(float)
    return df


def tasks_to_dataframe(extractions: list[MeetingExtraction]) -> pd.DataFrame:
    """
    Flattens one or more MeetingExtraction results (i.e. the full
    session history) into a single tidy DataFrame — one row per task,
    tagged with which meeting it came from. This is the one function
    every other function in this module builds on.
    """
    if not extractions:
        return _empty_task_df()

    rows = []
    for extraction in extractions:
        for task in extraction.tasks:
            rows.append({
                "id": task.id,
                "description": task.description,
                "owner": task.owner or "Unassigned",
                "deadline": pd.to_datetime(task.deadline) if task.deadline else pd.NaT,
                "priority": task.priority.value if hasattr(task.priority, "value") else str(task.priority),
                "status": task.status.value if hasattr(task.status, "value") else str(task.status),
                "confidence": float(task.confidence),
                "source_excerpt": task.source_excerpt,
                "meeting_id": extraction.meeting_id,
                "meeting_title": extraction.meeting_title,
            })

    if not rows:
        return _empty_task_df()

    df = pd.DataFrame(rows, columns=TASK_COLUMNS)
    df["deadline"] = pd.to_datetime(df["deadline"])
    return df


def compute_kpis(df: pd.DataFrame, previous_df: Optional[pd.DataFrame] = None,
                  as_of: Optional[date] = None) -> dict:
    """
    Top-level KPI numbers for the dashboard's metric row, with deltas
    against a previous snapshot (e.g. task counts before this meeting
    vs. after, or last meeting vs. this one) when provided.

    Deltas are omitted (None) rather than fabricated when there's no
    previous snapshot to compare against — a KPI card with no real
    prior state should show a plain number, not an invented "+0".
    """
    as_of = as_of or date.today()
    total = len(df)
    unassigned = int((df["owner"] == "Unassigned").sum()) if total else 0
    overdue = int(_overdue_mask(df, as_of).sum()) if total else 0
    owners = int(df.loc[df["owner"] != "Unassigned", "owner"].nunique()) if total else 0

    kpis = {
        "total_tasks": total,
        "unassigned_tasks": unassigned,
        "overdue_tasks": overdue,
        "active_owners": owners,
        "delta_total_tasks": None,
        "delta_unassigned_tasks": None,
        "delta_overdue_tasks": None,
    }

    if previous_df is not None and len(previous_df):
        prev_total = len(previous_df)
        prev_unassigned = int((previous_df["owner"] == "Unassigned").sum())
        prev_overdue = int(_overdue_mask(previous_df, as_of).sum())
        kpis["delta_total_tasks"] = total - prev_total
        kpis["delta_unassigned_tasks"] = unassigned - prev_unassigned
        kpis["delta_overdue_tasks"] = overdue - prev_overdue

    return kpis


def _overdue_mask(df: pd.DataFrame, as_of: date) -> pd.Series:
    """A task is overdue if it has a deadline in the past AND isn't
    already marked Done. Tasks with no deadline are never overdue —
    that's a data-completeness issue, not a risk signal, and is
    already surfaced separately via unassigned/no-deadline counts."""
    if df.empty:
        return pd.Series(dtype=bool)
    has_deadline = df["deadline"].notna()
    is_past = df["deadline"] < pd.Timestamp(as_of)
    not_done = df["status"] != "Done"
    return has_deadline & is_past & not_done


def workload_by_owner(df: pd.DataFrame) -> pd.DataFrame:
    """Task count per owner, for the workload bar chart. Excludes
    'Unassigned' from the chart itself (it's tracked as its own KPI,
    not a person's workload) but callers can re-add it if needed."""
    if df.empty:
        return pd.DataFrame(columns=["owner", "task_count"])
    counts = (
        df[df["owner"] != "Unassigned"]
        .groupby("owner", as_index=False)
        .size()
        .rename(columns={"size": "task_count"})
        .sort_values("task_count", ascending=False)
    )
    return counts.reset_index(drop=True)


def utilization_by_owner(df: pd.DataFrame, capacity: int = 5) -> pd.DataFrame:
    """
    Simple utilization = open task count / capacity, expressed as a
    percentage. `capacity` is a configurable "comfortable open-task
    load" per person (default 5), not derived from anything Gemini
    said — this is intentionally a plain, adjustable business rule.
    """
    if df.empty:
        return pd.DataFrame(columns=["owner", "open_tasks", "utilization_pct"])
    open_df = df[(df["owner"] != "Unassigned") & (df["status"] != "Done")]
    counts = open_df.groupby("owner", as_index=False).size().rename(columns={"size": "open_tasks"})
    counts["utilization_pct"] = (counts["open_tasks"] / capacity * 100).round(0).clip(upper=150)
    return counts.sort_values("utilization_pct", ascending=False).reset_index(drop=True)


def completion_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Status breakdown (Open / In Progress / Done) with counts and
    percentage share, for a completion chart or table."""
    if df.empty:
        return pd.DataFrame(columns=["status", "count", "pct"])
    counts = df.groupby("status", as_index=False).size().rename(columns={"size": "count"})
    counts["pct"] = (counts["count"] / counts["count"].sum() * 100).round(1)
    return counts


def priority_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Priority distribution (High/Medium/Low counts), for the
    priority chart. Ordered explicitly so charts don't sort
    alphabetically (which would put High last)."""
    if df.empty:
        return pd.DataFrame(columns=["priority", "count"])
    order = ["High", "Medium", "Low"]
    counts = df.groupby("priority", as_index=False).size().rename(columns={"size": "count"})
    counts["priority"] = pd.Categorical(counts["priority"], categories=order, ordered=True)
    return counts.sort_values("priority").reset_index(drop=True)


def deadline_risk_table(df: pd.DataFrame, as_of: Optional[date] = None,
                         horizon_days: int = 3) -> pd.DataFrame:
    """
    Tasks that are either already overdue or due within `horizon_days`
    — the "needs attention now" view. Adds a plain-language risk label
    computed in Python (no AI) so the UI can badge rows without an
    API call.
    """
    as_of = as_of or date.today()
    if df.empty:
        return pd.DataFrame(columns=[*TASK_COLUMNS, "risk"])

    horizon = pd.Timestamp(as_of) + pd.Timedelta(days=horizon_days)
    has_deadline = df["deadline"].notna()
    not_done = df["status"] != "Done"
    in_window = has_deadline & not_done & (df["deadline"] <= horizon)
    at_risk = df[in_window].copy()

    def _risk_label(deadline: pd.Timestamp) -> str:
        days = (deadline - pd.Timestamp(as_of)).days
        if days < 0:
            return f"Overdue ({abs(days)}d)"
        if days == 0:
            return "Due today"
        return f"Due in {days}d"

    if not at_risk.empty:
        at_risk["risk"] = at_risk["deadline"].apply(_risk_label)
        at_risk = at_risk.sort_values("deadline")
    else:
        at_risk["risk"] = pd.Series(dtype=str)

    return at_risk.reset_index(drop=True)


def known_team_members(extractions: list[MeetingExtraction]) -> list[str]:
    """
    Distinct owners seen across all processed meetings so far, used
    as the 'known team members' roster fed into Gemini's dynamic
    context (see gemini_service._build_context) — the model prefers
    real known names over guessing when a transcript is ambiguous.
    """
    df = tasks_to_dataframe(extractions)
    if df.empty:
        return []
    names = sorted(df.loc[df["owner"] != "Unassigned", "owner"].unique().tolist())
    return names


def find_task(extractions: list[MeetingExtraction], task_id: str) -> Optional[Task]:
    """
    Locates the actual Task object (not a DataFrame row) with the
    given id across all meetings. Returns the live object — mutating
    it mutates the session's real data, which is exactly what
    apply_task_update() below relies on. Returns None if not found,
    rather than raising, since "task no longer exists" is a normal
    condition a caller should handle (e.g. a stale UI reference).
    """
    for extraction in extractions:
        for task in extraction.tasks:
            if task.id == task_id:
                return task
    return None


def apply_task_update(extractions: list[MeetingExtraction], task_id: str, updates: dict) -> bool:
    """
    Applies a set of field updates (e.g. from a data_editor row edit
    or an approved AI recommendation) directly to the matching Task
    object in the session's meeting history. Returns True if a task
    was found and updated, False otherwise.

    This is the ONE place in the app that mutates a Task after
    creation — everywhere else in this module treats extractions as
    read-only input. Task has validate_assignment enabled (see
    models/schemas.py), so an invalid update (e.g. a bad priority
    value) raises pydantic.ValidationError here rather than silently
    corrupting state; callers should catch that and surface it as a
    normal error state, not let it crash the app.
    """
    task = find_task(extractions, task_id)
    if task is None:
        return False
    for field_name, value in updates.items():
        setattr(task, field_name, value)
    return True


def unassigned_tasks_for_prompt(df: pd.DataFrame) -> list[dict]:
    """
    Converts currently-unassigned, non-Done tasks into the plain
    dict/record shape gemini_service.recommend_assignments expects to
    inject into its prompt. Deliberately excludes internal-only
    columns (meeting_id, confidence) that aren't useful evidence for
    an assignment decision.
    """
    if df.empty:
        return []
    subset = df[(df["owner"] == "Unassigned") & (df["status"] != "Done")]
    if subset.empty:
        return []
    records = subset[["id", "description", "priority", "deadline", "source_excerpt"]].copy()
    records = records.rename(columns={"id": "task_id"})
    records["deadline"] = records["deadline"].apply(lambda d: d.date().isoformat() if pd.notna(d) else None)
    return records.to_dict("records")


def workload_context_for_prompt(df: pd.DataFrame, capacity: int = 5) -> list[dict]:
    """
    Converts utilization_by_owner's output into the plain record shape
    gemini_service.recommend_assignments expects as workload evidence.
    """
    util_df = utilization_by_owner(df, capacity=capacity)
    if util_df.empty:
        return []
    return util_df.to_dict("records")
