"""
Session state architecture for Synapse.

Streamlit reruns the entire script on every interaction, so anything
that must survive across reruns has to live in st.session_state rather
than a local variable. This module is the single place that:
  1. Defines every state key as a constant (no other file hardcodes a
     string key and risks a silent typo-based bug).
  2. Owns all reads/writes to those keys — app.py and components never
     call st.session_state[...] directly.

State is organized into four explicit categories, because conflating
them is exactly how apps end up with "why did that change?" bugs:

  DATA STATE      — the actual business record (meeting history).
                     Source of truth for the Pandas layer.
  UI STATE         — what the user is currently looking at/selected
                     (active tab, selected task/meeting). Purely
                     cosmetic — losing it costs convenience, not data.
  AI STATE         — the last Gemini call's raw outcome (result/error/
                     trace). Transient by nature — overwritten on
                     every new call, never accumulated.
  APPROVAL STATE   — AI recommendations awaiting a human decision.
                     Explicitly stale-guarded (see below) so a
                     recommendation generated against an older task
                     list can never be silently approved against a
                     task list that has since changed underneath it.

This module is intentionally "dumb": no Gemini calls, no Pandas math,
no schema-specific business rules beyond the stale-guard invariant
below. That keeps it unit-testable in isolation with a mocked
st.session_state (see tests/test_session.py) — which is NOT the same
as verifying it inside a real Streamlit runtime. Mocked tests catch
logic bugs in this module; they cannot catch Streamlit-specific
behavior (widget rerun timing, actual st.form semantics). Real
runtime verification happens separately, via `streamlit run app.py`.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

from models.schemas import ExtractionError, MeetingExtraction

# ============================================================
# DATA STATE — the business record. Source of truth for
# utils/data_pipeline.py. Only ever appended to, never mutated
# in place (see add_meeting).
# ============================================================

MEETINGS = "synapse_meetings"                  # list[MeetingExtraction]
MEETING_COUNTER = "synapse_meeting_counter"    # int, monotonic, used for ids

# ============================================================
# UI STATE — cosmetic/navigation only. Never read by the data
# pipeline or Gemini layer. Safe to reset without losing data.
# ============================================================

ACTIVE_VIEW = "synapse_active_view"            # "dashboard" | "history"
SELECTED_MEETING_ID = "synapse_selected_meeting_id"  # Optional[str]
SELECTED_TASK_ID = "synapse_selected_task_id"        # Optional[str]

# ============================================================
# AI STATE — outcome of the most recent Gemini call. Fully
# overwritten on every call (never accumulated), since it
# describes "what just happened," not history.
# ============================================================

LAST_EXTRACTION = "synapse_last_extraction"    # Optional[MeetingExtraction]
LAST_ERROR = "synapse_last_error"              # Optional[ExtractionError]
LAST_TRACE = "synapse_last_trace"              # list[dict]
PREV_KPI_SNAPSHOT = "synapse_prev_kpi_snapshot"  # Optional[dict] — KPI dict
                                                   # captured just before the
                                                   # most recent meeting was
                                                   # added, so the dashboard
                                                   # can show real deltas.

# ============================================================
# APPROVAL STATE — AI recommendations awaiting an explicit
# human decision. Each recommendation dict carries:
#   task_id, recommended_owner, rationale, confidence,
#   alternative_candidate, status ("pending"|"approved"|"rejected"),
#   generated_at_meeting_count (int — stale-guard, see below)
#
# STALE-GUARD INVARIANT: a recommendation is only valid if
# generated_at_meeting_count == the current number of meetings.
# If the user processes a new meeting after recommendations were
# generated, the task list/workload picture they were reasoned
# over is out of date — get_valid_pending_recommendations()
# filters these out rather than letting a UI accidentally apply
# a decision made against stale context.
# ============================================================

PENDING_RECOMMENDATIONS = "synapse_pending_recommendations"  # list[dict]
TASK_EDITS = "synapse_task_edits"              # dict[str, dict] — pending
                                                 # data_editor overrides not
                                                 # yet committed to a task


def init_session_state() -> None:
    """
    Idempotent initializer — safe to call at the top of every rerun.
    Uses setdefault semantics so it never clobbers state a previous
    rerun already built up. Every key declared above MUST have a
    default here — an undeclared key is a bug, not a case to handle
    defensively elsewhere with getattr/.get() fallbacks scattered
    around the codebase.
    """
    st.session_state.setdefault(MEETINGS, [])
    st.session_state.setdefault(MEETING_COUNTER, 0)

    st.session_state.setdefault(ACTIVE_VIEW, "dashboard")
    st.session_state.setdefault(SELECTED_MEETING_ID, None)
    st.session_state.setdefault(SELECTED_TASK_ID, None)

    st.session_state.setdefault(LAST_EXTRACTION, None)
    st.session_state.setdefault(LAST_ERROR, None)
    st.session_state.setdefault(LAST_TRACE, [])
    st.session_state.setdefault(PREV_KPI_SNAPSHOT, None)

    st.session_state.setdefault(PENDING_RECOMMENDATIONS, [])
    st.session_state.setdefault(TASK_EDITS, {})


def reset_session() -> None:
    """
    Full, safe reset back to init defaults — e.g. for a 'Start Fresh
    Demo' button. Deliberately reuses init_session_state's defaults
    (by clearing then re-initializing) rather than duplicating the
    default values here, so the two can never drift out of sync.
    """
    for key in (
        MEETINGS, MEETING_COUNTER, ACTIVE_VIEW, SELECTED_MEETING_ID,
        SELECTED_TASK_ID, LAST_EXTRACTION, LAST_ERROR, LAST_TRACE,
        PREV_KPI_SNAPSHOT, PENDING_RECOMMENDATIONS, TASK_EDITS,
    ):
        if key in st.session_state:
            del st.session_state[key]
    init_session_state()


# ------------------------------------------------------------
# DATA STATE accessors
# ------------------------------------------------------------

def next_meeting_id() -> str:
    """Increments the session's meeting counter and returns a stable
    id (e.g. 'meeting_003'). Must be called BEFORE invoking Gemini,
    since extract_tasks() requires a meeting_id up front."""
    st.session_state[MEETING_COUNTER] += 1
    return f"meeting_{st.session_state[MEETING_COUNTER]:03d}"


def get_meetings() -> list[MeetingExtraction]:
    return st.session_state[MEETINGS]


def get_meeting_count() -> int:
    return len(st.session_state[MEETINGS])


def add_meeting(extraction: MeetingExtraction, prev_kpi_snapshot: Optional[dict] = None) -> None:
    """
    Appends a newly-validated extraction to the session's history
    (append-only — existing meetings are never mutated here) and
    records the KPI snapshot from just before it was added.

    Any pending recommendations are implicitly invalidated by this
    call, since the task list they reasoned over has now changed —
    get_valid_pending_recommendations() enforces this on read, so
    nothing needs to be explicitly cleared here.
    """
    st.session_state[MEETINGS].append(extraction)
    st.session_state[LAST_EXTRACTION] = extraction
    st.session_state[LAST_ERROR] = None
    st.session_state[PREV_KPI_SNAPSHOT] = prev_kpi_snapshot


def has_meetings() -> bool:
    return len(st.session_state[MEETINGS]) > 0


# ------------------------------------------------------------
# UI STATE accessors
# ------------------------------------------------------------

def get_active_view() -> str:
    return st.session_state[ACTIVE_VIEW]


def set_active_view(view: str) -> None:
    st.session_state[ACTIVE_VIEW] = view


def get_selected_meeting_id() -> Optional[str]:
    return st.session_state[SELECTED_MEETING_ID]


def set_selected_meeting_id(meeting_id: Optional[str]) -> None:
    st.session_state[SELECTED_MEETING_ID] = meeting_id


def get_selected_task_id() -> Optional[str]:
    return st.session_state[SELECTED_TASK_ID]


def set_selected_task_id(task_id: Optional[str]) -> None:
    st.session_state[SELECTED_TASK_ID] = task_id


# ------------------------------------------------------------
# AI STATE accessors
# ------------------------------------------------------------

def set_last_result(
    extraction: Optional[MeetingExtraction] = None,
    error: Optional[ExtractionError] = None,
    trace: Optional[list[dict]] = None,
) -> None:
    """Records the outcome of the most recent extract_tasks() call.
    Exactly one of extraction/error should be set by the caller —
    this function doesn't enforce that itself, it just stores what
    it's given."""
    st.session_state[LAST_EXTRACTION] = extraction
    st.session_state[LAST_ERROR] = error
    st.session_state[LAST_TRACE] = trace or []


def get_last_error() -> Optional[ExtractionError]:
    return st.session_state[LAST_ERROR]


def get_last_trace() -> list[dict]:
    return st.session_state[LAST_TRACE]


def clear_last_error() -> None:
    st.session_state[LAST_ERROR] = None


def get_prev_kpi_snapshot() -> Optional[dict]:
    return st.session_state[PREV_KPI_SNAPSHOT]


# ------------------------------------------------------------
# APPROVAL STATE accessors
# ------------------------------------------------------------

def set_pending_recommendations(recommendations: list[dict]) -> None:
    """
    Stores newly-generated recommendations, stamping each with the
    current meeting count and an explicit 'pending' status. Callers
    should NOT set 'status' or 'generated_at_meeting_count' on the
    dicts they pass in — this function owns those two fields so the
    stale-guard invariant can't be bypassed by a forgetful caller.
    """
    current_count = get_meeting_count()
    stamped = []
    for rec in recommendations:
        rec = dict(rec)
        rec["status"] = "pending"
        rec["generated_at_meeting_count"] = current_count
        stamped.append(rec)
    st.session_state[PENDING_RECOMMENDATIONS] = stamped


def get_valid_pending_recommendations() -> list[dict]:
    """
    Returns only recommendations that are still 'pending' AND were
    generated against the current meeting count. This is the ONLY
    read path a UI should use to decide what to show for approval —
    it enforces the stale-guard invariant on every call rather than
    relying on callers to remember to check.
    """
    current_count = get_meeting_count()
    return [
        rec for rec in st.session_state[PENDING_RECOMMENDATIONS]
        if rec.get("status") == "pending" and rec.get("generated_at_meeting_count") == current_count
    ]


def _update_recommendation_status(task_id: str, new_status: str) -> bool:
    """Internal helper — finds the pending recommendation for
    task_id and sets its status explicitly. Returns True if a
    matching, still-valid recommendation was found and updated."""
    current_count = get_meeting_count()
    for rec in st.session_state[PENDING_RECOMMENDATIONS]:
        if (
            rec.get("task_id") == task_id
            and rec.get("status") == "pending"
            and rec.get("generated_at_meeting_count") == current_count
        ):
            rec["status"] = new_status
            return True
    return False


def approve_recommendation(task_id: str) -> bool:
    """Marks a recommendation approved. Does NOT itself modify the
    task's owner — the caller (business logic layer) is responsible
    for applying the change to the actual Task and must check this
    returned True before doing so. This keeps 'the human approved
    it' and 'the system applied it' as two distinct, auditable
    steps, per the human-in-the-loop requirement."""
    return _update_recommendation_status(task_id, "approved")


def reject_recommendation(task_id: str) -> bool:
    return _update_recommendation_status(task_id, "rejected")


def clear_pending_recommendations() -> None:
    st.session_state[PENDING_RECOMMENDATIONS] = []
