"""
Unit tests for utils/session.py.

IMPORTANT SCOPE NOTE: these tests run against a lightweight mock of
st.session_state (a plain dict with setdefault), not real Streamlit.
They verify the *logic* in session.py — state transitions, the
stale-recommendation guard, reset safety — in isolation. They do NOT
verify Streamlit-specific runtime behavior (actual widget rerun
timing, real st.form submission semantics, real st.session_state
quirks). That requires running `streamlit run app.py` by hand; do not
treat a green run of this file as "the app has been runtime-verified."

Run with: python3 tests/test_session.py
(kept dependency-free/pytest-optional so it runs in environments
where pydantic/streamlit aren't installed — see the mock/stand-in
setup below, which mirrors models/schemas.py's shape without
requiring pydantic itself.)
"""

from __future__ import annotations

import os
import sys
import types as _types
from dataclasses import dataclass, field
from datetime import date
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _install_mocks():
    class FakeSessionState(dict):
        def setdefault(self, key, default):
            return dict.setdefault(self, key, default)

    fake_st = _types.ModuleType("streamlit")
    fake_st.session_state = FakeSessionState()
    sys.modules["streamlit"] = fake_st

    class Priority(str, Enum):
        HIGH = "High"
        MEDIUM = "Medium"
        LOW = "Low"

    class TaskStatus(str, Enum):
        OPEN = "Open"

    @dataclass
    class Task:
        id: str
        description: str
        owner: str = "Unassigned"
        deadline: date = None
        priority: Priority = Priority.MEDIUM
        status: TaskStatus = TaskStatus.OPEN
        confidence: float = 0.8
        source_excerpt: str = ""

    @dataclass
    class MeetingExtraction:
        meeting_id: str
        meeting_title: str
        meeting_date: date
        tasks: list = field(default_factory=list)
        summary: str = ""
        raw_transcript_length: int = 0

    @dataclass
    class ExtractionError:
        stage: str
        message: str
        raw_response: str = None

    mod = _types.ModuleType("models.schemas")
    mod.Task = Task
    mod.MeetingExtraction = MeetingExtraction
    mod.ExtractionError = ExtractionError
    sys.modules["models.schemas"] = mod
    pkg = _types.ModuleType("models")
    pkg.schemas = mod
    sys.modules["models"] = pkg

    return Task, MeetingExtraction, ExtractionError


def run_all_tests():
    Task, MeetingExtraction, ExtractionError = _install_mocks()
    import utils.session as session

    failures = []

    def check(name, condition):
        if condition:
            print(f"  PASS: {name}")
        else:
            print(f"  FAIL: {name}")
            failures.append(name)

    session.init_session_state()

    id1 = session.next_meeting_id()
    m1 = MeetingExtraction(meeting_id=id1, meeting_title="Sync", meeting_date=date(2026, 8, 22), tasks=[])
    session.add_meeting(m1, prev_kpi_snapshot={"total_tasks": 0})
    check("meeting count after first add == 1", session.get_meeting_count() == 1)

    session.set_selected_meeting_id(id1)
    session.set_selected_task_id("t1")
    check("selected meeting id preserved", session.get_selected_meeting_id() == id1)
    check("selected task id preserved", session.get_selected_task_id() == "t1")

    recs = [
        {"task_id": "t1", "recommended_owner": "Priya", "rationale": "lowest workload", "confidence": 0.8},
        {"task_id": "t2", "recommended_owner": "Sam", "rationale": "skill match", "confidence": 0.7},
    ]
    session.set_pending_recommendations(recs)
    valid = session.get_valid_pending_recommendations()
    check("recommendations stamped as pending", all(r["status"] == "pending" for r in valid))
    check("recommendations stamped with correct meeting count", all(r["generated_at_meeting_count"] == 1 for r in valid))

    check("approve t1 succeeds", session.approve_recommendation("t1") is True)
    check("reject t2 succeeds", session.reject_recommendation("t2") is True)
    check("no pending recommendations remain", session.get_valid_pending_recommendations() == [])
    check("re-approving decided task fails", session.approve_recommendation("t1") is False)

    # Stale-guard: the critical invariant
    id2 = session.next_meeting_id()
    m2 = MeetingExtraction(meeting_id=id2, meeting_title="Sync2", meeting_date=date(2026, 8, 23), tasks=[])
    session.set_pending_recommendations([{"task_id": "t3", "recommended_owner": "Sam", "rationale": "x", "confidence": 0.6}])
    check("new recommendation valid before new meeting added", len(session.get_valid_pending_recommendations()) == 1)
    session.add_meeting(m2, prev_kpi_snapshot={"total_tasks": 0})
    check("STALE-GUARD: old recommendation invalid after new meeting", session.get_valid_pending_recommendations() == [])
    check("STALE-GUARD: stale recommendation cannot be approved", session.approve_recommendation("t3") is False)

    session.reset_session()
    check("reset clears meetings", session.get_meetings() == [])
    check("reset restores default view", session.get_active_view() == "dashboard")
    check("reset clears selection", session.get_selected_meeting_id() is None)
    check("reset clears recommendations", session.get_valid_pending_recommendations() == [])

    print()
    if failures:
        print(f"{len(failures)} TEST(S) FAILED: {failures}")
        sys.exit(1)
    else:
        print("ALL SESSION TESTS PASSED")


if __name__ == "__main__":
    run_all_tests()
