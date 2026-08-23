"""
Integration test for components/input_panel.py's _handle_submission().

Mocks: streamlit (session_state + cache_resource), google.genai (so
gemini_service/tools import without the real SDK installed), and
models.schemas (lightweight dataclass stand-ins, since pydantic isn't
installed in this sandbox). gemini_service.extract_tasks itself is
monkeypatched after import to avoid any real network call.

Uses the REAL utils/session.py and utils/data_pipeline.py — this test
exists specifically to catch integration bugs between input_panel,
session state, and the Pandas layer (e.g. incorrect KPI snapshot
timing, team roster not propagating across meetings).

Run with: python3 tests/test_input_panel.py
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
    # --- fake streamlit ---
    class FakeSessionState(dict):
        def setdefault(self, key, default):
            return dict.setdefault(self, key, default)

        def get(self, key, default=None):
            return dict.get(self, key, default)

    fake_st = _types.ModuleType("streamlit")
    fake_st.session_state = FakeSessionState()
    fake_st.cache_resource = lambda *a, **kw: (lambda f: f)  # no-op decorator
    sys.modules["streamlit"] = fake_st

    # --- fake google.genai ---
    google_pkg = _types.ModuleType("google")
    genai_mod = _types.ModuleType("google.genai")
    types_mod = _types.ModuleType("google.genai.types")

    class _FakeSchemaObj:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    types_mod.FunctionDeclaration = _FakeSchemaObj
    types_mod.Tool = _FakeSchemaObj
    types_mod.GenerateContentConfig = _FakeSchemaObj

    class FakeClient:
        def __init__(self, api_key=None):
            self.api_key = api_key

    genai_mod.Client = FakeClient
    genai_mod.types = types_mod
    google_pkg.genai = genai_mod

    sys.modules["google"] = google_pkg
    sys.modules["google.genai"] = genai_mod
    sys.modules["google.genai.types"] = types_mod

    # --- fake pydantic (only ValidationError is imported directly) ---
    fake_pydantic = _types.ModuleType("pydantic")

    class FakeValidationError(Exception):
        pass

    fake_pydantic.ValidationError = FakeValidationError
    sys.modules["pydantic"] = fake_pydantic

    # --- lightweight models.schemas stand-ins ---
    class Priority(str, Enum):
        HIGH = "High"
        MEDIUM = "Medium"
        LOW = "Low"

    class TaskStatus(str, Enum):
        OPEN = "Open"
        DONE = "Done"

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

    @dataclass
    class AssignmentRecommendation:
        task_id: str
        recommended_owner: str
        rationale: str
        confidence: float
        alternative_candidate: str = None

    mod = _types.ModuleType("models.schemas")
    mod.Task = Task
    mod.MeetingExtraction = MeetingExtraction
    mod.ExtractionError = ExtractionError
    mod.AssignmentRecommendation = AssignmentRecommendation
    mod.Priority = Priority
    mod.TaskStatus = TaskStatus
    sys.modules["models.schemas"] = mod
    pkg = _types.ModuleType("models")
    pkg.schemas = mod
    sys.modules["models"] = pkg

    return Task, MeetingExtraction, ExtractionError, Priority, TaskStatus


def run_all_tests():
    Task, MeetingExtraction, ExtractionError, Priority, TaskStatus = _install_mocks()

    import utils.session as session
    from components import input_panel

    failures = []

    def check(name, condition):
        if condition:
            print(f"  PASS: {name}")
        else:
            print(f"  FAIL: {name}")
            failures.append(name)

    session.init_session_state()

    # Capture what team_members roster gets passed to extract_tasks on
    # each call, to verify roster propagation across meetings.
    calls = []

    def fake_extract_tasks_success(transcript, meeting_id, meeting_title, meeting_date, team_members=None):
        calls.append({"meeting_id": meeting_id, "team_members": team_members, "title": meeting_title})
        task = Task(id=f"{meeting_id}_t1", description="Do the thing", owner="Priya",
                    priority=Priority.HIGH, status=TaskStatus.OPEN, confidence=0.9)
        extraction = MeetingExtraction(
            meeting_id=meeting_id, meeting_title=meeting_title, meeting_date=meeting_date,
            tasks=[task], summary="summary", raw_transcript_length=len(transcript),
        )
        return extraction, None, [{"attempt": 0, "tool_call_args": {"tasks": []}}]

    input_panel.gemini_service.extract_tasks = fake_extract_tasks_success

    # --- Test 1: first submission, no title given -> default title used ---
    result, error, trace = input_panel._handle_submission(
        "A" * 50, ""  # empty title -> should default
    )
    check("first submission succeeds", error is None and result is not None)
    check("meeting added to session history", session.get_meeting_count() == 1)
    check("default title applied when blank", result.meeting_title.startswith("Meeting on "))
    check("first call has empty team roster (no prior meetings)", calls[0]["team_members"] == [])
    check("trace stored in session", len(session.get_last_trace()) == 1)

    # --- Test 2: prev_kpi_snapshot captured BEFORE this meeting's tasks exist ---
    snap = session.get_prev_kpi_snapshot()
    check("prev_kpi_snapshot reflects state BEFORE meeting 1 (0 tasks)", snap["total_tasks"] == 0)

    # --- Test 3: second submission -> team roster now includes Priya ---
    result2, error2, trace2 = input_panel._handle_submission("B" * 50, "Second Sync")
    check("second submission succeeds", error2 is None)
    check("meeting count now 2", session.get_meeting_count() == 2)
    check("second call's roster includes owner from first meeting", "Priya" in calls[1]["team_members"])
    check("explicit title respected when provided", result2.meeting_title == "Second Sync")

    snap2 = session.get_prev_kpi_snapshot()
    check("prev_kpi_snapshot before meeting 2 reflects meeting 1's 1 task", snap2["total_tasks"] == 1)

    # --- Test 4: error path does not add a meeting, but does consume an id ---
    def fake_extract_tasks_error(transcript, meeting_id, meeting_title, meeting_date, team_members=None):
        return None, ExtractionError(stage="api_call", message="simulated failure"), [{"attempt": 0, "error": "x"}]

    input_panel.gemini_service.extract_tasks = fake_extract_tasks_error
    count_before = session.get_meeting_count()
    result3, error3, trace3 = input_panel._handle_submission("C" * 50, "Will Fail")
    check("error path returns no result", result3 is None)
    check("error path returns the error", error3 is not None and error3.message == "simulated failure")
    check("meeting count unchanged after error", session.get_meeting_count() == count_before)
    check("session records the error", session.get_last_error() is not None)

    # --- Test 5: audio submission path mirrors text path ---
    def fake_extract_audio_success(audio_bytes, mime_type, meeting_id, meeting_title, meeting_date, team_members=None):
        calls.append({"meeting_id": meeting_id, "team_members": team_members, "title": meeting_title, "mime_type": mime_type})
        task = Task(id=f"{meeting_id}_t1", description="Voice task", owner="Sam",
                    priority=Priority.MEDIUM, status=TaskStatus.OPEN, confidence=0.85)
        extraction = MeetingExtraction(
            meeting_id=meeting_id, meeting_title=meeting_title, meeting_date=meeting_date,
            tasks=[task], summary="voice summary", raw_transcript_length=len(audio_bytes),
        )
        return extraction, None, [{"attempt": 0, "tool_call_args": {"tasks": []}}]

    input_panel.gemini_service.extract_tasks_from_audio = fake_extract_audio_success
    count_before_audio = session.get_meeting_count()
    result5, error5, trace5 = input_panel._handle_audio_submission(b"fake audio bytes", "audio/wav", "")
    check("audio submission succeeds", error5 is None and result5 is not None)
    check("audio meeting added to session history", session.get_meeting_count() == count_before_audio + 1)
    check("audio default title uses 'Voice memo on' prefix", result5.meeting_title.startswith("Voice memo on"))
    check("audio call received correct mime type", calls[-1]["mime_type"] == "audio/wav")

    print()
    if failures:
        print(f"{len(failures)} TEST(S) FAILED: {failures}")
        sys.exit(1)
    else:
        print("ALL INPUT PANEL INTEGRATION TESTS PASSED")


if __name__ == "__main__":
    run_all_tests()
