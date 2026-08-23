"""
End-to-end wiring tests for app.py and cross-component integration.

Mocks the full streamlit API surface app.py touches (not a real
Streamlit runtime — see the module docstring pattern used throughout
tests/ for why that distinction matters) plus google.genai/pydantic,
and actually calls app.main() and components.ai_panel.render_ai_panel()
to catch wiring bugs (wrong function names, mismatched signatures)
that no amount of py_compile checking would catch.

Also verifies the single most important behavioral guarantee in the
project: an AI assignment recommendation only changes a task's owner
after explicit human approval, and a rejection leaves it untouched.

Run with: python3 tests/test_app_wiring.py
"""

from __future__ import annotations

import os
import sys
import types as _types
from dataclasses import asdict, dataclass, field
from datetime import date
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _FakeSessionState(dict):
    def setdefault(self, key, default):
        return dict.setdefault(self, key, default)

    def get(self, key, default=None):
        return dict.get(self, key, default)


class _Ctx:
    def __call__(self, *a, **kw):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __getattr__(self, name):
        return _Ctx()

    def __bool__(self):
        return False


def _install_mocks(log, button_selector):
    fake_st = _types.ModuleType("streamlit")
    fake_st.session_state = _FakeSessionState()
    fake_st.secrets = _FakeSessionState()
    fake_st.cache_resource = lambda *a, **kw: (lambda f: f)
    fake_st.set_page_config = lambda **kw: None
    fake_st.markdown = lambda *a, **kw: None
    fake_st.divider = lambda *a, **kw: None
    fake_st.info = lambda *a, **kw: log.append(("info", a[0] if a else ""))
    fake_st.success = lambda *a, **kw: log.append(("success", a[0] if a else ""))
    fake_st.error = lambda *a, **kw: log.append(("error", a[0] if a else ""))
    fake_st.warning = lambda *a, **kw: log.append(("warning", a[0] if a else ""))
    fake_st.caption = lambda *a, **kw: None
    fake_st.subheader = lambda *a, **kw: None
    fake_st.write = lambda *a, **kw: None
    fake_st.json = lambda *a, **kw: None
    fake_st.metric = lambda *a, **kw: None
    fake_st.dataframe = lambda *a, **kw: None
    fake_st.bar_chart = lambda *a, **kw: None
    fake_st.text_input = lambda *a, **kw: ""
    fake_st.text_area = lambda *a, **kw: ""
    fake_st.form_submit_button = lambda *a, **kw: False
    fake_st.file_uploader = lambda *a, **kw: None
    fake_st.rerun = lambda: log.append(("rerun", ""))

    def _button(label, *a, **kw):
        if button_selector["value"] is None:
            return False
        return kw.get("key") == button_selector["value"] or label == button_selector["value"]

    fake_st.button = _button
    fake_st.columns = lambda spec: tuple(_Ctx() for _ in range(spec if isinstance(spec, int) else len(spec)))
    fake_st.tabs = lambda labels: tuple(_Ctx() for _ in labels)
    fake_st.sidebar = _Ctx()
    fake_st.form = lambda *a, **kw: _Ctx()
    fake_st.expander = lambda *a, **kw: _Ctx()
    fake_st.container = lambda *a, **kw: _Ctx()
    fake_st.spinner = lambda *a, **kw: _Ctx()

    class _ColConfig:
        def __getattr__(self, name):
            return lambda *a, **kw: {"_col_config": name}

    fake_st.column_config = _ColConfig()
    fake_st.data_editor = lambda df, **kw: df  # pass-through: simulates "no edits made"
    sys.modules["streamlit"] = fake_st

    google_pkg = _types.ModuleType("google")
    genai_mod = _types.ModuleType("google.genai")
    types_mod = _types.ModuleType("google.genai.types")

    class _FakeSchemaObj:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    types_mod.FunctionDeclaration = _FakeSchemaObj
    types_mod.Tool = _FakeSchemaObj
    types_mod.GenerateContentConfig = _FakeSchemaObj

    class _FakePart:
        @staticmethod
        def from_bytes(data, mime_type):
            return _FakeSchemaObj(data=data, mime_type=mime_type)

    types_mod.Part = _FakePart

    class _FakeClient:
        def __init__(self, api_key=None):
            self.api_key = api_key

    genai_mod.Client = _FakeClient
    genai_mod.types = types_mod
    google_pkg.genai = genai_mod
    sys.modules["google"] = google_pkg
    sys.modules["google.genai"] = genai_mod
    sys.modules["google.genai.types"] = types_mod

    fake_pydantic = _types.ModuleType("pydantic")

    class _FakeValidationError(Exception):
        pass

    fake_pydantic.ValidationError = _FakeValidationError
    sys.modules["pydantic"] = fake_pydantic

    class Priority(str, Enum):
        HIGH = "High"
        MEDIUM = "Medium"
        LOW = "Low"

    class TaskStatus(str, Enum):
        OPEN = "Open"
        IN_PROGRESS = "In Progress"
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

        def model_dump(self):  # mirrors the real pydantic BaseModel API
            return asdict(self)

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

    os.environ["GEMINI_API_KEY"] = "test-key-for-offline-tests"
    return Task, MeetingExtraction, AssignmentRecommendation, Priority, TaskStatus


def run_all_tests():
    log = []
    button_selector = {"value": None}
    Task, MeetingExtraction, AssignmentRecommendation, Priority, TaskStatus = _install_mocks(log, button_selector)

    failures = []

    def check(name, condition):
        if condition:
            print(f"  PASS: {name}")
        else:
            print(f"  FAIL: {name}")
            failures.append(name)

    import app
    from utils import session
    import services.gemini_service as gs
    from components import ai_panel

    # --- app.main() must not crash with an empty session ---
    session.init_session_state()
    try:
        app.main()
        check("app.main() runs clean with empty session", True)
    except Exception as e:
        check(f"app.main() runs clean with empty session (raised {e!r})", False)

    # --- app.main() must not crash with a populated dashboard ---
    session.reset_session()
    t1 = Task(id="t1", description="Send report", owner="Priya", priority=Priority.HIGH, source_excerpt="x")
    t2 = Task(id="t2", description="Book the venue", owner="Unassigned", priority=Priority.MEDIUM, source_excerpt="y")
    m1 = MeetingExtraction(meeting_id="meeting_001", meeting_title="Sync", meeting_date=date(2026, 8, 22), tasks=[t1, t2])
    session.add_meeting(m1, prev_kpi_snapshot={"total_tasks": 0, "unassigned_tasks": 0, "overdue_tasks": 0, "active_owners": 0})
    try:
        app.main()
        check("app.main() runs clean with populated dashboard", True)
    except Exception as e:
        check(f"app.main() runs clean with populated dashboard (raised {e!r})", False)

    # --- AI recommendation -> approval -> real mutation ---
    def fake_recommend(unassigned_tasks, workload_context, known_team_members, as_of):
        rec = AssignmentRecommendation(task_id="t2", recommended_owner="Priya", rationale="lowest workload", confidence=0.75)
        return [rec], None, [{"attempt": 0, "tool_call_args": {}}]

    gs.recommend_assignments = fake_recommend
    button_selector["value"] = "Generate Assignment Recommendations"
    ai_panel.render_ai_panel()
    pending = session.get_valid_pending_recommendations()
    check("recommendation generated and stored as pending", len(pending) == 1 and pending[0]["task_id"] == "t2")

    button_selector["value"] = "approve_t2"
    ai_panel.render_ai_panel()
    check("HUMAN-IN-THE-LOOP: approval applies the real task mutation", t2.owner == "Priya")
    check("approved recommendation no longer pending", session.get_valid_pending_recommendations() == [])

    # --- AI recommendation -> rejection -> NO mutation ---
    t2.owner = "Unassigned"

    def fake_recommend2(unassigned_tasks, workload_context, known_team_members, as_of):
        return [AssignmentRecommendation(task_id="t2", recommended_owner="Sam", rationale="x", confidence=0.5)], None, []

    gs.recommend_assignments = fake_recommend2
    button_selector["value"] = "Generate Assignment Recommendations"
    ai_panel.render_ai_panel()
    button_selector["value"] = "reject_t2"
    ai_panel.render_ai_panel()
    check("HUMAN-IN-THE-LOOP: rejection leaves ownership unchanged", t2.owner == "Unassigned")

    print()
    if failures:
        print(f"{len(failures)} TEST(S) FAILED: {failures}")
        sys.exit(1)
    print("ALL APP WIRING TESTS PASSED")


if __name__ == "__main__":
    run_all_tests()
