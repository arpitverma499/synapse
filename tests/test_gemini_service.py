"""
Tests for services/gemini_service.py — specifically the shared retry
loop (_generate_with_tool), which is the riskiest, most complex logic
in the codebase and had never been exercised at runtime before this.

SCOPE NOTE: this test monkeypatches services.tools.validate_payload /
validate_recommendations with controllable stubs, rather than
exercising tools.py's real pydantic validation. That's a deliberate
boundary: pydantic isn't installed in this offline sandbox, so we
can't trigger genuine pydantic.ValidationError here. What this DOES
verify, against the real gemini_service.py code: the retry loop's
control flow (retry-on-no-tool-call, retry-on-validation-failure,
exhausting retries, transient-vs-permanent API error handling) and
that the correct ExtractionError.stage taxonomy value is produced for
each failure mode. Real pydantic validation behavior must be verified
separately once pydantic is installed (e.g. via `pip install -r
requirements.txt` and a real run).

Run with: python3 tests/test_gemini_service.py
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

        def get(self, key, default=None):
            return dict.get(self, key, default)

    fake_st = _types.ModuleType("streamlit")
    fake_st.session_state = FakeSessionState()
    fake_st.secrets = FakeSessionState()  # supports .get()
    fake_st.cache_resource = lambda *a, **kw: (lambda f: f)
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

    genai_mod.types = types_mod
    google_pkg.genai = genai_mod
    sys.modules["google"] = google_pkg
    sys.modules["google.genai"] = genai_mod
    sys.modules["google.genai.types"] = types_mod

    fake_pydantic = _types.ModuleType("pydantic")

    class FakeValidationError(Exception):
        pass

    fake_pydantic.ValidationError = FakeValidationError
    sys.modules["pydantic"] = fake_pydantic

    class Priority(str, Enum):
        HIGH = "High"

    class TaskStatus(str, Enum):
        OPEN = "Open"

    @dataclass
    class Task:
        id: str
        description: str
        owner: str = "Unassigned"

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

    return google_pkg, genai_mod, Task, MeetingExtraction, ExtractionError, FakeValidationError


# --- Fake Gemini response plumbing -----------------------------------

class _FakeFunctionCall:
    def __init__(self, name, args):
        self.name = name
        self.args = args


class _FakePart:
    def __init__(self, name=None, args=None):
        self.function_call = _FakeFunctionCall(name, args) if name else None


class _FakeCandidate:
    def __init__(self, parts):
        self.content = _types.SimpleNamespace(parts=parts) if hasattr(_types, "SimpleNamespace") else _Ns(parts=parts)


class _Ns:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _FakeResponse:
    def __init__(self, parts, text=""):
        self.candidates = [_Ns(content=_Ns(parts=parts))]
        self.text = text


def _tool_call_response(tool_name, args):
    return _FakeResponse([_FakePart(tool_name, args)])


def _no_tool_call_response():
    return _FakeResponse([], text="I think the tasks are...")


class _FakeModels:
    def __init__(self, program):
        self.program = list(program)
        self.calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1
        item = self.program.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _FakeClient:
    def __init__(self, program):
        self.models = _FakeModels(program)


def run_all_tests():
    google_pkg, genai_mod, Task, MeetingExtraction, ExtractionError, FakeValidationError = _install_mocks()
    os.environ["GEMINI_API_KEY"] = "test-key-for-offline-tests"

    import services.gemini_service as gs
    import services.tools as tools

    failures = []

    def check(name, condition):
        if condition:
            print(f"  PASS: {name}")
        else:
            print(f"  FAIL: {name}")
            failures.append(name)

    def set_program(program):
        genai_mod.Client = lambda api_key=None: _FakeClient(program)

    # --- Scenario 1: success on first attempt ---
    tools.validate_payload = lambda payload, *a, **kw: MeetingExtraction(
        meeting_id="m1", meeting_title="T", meeting_date=date.today(),
        tasks=[Task(id="t1", description="do x", owner="Sam")],
    )
    set_program([_tool_call_response(tools.TOOL_NAME, {"tasks": [], "summary": "s"})])
    result, error, trace = gs.extract_tasks("A" * 50, "m1", "T", date.today(), [])
    check("scenario1: succeeds on first attempt", error is None and result is not None)
    check("scenario1: trace has exactly 1 attempt", len(trace) == 1)

    # --- Scenario 2: no tool call, then success on retry ---
    set_program([_no_tool_call_response(), _tool_call_response(tools.TOOL_NAME, {"tasks": [], "summary": "s"})])
    result, error, trace = gs.extract_tasks("B" * 50, "m2", "T", date.today(), [])
    check("scenario2: recovers after AI_OUTPUT retry", error is None and result is not None)
    check("scenario2: trace has exactly 2 attempts", len(trace) == 2)

    # --- Scenario 3: validation fails, then succeeds on retry ---
    call_count = {"n": 0}

    def flaky_validate(payload, *a, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise FakeValidationError("bad priority enum")
        return MeetingExtraction(meeting_id="m3", meeting_title="T", meeting_date=date.today(), tasks=[])

    tools.validate_payload = flaky_validate
    set_program([
        _tool_call_response(tools.TOOL_NAME, {"tasks": [{"bad": "data"}], "summary": "s"}),
        _tool_call_response(tools.TOOL_NAME, {"tasks": [], "summary": "s"}),
    ])
    result, error, trace = gs.extract_tasks("C" * 50, "m3", "T", date.today(), [])
    check("scenario3: recovers after VALIDATION retry", error is None and result is not None)
    check("scenario3: trace has exactly 2 attempts", len(trace) == 2)

    # --- Scenario 4: exhausts retries on repeated no-tool-call -> AI_OUTPUT_ERROR ---
    set_program([_no_tool_call_response(), _no_tool_call_response(), _no_tool_call_response()])
    result, error, trace = gs.extract_tasks("D" * 50, "m4", "T", date.today(), [])
    check("scenario4: fails after exhausting retries", result is None and error is not None)
    check("scenario4: correct taxonomy stage", error.stage == "AI_OUTPUT_ERROR")
    check("scenario4: trace has exactly MAX_RETRIES+1=3 attempts", len(trace) == 3)

    # --- Scenario 5: transient API exception recovers on retry ---
    set_program([TimeoutError("network blip"), _tool_call_response(tools.TOOL_NAME, {"tasks": [], "summary": "s"})])
    tools.validate_payload = lambda payload, *a, **kw: MeetingExtraction(
        meeting_id="m5", meeting_title="T", meeting_date=date.today(), tasks=[]
    )
    result, error, trace = gs.extract_tasks("E" * 50, "m5", "T", date.today(), [])
    check("scenario5: recovers from transient API exception", error is None and result is not None)

    # --- Scenario 6: API exception exhausts retries -> AI_API_ERROR ---
    set_program([TimeoutError("x"), TimeoutError("x"), TimeoutError("x")])
    result, error, trace = gs.extract_tasks("F" * 50, "m6", "T", date.today(), [])
    check("scenario6: fails after exhausting API retries", result is None and error is not None)
    check("scenario6: correct taxonomy stage", error.stage == "AI_API_ERROR")

    # --- Scenario 7: too-short transcript -> DATA_ERROR, zero API calls made ---
    set_program([])  # empty program — if this raises IndexError, the code wrongly called the API
    result, error, trace = gs.extract_tasks("short", "m7", "T", date.today(), [])
    check("scenario7: too-short transcript rejected", result is None and error is not None)
    check("scenario7: correct taxonomy stage", error.stage == "DATA_ERROR")
    check("scenario7: no API call attempted", trace == [])

    # --- Scenario 8: missing API key -> AI_API_ERROR, zero attempts ---
    del os.environ["GEMINI_API_KEY"]
    result, error, trace = gs.extract_tasks("G" * 50, "m8", "T", date.today(), [])
    check("scenario8: missing key produces AI_API_ERROR", error is not None and error.stage == "AI_API_ERROR")
    check("scenario8: no attempts recorded", trace == [])
    os.environ["GEMINI_API_KEY"] = "test-key-for-offline-tests"

    # --- Scenario 9: recommend_assignments rejects an unknown task_id (business-rule ValueError path) ---
    def rec_validate(payload):
        recs_raw = payload.get("recommendations", [])
        objs = [
            _Ns(task_id=r["task_id"], recommended_owner=r["recommended_owner"],
                rationale=r["rationale"], confidence=r["confidence"],
                alternative_candidate=r.get("alternative_candidate"))
            for r in recs_raw
        ]
        unknown = [o.task_id for o in objs if o.task_id not in {"t1"}]
        if unknown:
            raise ValueError(f"unknown task ids: {unknown}")
        return objs

    tools.validate_recommendations = rec_validate
    set_program([
        tools_response := _tool_call_response(
            tools.RECOMMEND_TOOL_NAME,
            {"recommendations": [{"task_id": "t999", "recommended_owner": "Sam", "rationale": "x", "confidence": 0.5}]},
        ),
        _tool_call_response(
            tools.RECOMMEND_TOOL_NAME,
            {"recommendations": [{"task_id": "t1", "recommended_owner": "Sam", "rationale": "x", "confidence": 0.5}]},
        ),
    ])
    result, error, trace = gs.recommend_assignments(
        unassigned_tasks=[{"task_id": "t1", "description": "do x"}],
        workload_context=[{"owner": "Sam", "open_tasks": 1, "utilization_pct": 20}],
        known_team_members=["Sam"],
        as_of=date.today(),
    )
    check("scenario9: rejects unknown task_id then recovers", error is None and result is not None)
    check("scenario9: recommendation references the valid task", result[0].task_id == "t1")

    # --- Scenario 10: recommend_assignments with no unassigned tasks -> DATA_ERROR ---
    result, error, trace = gs.recommend_assignments([], [], ["Sam"], date.today())
    check("scenario10: empty unassigned list -> DATA_ERROR", error is not None and error.stage == "DATA_ERROR")

    print()
    if failures:
        print(f"{len(failures)} TEST(S) FAILED: {failures}")
        sys.exit(1)
    else:
        print("ALL GEMINI SERVICE RETRY-LOOP TESTS PASSED")


if __name__ == "__main__":
    run_all_tests()
