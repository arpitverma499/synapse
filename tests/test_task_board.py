"""
Tests for components/task_board.py's compute_edits() — the pure diff
function that decides what actually changed in a data_editor session.
Uses real pandas (available in this sandbox) with mocked streamlit/
pydantic/models.schemas so the module imports without those packages.

Run with: python3 tests/test_task_board.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types as _types
from datetime import date

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _install_mocks_and_load():
    fake_st = _types.ModuleType("streamlit")
    fake_st.session_state = {}
    sys.modules["streamlit"] = fake_st

    fake_pydantic = _types.ModuleType("pydantic")

    class FakeValidationError(Exception):
        pass

    fake_pydantic.BaseModel = object
    fake_pydantic.Field = lambda *a, **kw: None
    fake_pydantic.field_validator = lambda *a, **kw: (lambda f: f)
    fake_pydantic.ValidationError = FakeValidationError
    sys.modules["pydantic"] = fake_pydantic

    mod = _types.ModuleType("models.schemas")
    mod.MeetingExtraction = object
    mod.Task = object
    mod.ExtractionError = object
    mod.AssignmentRecommendation = object
    sys.modules["models.schemas"] = mod
    pkg = _types.ModuleType("models")
    pkg.schemas = mod
    sys.modules["models"] = pkg

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("task_board", os.path.join(root, "components", "task_board.py"))
    tb = importlib.util.module_from_spec(spec)
    sys.modules["task_board"] = tb
    spec.loader.exec_module(tb)
    return tb


def run_all_tests():
    tb = _install_mocks_and_load()
    failures = []

    def check(name, condition):
        if condition:
            print(f"  PASS: {name}")
        else:
            print(f"  FAIL: {name}")
            failures.append(name)

    orig = pd.DataFrame([
        {"id": "t1", "description": "x", "owner": "Priya", "priority": "High", "status": "Open", "deadline": pd.Timestamp("2026-08-25"), "meeting_title": "M"},
        {"id": "t2", "description": "y", "owner": "Unassigned", "priority": "Medium", "status": "Open", "deadline": pd.NaT, "meeting_title": "M"},
    ])

    def make_edited(t1_deadline, t1_owner="Priya", t1_priority="High", t1_status="Open", t2_owner="Unassigned", t2_deadline=None):
        return pd.DataFrame([
            {"id": "t1", "description": "x", "owner": t1_owner, "priority": t1_priority, "status": t1_status, "deadline": t1_deadline, "meeting_title": "M"},
            {"id": "t2", "description": "y", "owner": t2_owner, "priority": "Medium", "status": "Open", "deadline": t2_deadline, "meeting_title": "M"},
        ])

    check("no-op round trip (date vs Timestamp) -> no changes",
          tb.compute_edits(orig, make_edited(date(2026, 8, 25))) == {})

    check("single field change detected",
          tb.compute_edits(orig, make_edited(date(2026, 8, 25), t2_owner="Sam")) == {"t2": {"owner": "Sam"}})

    check("real deadline change detected",
          tb.compute_edits(orig, make_edited(date(2026, 8, 30)))["t1"]["deadline"] == date(2026, 8, 30))

    multi = tb.compute_edits(orig, make_edited(date(2026, 8, 25), t1_priority="Low", t1_status="Done"))
    check("multiple field changes on one row", multi == {"t1": {"priority": "Low", "status": "Done"}})

    cleared = tb.compute_edits(orig, make_edited(None, t2_deadline=date(2026, 9, 1)))
    check("deadline cleared / newly set", cleared == {"t1": {"deadline": None}, "t2": {"deadline": date(2026, 9, 1)}})

    check("empty original_df is safe", tb.compute_edits(pd.DataFrame(), orig) == {})
    check("empty edited_df is safe", tb.compute_edits(orig, pd.DataFrame()) == {})

    print()
    if failures:
        print(f"{len(failures)} TEST(S) FAILED: {failures}")
        sys.exit(1)
    print("ALL TASK BOARD TESTS PASSED")


if __name__ == "__main__":
    run_all_tests()
