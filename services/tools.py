"""
Tool definitions for Gemini function-calling.

This module owns exactly two things:
1. The JSON-schema declaration of the single tool Gemini is allowed to
   call (record_meeting_tasks) — the "contract" the model must fill in.
2. A thin validator that turns the tool call's raw arguments into a
   MeetingExtraction, using the schemas already defined in
   models/schemas.py.

It does NOT talk to the Gemini API directly (that's gemini_service.py's
job) and does NOT redefine Task/MeetingExtraction — it only imports and
reuses them, so there is exactly one source of truth for what a task is.
"""

from __future__ import annotations

from datetime import date

from google.genai import types

from models.schemas import AssignmentRecommendation, MeetingExtraction, Task

TOOL_NAME = "record_meeting_tasks"
RECOMMEND_TOOL_NAME = "recommend_task_assignments"

RECORD_TASKS_TOOL = types.FunctionDeclaration(
    name=TOOL_NAME,
    description=(
        "Records the complete set of action items extracted from a meeting "
        "transcript, along with a brief neutral summary of the meeting."
    ),
    parameters={
        "type": "OBJECT",
        "properties": {
            "summary": {
                "type": "STRING",
                "description": "One or two sentence neutral summary of the meeting.",
            },
            "tasks": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "id": {"type": "STRING"},
                        "description": {"type": "STRING"},
                        "owner": {"type": "STRING"},
                        "deadline": {
                            "type": "STRING",
                            "nullable": True,
                            "description": "ISO date YYYY-MM-DD, or null.",
                        },
                        "priority": {
                            "type": "STRING",
                            "enum": ["High", "Medium", "Low"],
                        },
                        "confidence": {"type": "NUMBER"},
                        "source_excerpt": {"type": "STRING"},
                    },
                    "required": ["id", "description", "owner", "priority"],
                },
            },
        },
        "required": ["summary", "tasks"],
    },
)


def get_tool() -> types.Tool:
    """Returns the extraction Tool object ready to pass into GenerateContentConfig."""
    return types.Tool(function_declarations=[RECORD_TASKS_TOOL])


RECOMMEND_ASSIGNMENTS_TOOL = types.FunctionDeclaration(
    name=RECOMMEND_TOOL_NAME,
    description=(
        "Records recommended owners for a set of unassigned tasks, based on "
        "current team workload and task context. Each recommendation must "
        "name a real, known team member — never invent a name."
    ),
    parameters={
        "type": "OBJECT",
        "properties": {
            "recommendations": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "task_id": {"type": "STRING"},
                        "recommended_owner": {"type": "STRING"},
                        "rationale": {
                            "type": "STRING",
                            "description": "Brief, specific reason: workload, task-description skill cues, deadline pressure.",
                        },
                        "confidence": {"type": "NUMBER"},
                        "alternative_candidate": {"type": "STRING", "nullable": True},
                    },
                    "required": ["task_id", "recommended_owner", "rationale", "confidence"],
                },
            },
        },
        "required": ["recommendations"],
    },
)


def get_recommend_tool() -> types.Tool:
    """Returns the assignment-recommendation Tool object."""
    return types.Tool(function_declarations=[RECOMMEND_ASSIGNMENTS_TOOL])


def validate_payload(
    payload: dict,
    meeting_id: str,
    meeting_title: str,
    meeting_date: date,
    transcript_len: int,
) -> MeetingExtraction:
    """
    Converts raw record_meeting_tasks arguments into a validated
    MeetingExtraction. Raises pydantic.ValidationError on bad data —
    the caller (gemini_service) decides whether to retry.

    This is the ONLY place tool-call output shape meets the schema
    layer, so gemini_service.py never has to know the tool's raw
    argument shape.
    """
    tasks_raw = payload.get("tasks", [])
    tasks = [Task(**t) for t in tasks_raw]
    return MeetingExtraction(
        meeting_id=meeting_id,
        meeting_title=meeting_title,
        meeting_date=meeting_date,
        tasks=tasks,
        summary=payload.get("summary", ""),
        raw_transcript_length=transcript_len,
    )


def validate_recommendations(payload: dict) -> list[AssignmentRecommendation]:
    """
    Converts raw recommend_task_assignments arguments into validated
    AssignmentRecommendation objects. Raises pydantic.ValidationError
    on bad data — the caller (gemini_service) decides whether to retry.
    """
    recs_raw = payload.get("recommendations", [])
    return [AssignmentRecommendation(**r) for r in recs_raw]