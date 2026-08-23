"""
Core data contracts for Synapse.

These schemas are the single source of truth for what a "task" is.
Gemini's tool-calling arguments are validated against these models
before anything touches the UI or the Pandas pipeline. If Gemini
returns something that doesn't fit, it is rejected here — not
silently coerced.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Priority(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class TaskStatus(str, Enum):
    OPEN = "Open"
    IN_PROGRESS = "In Progress"
    DONE = "Done"


class Task(BaseModel):
    """A single action item extracted from a meeting transcript."""

    model_config = {"validate_assignment": True}
    # validate_assignment=True means task.owner = "..." (e.g. from a
    # data_editor edit or an approved AI recommendation) re-runs this
    # model's validation, not just __init__ — an invalid edit raises
    # ValidationError immediately instead of silently corrupting state.

    id: str = Field(..., description="Unique task identifier, e.g. 'task_001'.")
    description: str = Field(..., min_length=3, max_length=300)
    owner: str = Field(
        default="Unassigned",
        description="Person responsible. 'Unassigned' if unclear from context.",
    )
    deadline: Optional[date] = Field(
        default=None,
        description="Deadline if explicitly stated or clearly implied. Null otherwise.",
    )
    priority: Priority = Field(default=Priority.MEDIUM)
    status: TaskStatus = Field(default=TaskStatus.OPEN)
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Model's confidence this was actually an action item (0-1).",
    )
    source_excerpt: str = Field(
        default="",
        max_length=250,
        description="The transcript snippet this task was extracted from, for human review.",
    )

    @field_validator("description")
    @classmethod
    def description_not_placeholder(cls, v: str) -> str:
        banned = {"n/a", "none", "todo", "task"}
        if v.strip().lower() in banned:
            raise ValueError("Task description is too vague to be a real action item.")
        return v.strip()


class MeetingExtraction(BaseModel):
    """The full validated result of processing one meeting transcript."""

    meeting_id: str
    meeting_title: str = Field(default="Untitled Meeting")
    meeting_date: date
    tasks: list[Task] = Field(default_factory=list)
    summary: str = Field(
        default="",
        max_length=500,
        description="One or two sentence neutral summary of the meeting, not a task.",
    )
    raw_transcript_length: int = 0

    @property
    def task_count(self) -> int:
        return len(self.tasks)

    @property
    def unassigned_count(self) -> int:
        return sum(1 for t in self.tasks if t.owner == "Unassigned")


class ExtractionError(BaseModel):
    """Returned instead of MeetingExtraction when extraction fails validation
    even after retry — lets the UI show a clear partial/failed state instead
    of crashing or silently losing data.

    `stage` is a coarse error category, used by the UI to decide how to
    respond (retry advice, auth hint, etc.) without parsing free-text
    messages. Allowed values:
      DATA_ERROR       — input itself was invalid (e.g. empty/too-short
                          transcript). Never retried — there's nothing
                          to retry, the input needs to change.
      AI_API_ERROR      — the call to Gemini failed (auth, network,
                          timeout, rate limit). Auth failures are
                          permanent and are never retried (see
                          gemini_service.get_client, which raises
                          before the retry loop even starts). Network/
                          timeout failures ARE retried, up to MAX_RETRIES.
      AI_OUTPUT_ERROR   — Gemini responded but didn't call the expected
                          tool at all. Retried once with a corrective
                          reminder prompt.
      VALIDATION_ERROR  — Gemini called the tool, but the arguments
                          didn't pass schema validation. Retried once
                          with the validation error fed back to the
                          model for self-correction.
    """

    stage: str
    message: str
    raw_response: Optional[str] = None


class AssignmentRecommendation(BaseModel):
    """
    An AI-generated suggestion for who should own an unassigned task,
    with the reasoning a human reviewer needs to evaluate it. This is
    a recommendation only — nothing in the system applies it to a
    Task's `owner` field without an explicit human approval step
    (see utils/session.py's approve_recommendation and the stale-guard
    invariant it enforces).
    """

    task_id: str
    recommended_owner: str = Field(..., min_length=1)
    rationale: str = Field(..., min_length=3, max_length=300)
    confidence: float = Field(ge=0.0, le=1.0)
    alternative_candidate: Optional[str] = Field(
        default=None,
        description="A plausible second choice, if any — gives the human reviewer an option besides accept/reject.",
    )
