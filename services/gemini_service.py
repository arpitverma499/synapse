"""
Gemini orchestration service.

Design intent: Gemini is used as a specialized reasoning engine with
two distinct jobs — extracting action items (from text or audio) and
recommending owners for unassigned tasks given current workload — not
as a conversational chatbot. All deterministic math (workload totals,
overdue calculation, utilization) happens in utils/data_pipeline.py,
NOT here. This service's job is turning unstructured input into
validated structured output, with retry-on-invalid-schema, and never
performs arithmetic Gemini could get subtly wrong.

Every public function returns (result, error, trace):
  result — the validated domain object (or None on failure)
  error  — an ExtractionError with a taxonomy `stage` (or None on success)
  trace  — the raw sequence of tool-call attempts, for the UI's Agent
           Trace panel

ExtractionError.stage taxonomy (see models/schemas.py for the full
contract this service must honor):
  DATA_ERROR       — input itself was invalid. Never retried.
  AI_API_ERROR      — the call to Gemini failed (auth/network/timeout).
                      Auth failures raise before the retry loop starts
                      and are never retried; network/timeout errors
                      ARE retried up to MAX_RETRIES.
  AI_OUTPUT_ERROR   — Gemini responded but didn't call the expected
                      tool. Retried once with a corrective prompt.
  VALIDATION_ERROR  — Gemini called the tool, but the arguments failed
                      schema or business-rule validation. Retried once
                      with the error fed back for self-correction.
"""

from __future__ import annotations

import json
import os
from datetime import date
from typing import Any, Callable, Optional

import streamlit as st
from google import genai
from google.genai import types
from pydantic import ValidationError

from models.schemas import AssignmentRecommendation, ExtractionError, MeetingExtraction
from services import tools

MODEL_NAME = "gemini-2.0-flash"
MAX_RETRIES = 2
MIN_TRANSCRIPT_LENGTH = 20  # below this, extraction is rejected before any API call

EXTRACTION_SYSTEM_PROMPT = """You are Synapse's Meeting Operations Analyst.

Your sole job is to read a raw meeting transcript (or a transcription
of spoken audio) and extract concrete, actionable tasks — commitments
someone made to do something, with a clear (even if implicit) owner
and, where stated, a deadline.

Rules:
- Only extract genuine action items. Do NOT extract general discussion
  points, decisions with no follow-up action, or vague intentions.
- If an owner is not clearly stated or strongly implied, set owner to
  "Unassigned" rather than guessing.
- If a deadline is not explicitly stated or clearly implied (e.g. "by
  Friday", "before the next sprint"), leave it null. Do not invent dates.
- Assign priority based on language cues (urgency words, business
  impact) — default to Medium when unclear.
- Include a short source_excerpt (the exact phrase that justifies the
  task) for every task so a human can verify it quickly.
- You MUST report your results by calling the record_meeting_tasks tool
  exactly once with the complete list. Do not respond in plain text.
"""

RECOMMENDATION_SYSTEM_PROMPT = """You are Synapse's Task Assignment Analyst.

You recommend who should own each currently-unassigned task, using the
team's current workload and each task's description as evidence. You
are a decision-support tool, not a decision-maker: your recommendations
are always subject to human approval before anything changes.

Rules:
- Only recommend team members from the provided known-team-members list.
  Never invent a name. If no one is a reasonable fit, recommend
  "Unassigned" with a rationale explaining why.
- Prefer people with lower current workload when task-relevant skill
  cues are absent or tied.
- Weigh deadline pressure: a task due soon should favor whoever has
  capacity to act on it soon, not just whoever has the fewest tasks
  overall.
- Give a specific, brief rationale (workload numbers, skill cues in
  the task description, deadline pressure) — never a generic
  "seems like a good fit."
- Where reasonable, name one alternative_candidate as a second option
  for the human reviewer.
- You MUST report your results by calling the recommend_task_assignments
  tool exactly once with the complete list. Do not respond in plain text.
"""


class GeminiServiceError(Exception):
    """Raised for unrecoverable service-level failures (bad key, network)."""


@st.cache_resource(show_spinner=False)
def get_client() -> genai.Client:
    """Initialize the Gemini client once per session. Raises a clear
    error if no API key is configured, instead of failing deep in a
    request later."""
    api_key = None
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", None)
    except Exception:
        # No secrets.toml present at all (common in local dev) — fall
        # through to environment variable instead of crashing.
        api_key = None
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise GeminiServiceError(
            "No Gemini API key found. Add GEMINI_API_KEY to Streamlit "
            "secrets or your local .env file."
        )
    return genai.Client(api_key=api_key)


def _build_context(meeting_title: str, meeting_date: date, team_members: Optional[list[str]]) -> str:
    """Dynamic f-string context injected alongside the transcript so the
    model has today's date (for deadline interpretation) and a known
    team roster (to prefer real names over 'Unassigned' when possible)."""
    roster = ", ".join(team_members) if team_members else "no known roster provided"
    return (
        f"Meeting title: {meeting_title}\n"
        f"Meeting date: {meeting_date.isoformat()}\n"
        f"Known team members: {roster}\n"
        f"(Use the meeting date to resolve relative deadlines like 'by Friday'.)"
    )


def _extract_tool_call(response, tool_name: str) -> Optional[dict]:
    """Pull a named function call's arguments out of a Gemini response.
    Returns None if the model didn't call that tool."""
    try:
        for part in response.candidates[0].content.parts:
            if getattr(part, "function_call", None) and part.function_call.name == tool_name:
                return dict(part.function_call.args)
    except (AttributeError, IndexError):
        return None
    return None


def _generate_with_tool(
    contents: Any,
    system_prompt: str,
    tool: types.Tool,
    tool_name: str,
    validate_fn: Callable[[dict], Any],
) -> tuple[Optional[Any], Optional[ExtractionError], list[dict]]:
    """
    Shared retry loop used by every Gemini call in this service:
    call the model -> look for the expected tool call -> validate its
    arguments -> retry with a corrective prompt on failure, up to
    MAX_RETRIES. Centralized here so extraction, audio extraction, and
    recommendation all share one retry/error-taxonomy implementation
    instead of three near-duplicate copies.

    `contents` may be a str (text-only) or a list of parts (e.g. text +
    inline audio data) — passed straight through to generate_content.
    `validate_fn` receives the raw tool-call argument dict and must
    return the validated domain object, raising pydantic.ValidationError
    or ValueError (for business-rule failures like an unknown task id)
    on bad data.
    """
    trace: list[dict] = []

    try:
        client = get_client()
    except GeminiServiceError as e:
        return None, ExtractionError(stage="AI_API_ERROR", message=str(e)), trace

    correction_note = ""
    for attempt in range(MAX_RETRIES + 1):
        if isinstance(contents, list):
            request_contents = contents + [correction_note] if correction_note else contents
        else:
            request_contents = contents + correction_note

        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=request_contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    tools=[tool],
                    temperature=0.2,
                ),
            )
        except Exception as e:  # network/timeout/API-side failures
            trace.append({"attempt": attempt, "error": f"API call failed: {e}"})
            if attempt == MAX_RETRIES:
                return None, ExtractionError(stage="AI_API_ERROR", message=str(e)), trace
            continue

        payload = _extract_tool_call(response, tool_name)
        trace.append({"attempt": attempt, "tool_call_args": payload})

        if payload is None:
            correction_note = (
                f"\n\nREMINDER: You must call the {tool_name} tool. "
                "Do not respond in plain text."
            )
            if attempt == MAX_RETRIES:
                return None, ExtractionError(
                    stage="AI_OUTPUT_ERROR",
                    message=f"Model did not return a structured {tool_name} call after retries.",
                    raw_response=str(getattr(response, "text", "")),
                ), trace
            continue

        try:
            result = validate_fn(payload)
            return result, None, trace
        except (ValidationError, ValueError) as ve:
            correction_note = (
                f"\n\nYour previous tool call had invalid data: {ve}. "
                f"Please call {tool_name} again with corrected, valid arguments."
            )
            if attempt == MAX_RETRIES:
                return None, ExtractionError(
                    stage="VALIDATION_ERROR",
                    message=f"Response failed validation after retries: {ve}",
                    raw_response=json.dumps(payload),
                ), trace

    # Unreachable in practice (every branch above returns by the final
    # attempt), but keeps the function total for static analysis.
    return None, ExtractionError(stage="AI_OUTPUT_ERROR", message="Unknown extraction failure."), trace


def extract_tasks(
    transcript: str,
    meeting_id: str,
    meeting_title: str,
    meeting_date: date,
    team_members: Optional[list[str]] = None,
) -> tuple[Optional[MeetingExtraction], Optional[ExtractionError], list[dict]]:
    """Extracts tasks from a pasted text transcript."""
    if not transcript or len(transcript.strip()) < MIN_TRANSCRIPT_LENGTH:
        return None, ExtractionError(
            stage="DATA_ERROR",
            message="Transcript is too short to contain meaningful content.",
        ), []

    context = _build_context(meeting_title, meeting_date, team_members)
    contents = f"{context}\n\nTRANSCRIPT:\n{transcript}"

    def _validate(payload: dict) -> MeetingExtraction:
        return tools.validate_payload(payload, meeting_id, meeting_title, meeting_date, len(transcript))

    return _generate_with_tool(
        contents, EXTRACTION_SYSTEM_PROMPT, tools.get_tool(), tools.TOOL_NAME, _validate
    )


def extract_tasks_from_audio(
    audio_bytes: bytes,
    mime_type: str,
    meeting_id: str,
    meeting_title: str,
    meeting_date: date,
    team_members: Optional[list[str]] = None,
) -> tuple[Optional[MeetingExtraction], Optional[ExtractionError], list[dict]]:
    """
    Extracts tasks directly from spoken audio (a voice memo describing
    action items, or a short meeting recording) using Gemini's native
    audio understanding — no separate transcription step. Reuses the
    exact same tool/validation/schema as text extraction, so a task
    created by voice is indistinguishable in the system from one
    extracted from a pasted transcript.
    """
    if not audio_bytes:
        return None, ExtractionError(
            stage="DATA_ERROR",
            message="No audio was provided.",
        ), []

    context = _build_context(meeting_title, meeting_date, team_members)
    audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
    instruction_part = (
        f"{context}\n\nThe following audio is a spoken meeting note or voice "
        "memo. Listen to it and extract action items exactly as you would "
        "from a written transcript."
    )
    contents = [instruction_part, audio_part]

    def _validate(payload: dict) -> MeetingExtraction:
        return tools.validate_payload(payload, meeting_id, meeting_title, meeting_date, len(audio_bytes))

    return _generate_with_tool(
        contents, EXTRACTION_SYSTEM_PROMPT, tools.get_tool(), tools.TOOL_NAME, _validate
    )


def recommend_assignments(
    unassigned_tasks: list[dict],
    workload_context: list[dict],
    known_team_members: list[str],
    as_of: date,
) -> tuple[Optional[list[AssignmentRecommendation]], Optional[ExtractionError], list[dict]]:
    """
    Recommends an owner for each currently-unassigned task, using
    deterministic workload data computed by utils/data_pipeline.py
    (never recomputed here) as evidence. Returns recommendations only
    — nothing is applied to any Task until a human approves it via
    utils/session.py's approve_recommendation.

    `unassigned_tasks` — list of {task_id, description, priority,
        deadline, source_excerpt} dicts (from data_pipeline).
    `workload_context` — list of {owner, open_tasks, utilization_pct}
        dicts (from data_pipeline.utilization_by_owner).
    """
    if not unassigned_tasks:
        return None, ExtractionError(
            stage="DATA_ERROR",
            message="There are no unassigned tasks to generate recommendations for.",
        ), []

    if not known_team_members:
        return None, ExtractionError(
            stage="DATA_ERROR",
            message="No known team members yet — process at least one meeting with named owners first.",
        ), []

    valid_task_ids = {t["task_id"] for t in unassigned_tasks}

    context = (
        f"As of date: {as_of.isoformat()}\n"
        f"Known team members: {', '.join(known_team_members)}\n\n"
        f"Current workload (open tasks, utilization %):\n{json.dumps(workload_context, default=str)}\n\n"
        f"Unassigned tasks needing a recommendation:\n{json.dumps(unassigned_tasks, default=str)}"
    )

    def _validate(payload: dict) -> list[AssignmentRecommendation]:
        recs = tools.validate_recommendations(payload)
        unknown = [r.task_id for r in recs if r.task_id not in valid_task_ids]
        if unknown:
            raise ValueError(f"Recommended task_id(s) not in the unassigned task list: {unknown}")
        return recs

    return _generate_with_tool(
        context, RECOMMENDATION_SYSTEM_PROMPT, tools.get_recommend_tool(), tools.RECOMMEND_TOOL_NAME, _validate
    )
