"""
Meeting input panel — the only place in Synapse that triggers a
Gemini call, and only in response to an explicit form submission
(never on a widget change, tab switch, or plain rerun).

Split into two layers per the architecture:
  _handle_submission()  — pure orchestration (session + pipeline +
                           gemini_service wiring). No st.* widget
                           calls, so it's testable without a real
                           Streamlit runtime.
  render_input_panel()  — the st.form itself and all UI state
                           (initial / invalid input / loading /
                           success / error). Contains no business
                           logic — it delegates to the function above.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import streamlit as st

from models.schemas import ExtractionError, MeetingExtraction
from services import gemini_service
from utils import data_pipeline, errors, session

CLEAR_DRAFT_FLAG = "synapse_clear_draft_flag"
TRANSCRIPT_WIDGET_KEY = "synapse_transcript_widget"
TITLE_WIDGET_KEY = "synapse_title_widget"

AUDIO_NONCE_KEY = "synapse_audio_uploader_nonce"
AUDIO_TITLE_KEY = "synapse_audio_title_widget"


def _commit_result(
    result: Optional[MeetingExtraction],
    error: Optional[ExtractionError],
    trace: list[dict],
    prev_kpis: dict,
) -> None:
    """Shared by both the text and audio submission paths: record the
    outcome in AI state, and only append to the meeting history (DATA
    state) on genuine success."""
    session.set_last_result(extraction=result, error=error, trace=trace)
    if result is not None:
        session.add_meeting(result, prev_kpi_snapshot=prev_kpis)


def _handle_submission(
    transcript: str, meeting_title: str
) -> tuple[Optional[MeetingExtraction], Optional[ExtractionError], list[dict]]:
    """
    Orchestrates one text extraction request: generates a meeting id,
    builds the team roster and pre-call KPI snapshot from existing
    session history (deterministic, Pandas — no AI), calls Gemini,
    and commits the result (or error) back to session state.

    Note: the meeting counter advances even if extraction ultimately
    fails. This is a deliberate simplification — meeting ids are
    opaque identifiers, not a gapless sequence, so a skipped number
    on failure is harmless and avoids rollback complexity.
    """
    meeting_id = session.next_meeting_id()
    title = meeting_title.strip() if meeting_title and meeting_title.strip() else f"Meeting on {date.today().isoformat()}"

    existing_meetings = session.get_meetings()
    team_members = data_pipeline.known_team_members(existing_meetings)
    prev_df = data_pipeline.tasks_to_dataframe(existing_meetings)
    prev_kpis = data_pipeline.compute_kpis(prev_df, as_of=date.today())

    result, error, trace = gemini_service.extract_tasks(
        transcript=transcript,
        meeting_id=meeting_id,
        meeting_title=title,
        meeting_date=date.today(),
        team_members=team_members,
    )

    _commit_result(result, error, trace, prev_kpis)
    return result, error, trace


def _handle_audio_submission(
    audio_bytes: bytes, mime_type: str, meeting_title: str
) -> tuple[Optional[MeetingExtraction], Optional[ExtractionError], list[dict]]:
    """
    Same orchestration as _handle_submission, but for a voice memo:
    Gemini's native audio understanding extracts tasks directly from
    the recording — no separate transcription step — through the
    identical record_meeting_tasks tool and schema, so a voice-created
    task is indistinguishable from a transcript-created one downstream.
    """
    meeting_id = session.next_meeting_id()
    title = meeting_title.strip() if meeting_title and meeting_title.strip() else f"Voice memo on {date.today().isoformat()}"

    existing_meetings = session.get_meetings()
    team_members = data_pipeline.known_team_members(existing_meetings)
    prev_df = data_pipeline.tasks_to_dataframe(existing_meetings)
    prev_kpis = data_pipeline.compute_kpis(prev_df, as_of=date.today())

    result, error, trace = gemini_service.extract_tasks_from_audio(
        audio_bytes=audio_bytes,
        mime_type=mime_type,
        meeting_id=meeting_id,
        meeting_title=title,
        meeting_date=date.today(),
        team_members=team_members,
    )

    _commit_result(result, error, trace, prev_kpis)
    return result, error, trace


def _render_transcript_tab() -> None:
    # Apply a pending draft-clear from a PRIOR successful submission,
    # before the widgets below are instantiated this run. Doing this
    # here (not immediately after success) avoids Streamlit's "cannot
    # modify a widget's value after it's been instantiated" error.
    if st.session_state.get(CLEAR_DRAFT_FLAG):
        st.session_state[TRANSCRIPT_WIDGET_KEY] = ""
        st.session_state[TITLE_WIDGET_KEY] = ""
        st.session_state[CLEAR_DRAFT_FLAG] = False

    with st.form("meeting_input_form", border=True):
        title = st.text_input(
            "Meeting title (optional)",
            key=TITLE_WIDGET_KEY,
            placeholder="e.g. Weekly Product Sync",
        )
        transcript = st.text_area(
            "Paste meeting transcript",
            key=TRANSCRIPT_WIDGET_KEY,
            height=220,
            placeholder="Paste the raw meeting transcript here...",
        )
        submitted = st.form_submit_button(
            "Extract Action Items", type="primary", use_container_width=True
        )

    if not submitted:
        return  # INITIAL STATE — idle form, nothing else to render

    stripped = transcript.strip() if transcript else ""
    if len(stripped) < gemini_service.MIN_TRANSCRIPT_LENGTH:
        # INVALID INPUT STATE — text is preserved (no clear-on-submit),
        # no API call was made, user can fix and resubmit immediately.
        st.warning(
            f"Transcript is too short ({len(stripped)} characters). "
            f"Paste at least {gemini_service.MIN_TRANSCRIPT_LENGTH} characters "
            "of real meeting content before extracting."
        )
        return

    # LOADING STATE
    with st.spinner("Synapse is reading the transcript and extracting action items..."):
        result, error, _trace = _handle_submission(transcript, title)

    if error is not None:
        # ERROR STATE — specific, human-readable, no traceback ever shown
        st.error(f"Extraction failed: {error.message}")
        hint = errors.friendly_hint(error)
        if hint:
            st.caption(hint)
        return

    # SUCCESS STATE
    st.success(f'Extracted {len(result.tasks)} task(s) from "{result.meeting_title}".')
    st.session_state[CLEAR_DRAFT_FLAG] = True
    st.rerun()


def _render_audio_tab() -> None:
    st.caption(
        "Record a short voice memo describing action items (or upload one) — "
        "Synapse listens and extracts tasks the same way it does from a transcript."
    )

    nonce = st.session_state.get(AUDIO_NONCE_KEY, 0)
    recorded = None
    if hasattr(st, "audio_input"):
        # st.audio_input (mic recorder) — available in Streamlit 1.38+.
        # Guarded with hasattr so the app degrades gracefully to
        # file-upload-only on older Streamlit versions rather than
        # crashing at import time.
        recorded = st.audio_input("Record a voice memo", key=f"synapse_audio_rec_{nonce}")

    uploaded = st.file_uploader(
        "...or upload an audio file",
        type=["wav", "mp3", "m4a", "ogg", "webm"],
        key=f"synapse_audio_upload_{nonce}",
    )

    audio_title = st.text_input(
        "Meeting title (optional)", key=AUDIO_TITLE_KEY, placeholder="e.g. Quick voice note"
    )

    # A plain button, not st.form: recording/uploading audio already
    # triggers its own rerun to preview the clip, so a form wouldn't
    # add extra call-gating here — the button click is still the one
    # and only explicit action that triggers a Gemini call.
    submit = st.button("Extract Action Items from Audio", type="primary", use_container_width=True)

    if not submit:
        return  # INITIAL STATE

    audio_file = recorded or uploaded
    if audio_file is None:
        # INVALID INPUT STATE
        st.warning("Record or upload an audio file before extracting.")
        return

    audio_bytes = audio_file.getvalue()
    mime_type = getattr(audio_file, "type", None) or "audio/wav"

    # LOADING STATE
    with st.spinner("Synapse is listening and extracting action items..."):
        result, error, _trace = _handle_audio_submission(audio_bytes, mime_type, audio_title)

    if error is not None:
        # ERROR STATE
        st.error(f"Extraction failed: {error.message}")
        hint = errors.friendly_hint(error)
        if hint:
            st.caption(hint)
        return

    # SUCCESS STATE
    st.success(f'Extracted {len(result.tasks)} task(s) from "{result.meeting_title}".')
    st.session_state[AUDIO_NONCE_KEY] = nonce + 1  # forces fresh widget instances next render
    st.session_state[AUDIO_TITLE_KEY] = ""
    st.rerun()


def render_input_panel() -> None:
    """Renders the meeting input panel (text + audio tabs) and handles
    every resulting UI state. Call once per rerun from app.py."""
    st.subheader("New Meeting")
    tab_text, tab_audio = st.tabs(["📝 Paste Transcript", "🎙️ Voice Input"])
    with tab_text:
        _render_transcript_tab()
    with tab_audio:
        _render_audio_tab()
