# Synapse — Final Rubric Self-Audit

Scored as a hostile evaluator actively looking for reasons to deduct
marks, against the original 100-point MirAI capstone rubric. Scores are
NOT inflated. Where something is genuinely strong, it's scored as such;
where something is unverified or missing, it's marked down explicitly.

---

## 1. Technical Implementation & Architecture — **21/25**

**Strong:**
- `st.session_state` is architected with explicit DATA/UI/AI/APPROVAL
  separation and a stale-recommendation guard, backed by 16 passing unit
  tests including the guard's critical edge case.
- `st.form` gates the only Gemini-triggering actions; no accidental
  calls on widget change, confirmed by 19 integration test assertions.
- The Pandas pipeline (`utils/data_pipeline.py`) is comprehensive,
  handles empty/malformed input safely, and is independently tested with
  real pandas (7+ scenarios including a genuine `Timestamp` vs `date`
  type-mismatch edge case caught during testing, not assumed away).
- Architecture was explicitly audited for circular imports, duplicated
  schemas, and business-logic-in-app.py — none found; `app.py` is a
  ~110-line pure composition layer.

**Costs marks:**
- **"Zero terminal errors" cannot be honestly claimed.** This project
  was built in an offline sandbox with no `streamlit`, `pydantic`, or
  `google-genai` installed, and no network access. Every test mocks
  those dependencies to verify logic. The app has never actually been
  executed with `streamlit run app.py`. A real run may surface issues
  no mock caught (e.g. a Streamlit widget API detail, a pandas dtype
  quirk not covered by the test scenarios written).
- `services/gemini_service.py`'s SDK usage (`types.Part.from_bytes`,
  `GenerateContentConfig` fields, `response.candidates[0].content.parts[i].function_call`
  structure) is written from best knowledge of the `google-genai` SDK
  but has never executed against the real library — a minor API
  mismatch is plausible and would need a small fix on first real run.

---

## 2. AI Integration & Prompt Engineering — **17/20**

**Strong:**
- Two distinct, genuinely necessary Gemini use cases (extraction,
  assignment recommendation), both via structured function-calling, not
  free-text parsing.
- System prompts are domain-specific personas with explicit rules, not
  generic assistant framing.
- Dynamic f-string context (today's date, known team roster, current
  workload) is real, not decorative.
- Audio multimodality reuses the exact same tool/schema as text
  extraction — genuinely integrated, not a bolted-on gimmick.
- Retry loop distinguishes 4 error categories (`DATA_ERROR`/
  `AI_API_ERROR`/`AI_OUTPUT_ERROR`/`VALIDATION_ERROR`), verified against
  10 scripted-client scenarios covering recovery and exhaustion for each.

**Costs marks:**
- Within `AI_API_ERROR`, transient (network/timeout) and permanent
  (mid-loop auth expiry) failures aren't actually distinguished — both
  are retried identically. Auth failure at startup IS caught before the
  loop (`get_client()`), but a token that expires mid-session wouldn't be.
- Never executed against the real Gemini API — prompt quality,
  real-world extraction accuracy, and actual latency are unverified.

---

## 3. UI/UX & Data Visualization — **15/20**

**Strong:**
- Columns, expanders, tabs, KPI cards with real deltas, `st.data_editor`
  used for genuine inline task correction (not decoratively), custom CSS
  for a non-default visual identity, and explicit initial/invalid/
  loading/success/error states throughout both input forms.
- `compute_edits()` diffing logic is real and tested against a realistic
  `data_editor` output shape, including the `Timestamp`/`date` mismatch.

**Costs marks:**
- **`st.map` was deliberately omitted** — this project has no genuine
  geographic dimension, and the team judged a forced map worse than an
  honest omission. Defensible, but the rubric explicitly lists it as a
  scoring criterion, so this costs real points regardless of the
  reasoning.
- **Zero visual verification.** No screenshot exists. Every layout,
  spacing, and color decision was made without ever seeing it render —
  "looks professional, not AI-generated" is an intent, not a confirmed
  outcome.
- Custom CSS is minimal (a handful of rules) — real visual polish is
  unverified and likely needs iteration after a first real run.

---

## 4. Deployment & Cloud Engineering — **8/15**

**Strong:**
- `requirements.txt` is minimal and has no local-only dependencies.
- Secrets are never hardcoded; `st.secrets`-first with environment
  fallback, defensive against a missing `secrets.toml`.
- `.streamlit/config.toml` sets a real theme and disables
  `showErrorDetails` for production safety.

**Costs marks — this is the single biggest gap:**
- **The app has not actually been deployed anywhere.** The rubric asks
  for "successfully deployed and live on Streamlit Community Cloud" —
  that literally cannot be satisfied from this offline sandbox. This is
  deployment-*ready*, not deployed. Doing the actual deploy (and fixing
  whatever small thing breaks on first real run) is required before
  submission, not optional polish.

---

## 5. Open-Source Branding (GitHub) — **7/10**

**Strong:**
- `README.md` is complete: overview, problem/solution, architecture,
  Mermaid diagram, tech stack, installation, environment setup, usage,
  deployment steps, testing (with an honest scope caveat), limitations,
  future improvements.

**Costs marks:**
- No actual GitHub repository exists yet — the README references a live
  demo link and setup instructions that aren't validated against a real
  repo. This needs to be pushed, and the live demo link filled in, before
  submission.
- Screenshots section is explicitly a placeholder (honestly labeled as
  such) rather than filled in.

---

## 6. System Design & Documentation — **9/10**

**Strong:**
- The Mermaid diagram was written by tracing the actual code paths
  (`gemini_service.py`'s real retry/validation flow, the real approval
  gate), not drafted speculatively and left unsynced.
- Architecture documentation in the README matches the real file
  structure and each file's real responsibility.

**Costs marks:**
- Minor: the diagram doesn't show the audio input path as a visually
  distinct branch (it's folded into "Input" generically) — a fully
  precise diagram would separate the two extraction entry points more
  visibly.

---

## TOTAL: 77/100

## What would cost us marks — ranked by severity

1. **Not deployed live** (Deployment, −7). Fix: actually deploy to
   Streamlit Community Cloud and confirm it works.
2. **Never run against the real Gemini API or real Streamlit** (spread
   across Technical/AI/UI, roughly −9 combined). Fix: `pip install -r
   requirements.txt`, run locally, fix whatever small API mismatches
   surface, then re-verify.
3. **No GitHub repo pushed yet, no live demo link, no screenshots**
   (GitHub, −3). Fix: push the repo, deploy, capture screenshots, fill
   in the README's placeholder sections.
4. **`st.map` omitted** (UI/UX, partial credit lost). Defensible
   reasoning is documented; evaluators may or may not accept it.
5. **Transient vs. permanent API error distinction is incomplete**
   (AI Integration, minor). A real fix is small (check exception type/
   message for auth-related signals before deciding to retry) but
   wasn't implemented to avoid guessing at Gemini SDK exception types
   without being able to verify them against the real library.

None of these are "unfixable" — they're the specific, concrete gap
between "built and logic-tested in an offline sandbox" and "verified in
the real environment it will be judged in." That verification pass is
the highest-leverage thing to do before the Aug 25 deadline.
