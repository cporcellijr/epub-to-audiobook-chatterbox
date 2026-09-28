# Multi-voice build brief: a cast of voices chosen with a local LLM

You are adding multi-voice narration to `cporcellijr/epub-to-audiobook-chatterbox`. Today every book is
read by one voice. With this feature, a local LLM works out who speaks each line of dialogue, the
owner reviews a cast list (character, gender, voice), and the book is generated with each line in its
speaker's voice and the narrator's voice for everything else. You have no GPU and no LLM, so you
build the feature, its unit tests and a validation script. The owner's local session runs the script
against the real local LLM and Chatterbox before the feature is used.

## 1. Owner's decisions

- **Local LLM only.** No cloud model; book text never leaves the machine. Talk to any
  OpenAI-compatible chat endpoint (Ollama, llama.cpp's server, LM Studio): base URL, model name and an
  optional API key come from settings. Don't install or configure the LLM server yourself.
- **Not at the same time as speech generation.** The LLM pass and book generation never overlap. The
  GPU has 12 GB; Chatterbox holds about 4.4 GB and Kokoro about 1.2 GB even when idle. So the LLM pass
  should free Chatterbox's memory first and reload it afterwards (section 4.5).
- **Build it or borrow it.** The reference is
  [prakharsr/audiobook-creator](https://github.com/prakharsr/audiobook-creator), which does character
  identification and multi-voice narration. Its licence is GPL (check `LICENSE` for the exact
  version first). This repo is MIT and **public**. You have two options; choose per component, and
  explain your choice in the report:
  - **Build fresh (default).** Take ideas only: prompt shapes, how to merge character aliases, what
    worked for them. Write your own code. The licence stays MIT.
  - **Borrow code, only if it saves real work** (a component that would take a lot of effort to write
    and fits this pipeline without heavy rework). Then relicense the repo:
    - Replace `LICENSE` with the matching GPL text (GPL-3.0, or AGPL-3.0 if theirs is AGPL).
    - Keep every existing MIT notice in `THIRD_PARTY_NOTICES.md` (p0n1/epub_to_audiobook,
      Chatterbox-TTS-Server, abogen), and add audiobook-creator with its licence and exactly what was
      taken.
    - Say so in the README's first section.
    - Mark borrowed files or functions with a short header comment naming the source.

  Don't borrow a little and leave the licence as it is.

## 2. How the app works today (read these first)

- `audiobook_generator/book_parsers/epub_book_parser.py`: chapters in spine order. HTML block
  elements become paragraphs, marked with `PARAGRAPH_MARK` (`@BRK#`) in the chapter text.
- `audiobook_generator/tts_providers/openai_tts_provider.py`: paced narration.
  - `paced_units()` splits a chapter into sentence-sized units. A sentence under 40 characters joins
    the next one, so a unit can span a speaker change.
  - `_split_oversized_unit` splits anything over 450 characters, marked `continues_previous`, so it
    gets no pause.
  - `_paced_text_to_speech()` sends each unit to `/v1/audio/speech` with
    `voice=self.config.voice_name` and joins the audio with sentence and paragraph pauses.
  - Paragraph mode (`paced_unit_mode="paragraph"`) and Kokoro exist too; Kokoro always uses sentence
    units. Two engines exist: `chatterbox` (voices are file names such as `Elena.wav`) and `kokoro`
    (ids such as `af_heart`).
- `audiobook_generator/ui/chatterbox_ui.py`: the web UI (tabs "Make audiobook" and "Voice lab"),
  `build_config(...)`, `queue_settings(...)` and the estimates.
  - Queued jobs persist their `build_config` keyword arguments in `queue.json`. New arguments must
    have defaults so older jobs still build.
  - Jobs run one at a time in spawned processes (`audiobook_generator/ui/job_queue.py`).
- Chatterbox (`chatterbox/server.py`) serves one generation at a time and caches each voice's
  conditioning, so switching voices per sentence costs nothing after a voice's first use. It has
  `POST /api/unload` and a reload path. Read the code to confirm which endpoint reloads the model and
  what `/v1/audio/speech` returns while it is unloaded.
- Tests: `docs/chatterbox-edition/WORKLOG.md` section 8. The app suite has 294 tests; all must still pass.

## 3. What to build, in two phases

**Phase 1: dialogue voice, no LLM.** A voice mode with a narrator voice plus one "dialogue" voice for
everything inside quotation marks. It needs only the dialogue splitting below and per-unit voices, and
it is the fallback whenever no LLM is configured.

**Phase 2: cast from a local LLM.** Speaker attribution, a cast list per book, owner review, then
generation with per-character voices.

## 4. Design

1. **Dialogue splitting (no LLM).**
   - Split each paragraph into narration and quoted spans, handling straight and curly quotes and
     single-quote dialogue styles.
   - Handle a quote that runs across paragraphs: the usual style opens each paragraph with a quote
     mark and closes only the last.
   - Handle apostrophes that aren't quotes, and em-dash dialogue if cheap.
   - Output ordered segments, each with a kind (narration or dialogue), an id stable within the
     chapter, and its text.
2. **Units follow speakers.** Paced units must never span two voices. Build units within each
   segment, not across segments: the short-sentence joining and the 450-character split still apply,
   inside a segment. The pause between a narration segment and a dialogue segment in the same
   paragraph is the sentence pause; paragraph pauses are unchanged. Sentence mode must still produce
   exactly today's units when the voice mode is single voice (test this).
3. **Attribution (LLM).**
   - Send each chapter in windows of numbered dialogue lines with enough surrounding narration to
     follow the conversation, plus the running character list.
   - Ask for JSON: each line id mapped to a speaker name, plus any new characters with gender
     (female/male/unknown) and a rough age (child/adult/elderly/unknown).
   - Validate strictly: unknown ids, missing ids, invalid JSON. Retry a window once, then mark its
     lines as unknown speaker (they get the dialogue voice).
   - Merge aliases across windows and chapters ("Tom", "Mr. Baker", "Thomas Baker").
   - Use the endpoint's JSON mode where it exists (`response_format`), and parse robustly where it
     doesn't. Keep prompts in one place, easy to tune.
   - Record progress in the job log. Make the chapter window size a constant.
4. **Cast.**
   - Store the analysis per book as JSON in the app's data folder, keyed so re-analysis is possible:
     the character list, line counts, gender and age, the chosen voice per character, the narrator
     voice, and the per-line attributions.
   - Suggest voices automatically: distinct voices for the characters with the most lines; gender
     taken from voice metadata; narrator and main characters never share a voice.
   - Voice gender metadata:
     - Kokoro voices carry it in the id prefix (`af_`, `am_`, `bf_`, `bm_`).
     - Chatterbox voices don't. Add a small owner-editable mapping (voice to female/male/neutral),
       with a place in the Voice lab to set it. Don't guess genders from names in code.
   - A cast's voices must belong to the job's engine.
5. **Running the LLM pass.**
   - Make cast analysis a queue job type, so it runs one at a time with books and can never overlap
     generation. Show it in the queue with progress.
   - Before it calls the LLM, it frees Chatterbox's GPU memory (`/api/unload`) when a setting allows
     (default on). Afterwards it reloads Chatterbox and waits until `/api/model-info` reports
     `loaded`, even if the analysis failed.
   - The next book job must never start while Chatterbox is unloaded.
   - Kokoro has no unload API; leave it.
6. **UI.**
   - In the Make audiobook tab, add a voice mode choice: single voice (default, exactly today's
     behaviour), narrator + dialogue voice, or cast (LLM). Cast mode needs the LLM configured.
   - Cast mode adds an "Analyse cast" step for the ticked chapters, which queues the analysis job.
     When it finishes, show an editable cast table: character, lines, gender, voice (dropdown for the
     job's engine), and a sample button reusing the existing sample path.
   - "Add to queue" then carries the cast with the job.
   - Keep the existing layout conventions in `chatterbox_ui.py`.
7. **Settings.**
   - `LLM_BASE_URL`, `LLM_MODEL` and `LLM_API_KEY` (optional); empty `LLM_BASE_URL` hides cast mode.
   - The unload-Chatterbox toggle.
   - Pass them through `docker-compose.chatterbox.yml` as `${VAR:-}` and document them in
     `.env.example` and the README, following how `KOKORO_BASE_URL` was added.
   - Read settings per call, never at import.
8. **Generation.**
   - The provider takes a per-unit voice: narrator, dialogue voice, or a character's voice from the
     cast.
   - An unknown speaker gets the dialogue voice.
   - Tag chapter files and the M4B exactly as today.
   - Resume (`skip_existing`) and retries must keep working.

## 5. Traps

- A chapter can be long. Budget the context window, and don't send the whole chapter in one request
  to a 7-9B local model.
- Small local models drift from JSON and invent line ids; validation and retry are not optional.
- The owner's library is private. **Never put real book titles, authors or passages in code, tests,
  fixtures, docs or commit messages.** Use invented text or public-domain text (e.g. Project
  Gutenberg) for fixtures and the validation set.
- Old queued jobs have no voice-mode or cast keys and must build as single voice.
- Don't change existing behaviour silently: single voice mode must send exactly the same requests
  as today (a test should prove it on a fixture).

## 6. Tests you can run without GPU or LLM

- Quote splitting on tricky invented passages.
- Units never spanning a voice change.
- Single-voice unit sequences unchanged.
- Attribution parsing with mocked LLM responses: good JSON, broken JSON, missing and unknown ids,
  aliases.
- Cast suggestions: gender, distinct voices, narrator never reused.
- Cast persistence.
- `build_config` / `queue_settings` compatibility with old jobs.
- The analysis job's unload/reload order, including when the analysis fails, and the queue never
  starting a book while Chatterbox is unloaded (mock the HTTP calls).

Each new behaviour gets at least one test that fails with the change reverted.

## 7. Validation script for the owner's machine

Add `docs/chatterbox-edition/experiments/multivoice/validate_multivoice.py` plus a hand-labelled
fixture: public-domain dialogue-heavy passages, a few hundred lines in total, with the true speaker
of every line. Run against the configured local LLM, it prints:

- attribution accuracy per passage and overall;
- the confusion between the top characters;
- the rate of invalid JSON before and after retries;
- time per 1,000 lines;
- that Chatterbox was unloaded during the pass and reloaded after.

It then generates a two-minute multi-voice sample through the real pipeline for listening. Include
the exact `docker run` command, following the other scripts under `experiments/`.

## 8. Out of scope

Emotion or style per line, installing or tuning the LLM server, per-character speed, anything GPU-side.

## 9. Delivery

- Work on branch `feature/multivoice` from `main` and push it. Don't merge or open a PR; the owner's
  local session reviews, validates and merges. Phases may be separate commits.
- Commits: `git -c user.name="Black Cat Media Dev" -c user.email="cporcellijr@gmail.com" commit`,
  with subjects like `UI: ...` or `Provider: ...`. No Co-Authored-By or other trailers.
- Match the surrounding style: type hints and a short docstring on new functions, comments only where
  the reason isn't obvious.
- Docs: README section for the feature and its settings, a WORKLOG section "Multi-voice narration",
  and licence changes if you borrowed code.
- Report back:
  - build-or-borrow per component, and why;
  - files changed and why;
  - design decisions;
  - tests run and results;
  - what you could not verify;
  - the command to run the validation script.
