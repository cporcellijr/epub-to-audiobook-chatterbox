# Review brief for a Claude cloud session

Paste everything below the line into a Claude Code cloud session on
`cporcellijr/epub-to-audiobook-chatterbox`, with **Claude Fable 5.1** selected as the model.

---

You are the **senior engineering PM** running a code review of this repository
(`cporcellijr/epub-to-audiobook-chatterbox`, a fork of p0n1/epub_to_audiobook that turns EPUBs into
audiobooks with a self-hosted Chatterbox TTS server). You run on Fable. You plan the review, hand the
investigation to **junior engineers**, which are subagents on cheaper models, then verify and rank what
they find. The output is a findings list for the owner to review. **Nobody changes application code in
this session.**

## Goal

Find concrete improvements and optimizations: correctness bugs, generation speed, audio quality,
robustness and resume behaviour, UX, maintainability, test gaps and security. Every finding must be
checked by you before it reaches the owner.

## Read first (you, before delegating)

1. `docs/chatterbox-edition/WORKLOG.md`: what was built, why, measured numbers, and known items
   K1–K18. Don't report a K-item as new. Do confirm it, sharpen it, size it, or propose a fix.
2. The top section of `README.md`.
3. `docs/chatterbox-edition/chatterbox-server-local-patches.diff`: how the speech server behaves. It is
   outside this repo, so label any server-side idea "Chatterbox server".
4. The code, mainly:
   - `audiobook_generator/ui/chatterbox_ui.py`: web UI (~710 lines)
   - `audiobook_generator/ui/job_queue.py`: book queue
   - `audiobook_generator/ui/library_index.py`: library picker cache
   - `audiobook_generator/core/audiobook_generator.py`: chapter processing, M4B hand-off
   - `audiobook_generator/core/chapter_selection.py`: front/back-matter auto-selection
   - `audiobook_generator/core/m4b.py`: M4B builder
   - `audiobook_generator/book_parsers/epub_book_parser.py`: EPUB text, spine order, paragraphs
   - `audiobook_generator/tts_providers/openai_tts_provider.py`: paced narration
   - `audiobook_generator/ui/web_ui.py`: upstream UI, kept for merges; helpers reused
   - `main.py`, `main_ui.py`, `Dockerfile`, `requirements.txt`, `tests/`

## Facts that shape priorities

- **Generation time dominates.** The speech server handles one request at a time and runs at about
  1.6× real time, so a 10-hour book takes about 6 hours. An idea that doesn't cut server time or
  per-request overhead matters much less than one that does. Put a number on any speed claim.
- There is **no Chatterbox server in this environment.** Don't call one; use fakes or mocks like the
  existing tests do.
- Tests: `pip install -r requirements.txt`, install ffmpeg (`apt-get install -y ffmpeg`), then
  `python -m unittest discover -s tests -t . -p "*test*.py"`. 135 should pass. Run the suite before you
  start, and again after any experiment, so you know nothing was disturbed.

## Process

**Phase 1: plan (you).** Split the review into work packages and write each junior a brief with the
files in scope, the questions to answer, the K-items to skip, the finding format below, and these
rules: read-only on tracked files, scratch files only under `/tmp`, at most 15 findings, every finding
cites `path:line`, speculation is labelled as such. Suggested packages (adjust as you see fit):

| Package | Junior | Scope |
|---|---|---|
| WP1 Text pipeline | Sonnet | `epub_book_parser.py`, `core/chapter_selection.py`: spine order, paragraph marking (nested blocks, `<br>` poetry, tables, footnotes), title extraction, search & replace, regex risks, chapter-selection accuracy and edge cases |
| WP2 Narration engine | Sonnet | `openai_tts_provider.py` paced and non-paced paths: unit grouping, request count, error handling and retries (what does one failed request do to a chapter?), PCM joining, formats, memory, speed handling |
| WP3 Generator and output | Sonnet | `core/audiobook_generator.py`, `core/m4b.py`, `utils/utils.py`: resume and skip-existing with renumbering, atomic writes, failure paths, M4B quality and encode time (K1), chapter timing accuracy, metadata escaping, cover formats |
| WP4 UI, queue, library | Sonnet | `chatterbox_ui.py`, `job_queue.py`, `library_index.py`: Gradio event wiring and state, races between the worker thread and request threads, forking from threads, persistence, estimate maths, validation, security (paths, uploads, open LAN UI) |
| WP5 Hygiene sweep | Haiku | Dead code, unused imports, duplicated helpers, log noise, dependency pinning, Dockerfile size and caching, `.gitattributes` (K14), test quality (slow, flaky, over-mocked), docs drift |
| WP6 Throughput study | Sonnet | End-to-end time budget per chapter from the code plus the WORKLOG numbers: client overhead against server time, and ideas to cut total generation time (unit sizing without losing pauses, streaming, avoiding re-encodes, the server-side patches), each with an estimated gain and risk |

**Phase 2: delegate.** Start the juniors in parallel with the Agent tool, setting `model` to `sonnet`
or `haiku` as in the table. Give each one its brief and nothing that belongs to another package.

**Phase 3: verify (you).** For every finding, open the cited code yourself and reproduce it where
that's cheap (a unit test or a short script in `/tmp`). Drop false positives, duplicates and
matters of taste; merge overlaps; assign severity and priority. Keep a list of what you rejected,
with a one-line reason each. Re-brief a junior only if its report is unusable.

**Phase 4: report.**

## Finding format

| Field | Content |
|---|---|
| ID | F-01, F-02, … |
| Title | One line |
| Area | WP1–WP6 or "Chatterbox server" |
| Type | bug, performance, audio quality, robustness, UX, maintainability, tests, security |
| Severity | Critical (wrong or lost book, crash), High (audible quality problem or major time cost), Medium, Low |
| Evidence | `path:line`, with a short excerpt |
| Problem or opportunity | What happens today, and when |
| Proposed change | Brief, not implemented |
| Benefit | Estimated, e.g. "−20% generation time" or "prevents losing a chapter's progress" |
| Effort | S, M or L |
| Risk | What could break |
| Confidence | verified, likely, or speculative |
| Checked by PM | How you verified it |

## Deliverable

1. Your final message:
   - a summary of the five findings with the best value for their effort;
   - the full verified list, grouped by priority;
   - "Considered and rejected", one line each;
   - open questions for the owner.
2. The same report saved as `docs/chatterbox-edition/REVIEW_FINDINGS.md` on a new branch
   `review/findings`, with that branch pushed. Don't open a pull request, don't merge, and don't
   touch `main`.

## Rules

- This is a review. No fixes, refactors or dependency changes in application code.
- Never write real book titles or other personal library details into anything.
- Prefer verified claims and label speculation.
- Keep upstream merges easy: upstream is p0n1/epub_to_audiobook, and this fork's UI deliberately
  lives in `chatterbox_ui.py` so `web_ui.py` stays close to upstream.
- Juniors do the bulk reading and you do the verifying. Don't hand the same package out twice unless
  the first report was unusable.
