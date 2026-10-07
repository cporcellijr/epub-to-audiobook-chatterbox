# Work log: Setup, upstream changes, parser, review fixes, run and deploy

Part of the project work log. Sections keep the numbers they were written with; [WORKLOG.md](../WORKLOG.md) lists every section and which file holds it.

Sections here: §1, §4, §5, §6, §7, §8, §9, §10, §11, §32.

## Where things stand (2026-10-07)

- §1 describes the stack as first set up, and §5 lists what was changed from upstream and why.
- §8 is the run, test and deploy reference.
- §6's measurements and §7's limitations date from the Chatterbox weeks (2026-09-27/28). For current
  speeds see [breeze.md](breeze.md).
- Kokoro (§11) has been hidden since 2026-10-01 (§35).
- The EPUB parser splits a file that holds several contents-linked chapters (§32).

## 1. The setup

| Component | Role | Notes |
|---|---|---|
| [Chatterbox-TTS-Server](https://github.com/devnen/Chatterbox-TTS-Server) (MIT), container `chatterbox`, port 8004 | Speech engine | "Original" English model, BF16, RTX 4070 (12 GB), ~4.4 GiB VRAM. Built from `chatterbox/`; local patches in section 2.3. |
| This app, container `epub-to-audiobook`, port 7860 | EPUB → audiobook | Local build of this repo; talks to Chatterbox's OpenAI-compatible `/v1/audio/speech`. |
| BookOrbit (self-hosted library and reader) | Plays the finished books; also has live read-aloud TTS | Separate project; see section 4. |
| Kokoro-FastAPI, container `kokoro`, port 8880 | Optional second engine | v0.9.0, OpenAI-compatible. Measured 52.9x real time vs Chatterbox's ~1.8-1.6x, but no voice cloning and flatter delivery. Wired into this app since 2026-09-28 (opt-in via `KOKORO_BASE_URL`; section 11). |

How a book flows through this app:

```
EPUB (library mount or upload)
 → EpubBookParser: spine order, paragraphs from HTML blocks
 → chapter_selection: pre-tick story, untick front/back matter (user can override)
 → JobQueue: one book at a time
 → OpenAITTSProvider paced mode: sentence-sized units (or whole paragraphs, opt-in)
     → Chatterbox /v1/audio/speech (WAV) → join PCM with sentence/paragraph silences, one speed
       change per chapter → chapter AAC in <book>/.chapters/ (MP3 when not making an M4B)
 → m4b.build_m4b: concat by stream copy (no re-encode), chapter markers, cover, tags
     → <book>/<Title>.m4b
 → audiobook library → BookOrbit
```

## 4. BookOrbit live read-aloud (outside this repo)

- Its reader fetches 3 blocks ahead. With Chatterbox Original at ~1.6× real time that wasn't enough
  after a start or a chapter change. A local reader patch (10 s start-up buffer, fetches the next
  chapter's first clip during the last 4 blocks) made it smooth. BookOrbit is AGPL-licensed, so
  that patch is not included here.
- The server keeps every synthesized clip in memory with no expiry (key includes voice, speed and
  text), so replaced voice files don't affect already-played passages until it restarts.
- Clicking a voice in its picker only previews it; "Save as default/book voice" is required.
- An admin-curated voice list, once set, is enforced for synthesis, and the provider object is
  cached in memory: database edits need a restart.
- Its player mishandles a book made of many MP3 files (a 15 s skip jumped a whole chapter). That is
  why this app now produces a single M4B.

## 5. What was built in this repo, and why

| # | Change | Why | Where |
|---|---|---|---|
| 1 | Pointed the upstream OpenAI provider at Chatterbox | Works unmodified: voice names aren't validated and Chatterbox ignores the model name and `instructions` | config only |
| 2 | Merged upstream PRs #191, #189, #186 | Skip existing chapters and leak fixes; larger chapter ranges; cover/title/author. One conflict (both removed the per-task book parser) resolved to #186's design | several |
| 3 | Read chapters in spine order; skip the nav document and non-linear items | Upstream iterated the manifest: the table of contents was narrated and chapters could come out of order (test fixture: 24 → 23 chapters) | `book_parsers/epub_book_parser.py` |
| 4 | Write chapters to a hidden `.part` file and rename when complete | A crash could leave a truncated or untagged chapter that "skip existing" would then keep | `core/audiobook_generator.py` |
| 5 | Output folder from the EPUB title; default re-evaluated per page load | The default timestamp was fixed at server start, so consecutive books would share a folder | `ui/web_ui.py` helpers |
| 6 | Voice list from the mounted Chatterbox voices folder | Asking the server timed out while it was generating (it is serial) | `ui/web_ui.py` helpers |
| 7 | Personal UI: one provider, Advanced section, Voice lab (preview, delivery sliders saved to Chatterbox, add voices with gap removal) | Upstream's UI exposes four providers and many options that don't apply | `ui/chatterbox_ui.py` |
| 8 | Searchable library picker, "Title — Author" from each EPUB's metadata, cached in `library_index.json` | Typing a title beats uploading files; 1,517 books: first build ~13 s, refresh ~5 s | `ui/library_index.py` |
| 9 | Preview writes nothing | #186 dropped `cover.jpg` into the output folder even when previewing, leaving cover-only "books" in the library | `core/audiobook_generator.py` |
| 10 | Chapter checkboxes with automatic selection; ticked chapters renumbered 1…n | Front/back matter (title pages, copyright, contents, acknowledgements, "also by", newsletter pages) was being narrated | `core/chapter_selection.py`, UI |
| 11 | Paced narration: paragraphs from HTML block elements; one request per sentence-sized unit (sentences under 40 characters join the next); inserted pauses, 0.35 s after sentences and 0.9 s between paragraphs by default; at speeds other than 1.0 the finished chapter is stretched once, pauses included | Upstream collapsed paragraph breaks (the OpenAI provider's break marker was whitespace), and long requests left only ~0.2 s between sentences | `tts_providers/openai_tts_provider.py`, parser |
| 12 | Single M4B per book: chapters in a hidden `.chapters` folder, merged only when all succeed; AAC chapters joined by stream copy, chapter markers, cover, tags | BookOrbit mishandles multi-file books; the library never sees a half-made book; failed books can be resumed | `core/m4b.py`, generator |
| 13 | Book queue: persisted, runs one book at a time, pause/resume/stop, remove, retry; a book interrupted by a restart resumes first | Queue several books | `ui/job_queue.py`, UI |

Chapter auto-selection is adapted from abogen's scoring (MIT; see `THIRD_PARTY_NOTICES.md`) with four
changes: short sections only count against a chapter inside the runs at the start or end of the
book; "chapter/part/book" marks real content only at the start of a title; text keywords only
count on sections under 3,000 characters; anything over 20,000 characters is always story. It was
checked by hand on 120 real books: in an 80-book sample 449 of 4,186 sections were unticked, all
non-story after one fix (a novel stored as a single section that begins with its copyright line).

## 6. Measurements

| What | Number |
|---|---|
| Speech rate, stock "Elena" voice, 1.0× | 20.2 characters of text per second of audio (5 chapters, all within ±1%) |
| Book generation speed | ~1.8× real time before paced narration (a 30-minute chapter ≈ 17 minutes); ~1.6× with it |
| Paced narration cost | +14% generation time; audio +18% longer (the pauses) |
| Pauses before pacing | Chapter of 424 sentences: 438 pauses ≥ 0.12 s, of which 212 under 0.2 s and 41 at 0.5 s or more |
| Same passage, A/B | Before: 83 s audio, 0 pauses ≥ 0.6 s. After: 98 s, 14 pauses ≥ 0.6 s |
| Very short inputs ("Yes.", "Hmm?") | 0.36–1.32 s of audio; no runaway generations |
| Library index | 1,517 EPUBs; first build ~13 s; refresh ~5 s; parsing one book 0.04–0.19 s |
| Tests | 404 passing (294 before multi-voice, section 13; 241 before Kokoro + voice deletion, section 11), plus 44 for `chatterbox/` (135 before the review fixes; upstream baseline: 28 tests, 2 erroring on a mock that closed stdout) |

## 7. Known limitations and open questions

| ID | Item |
|---|---|
| K1 | Resolved 2026-09-28 (F-06): for an M4B the chapters are AAC and are stream-copied into it, so there is one lossy encode. Was: MP3 chapters re-encoded to AAC. |
| K2 | Time estimates are calibrated on one voice; other voices speak at different rates. |
| K3 | A very short, untitled opening section (under ~1,000 characters) is unticked; real openings titled "Chapter…" or "Prologue" are always kept. |
| K4 | Some collections repeat a title/author/copyright header inside every story; it is narrated. Search & replace can remove it; nothing automatic yet. |
| K5 | The non-paced path (CLI default) still sends 1,800-character requests, so the 1000-token cap can cut words. |
| K6 | One failed request fails the whole chapter; the chapter's finished units are discarded. Partly resolved (F-02): connection errors and 502/503/504 are retried for up to 10 minutes, so a Chatterbox restart is waited out. |
| K7 | Per-request overhead: a 30-minute chapter is several hundred requests, each paying the server's fixed cost. Partly resolved (F-05): Advanced → Narration units → Whole paragraphs sends about a quarter as many requests; experimental until checked on real chapters. |
| K8 | Queue: all books log to one shared file; progress counts finished chapter files only. Resolved (F-19, F-29): job processes are spawned, not forked, and stale chapter files from an earlier run are not counted. |
| K9 | The library index walks the whole library on every page load (~5 s over a Windows bind mount). |
| K10 | Pauses, M4B output and chapter selection are UI-only; the CLI has no flags for them. |
| K11 | `requirements.txt` pins every dependency; the local image's Gradio 5.50 vs. upstream's image 5.33 is a deliberate pin, not drift. |
| K12 | Security: the UI has no login and is reachable on the LAN (the owner's choice); the Voice lab writes into the voices folder and changes Chatterbox's global settings. Since 2026-09-28 the book and the output folder must be inside the library and output folders (F-21), and Chatterbox refuses cross-origin browser calls (F-55). |
| K13 | Upstream's `web_ui.py` stays in place to keep upstream merges easy; `main_ui.py` uses `chatterbox_ui.py`, which reuses helpers from it. |
| K14 | Resolved (F-35): `.gitattributes` now forces LF for `*.sh`. Was: upstream's `* text=auto` gave Windows checkouts a CRLF `entrypoint.sh`, and the container failed to start. |
| K15 | The OpenAI provider logs "Unsupported model name … unable to retrieve the price" for every chapter; the cost estimate is meaningless for a local server. |
| K16 | Voice lab "Save for books" changes Chatterbox's global settings at once, including for a book in progress and for BookOrbit. |
| K17 | Resolved (F-27): units are generated at 1.0 and the finished chapter is stretched once, pauses included. Was: Chatterbox stretched each unit separately. |

## 8. Run, test, deploy

- Tests (need ffmpeg): `python -m unittest discover -s tests -t . -p "*test*.py"`. The owner runs
  them inside the image: `docker run --rm --entrypoint python3 -v <repo>:/src -w /src
  epub_to_audiobook:local -m unittest discover -s tests -t . -p "*test*.py"`.
- Deploy both containers: `docker compose -f docker-compose.chatterbox.yml up -d --build`, with
  this machine's paths in `.env` (see `.env.example`). The owner's stack folder has a small
  `compose.yaml` that includes the repo's file with its own `.env`. Rebuilding the app is quick;
  rebuilding Chatterbox only redoes changed layers unless its requirements change. Restarting the app
  while a book is generating stops it, and the queue resumes it on the next start; restarting
  Chatterbox pauses the book, since the app waits up to 10 minutes for it (F-02). Compose starts the
  app only once Chatterbox's healthcheck reports its model loaded.
- Chatterbox's settings: on a first run, copy `chatterbox/config.audiobook.yaml` (this deployment's
  tuned values) into `CHATTERBOX_DATA` as `config.yaml`; `chatterbox/config.yaml` is upstream's
  template. The whole `CHATTERBOX_DATA` folder is mounted so settings saves are atomic (F-52).
- Chatterbox upstream updates: `git subtree pull --prefix=chatterbox
  https://github.com/devnen/Chatterbox-TTS-Server.git main --squash`, then re-check the patches.
- Upstream updates: `git fetch upstream && git merge upstream/main`, then run the tests and rebuild.
- On Windows, set `git config core.eol lf` before checking out (K14).

## 9. Lessons

- Put a size guard on any test that sends text to Chatterbox. A paragraph-splitting bug once sent a
  whole 32,000-character chapter as one request; because the server is serial, it was blocked for
  ~15 minutes until restarted.
- Measure instead of guessing: pause lengths with ffmpeg `silencedetect`, speech rate from finished
  chapters, throughput from the server's token logs. Two early theories ("chunk crossfades cause the
  tin-can sound", "buffering causes the mid-sentence pauses") were wrong.
- Keep real library titles out of test fixtures and docs.
- A stream copy keeps whatever timestamps it is handed. Test audio joins with realistic,
  variable-bitrate audio: a constant tone hid the AAC chapter-join bug in section 10.

## 10. Review fixes (2026-09-28)

The cloud review (`REVIEW_FINDINGS.md`, `REVIEW_FINDINGS_CHATTERBOX.md`) was carried out in five
packages (text, generator and M4B, narration, queue and UI, Chatterbox and deployment), each built
by a junior agent in its own worktree and reviewed and merged by the PM. The status of every finding
is at the top of `REVIEW_FINDINGS.md`. What changes in use:

- M4B books: chapters are generated as AAC and stream-copied into the book.
- Speed: units are generated at 1.0 and each finished chapter is stretched once.
- Units over 450 characters are split at punctuation (or a space), so none reach the token cap
  (F-07); symbol-only units such as scene breaks are dropped (F-18).
- The app waits out a Chatterbox restart (F-02).
- Chatterbox: generation runs in a worker thread under one lock, so health checks, the voice list
  and settings saves answer during a book; model reload and unload wait for that lock; settings saves
  are atomic; cross-origin browser calls are refused; a warning is logged when a chunk hits the
  1000-token cap (F-56). The model package is pinned to `chatterbox-v2@cc03573`; bump it
  deliberately.
- New option: Advanced → Narration units → Whole paragraphs (experimental, about 8% faster).

Caught while integrating: F-06's stream copy placed each AAC chapter join where ffprobe *estimated*
the previous chapter ended. ADTS AAC has no duration header, so the estimate comes from the bitrate,
and it was off by up to 8% per chapter: chapter markers drifted, and ffmpeg squashed the overlapping
audio packets to zero length. Chapter lengths are now summed from the packets and written into the
concat list. The F-06 test used a pure tone, whose constant bitrate hides the problem; the new test
uses a chapter that starts loud and ends quiet.

## 11. Kokoro engine and voice deletion (2026-09-28)

**Kokoro as a second engine.** Opt-in via `KOKORO_BASE_URL` (compose passes it through as
`${KOKORO_BASE_URL:-}`); when unset the Make tab shows no Engine choice and everything behaves
exactly as before. When set, an Engine dropdown next to Voice switches between Chatterbox and
Kokoro; the Voice dropdown swaps to Kokoro's English voices (`af_`/`am_`/`bf_`/`bm_` prefixes;
other prefixes are other languages and are excluded), labelled e.g. "Heart · American female · A",
best grade first, from its live `/v1/audio/voices`, falling back to just the configured (or
`af_heart`) default with a warning if Kokoro can't be reached. A **Sample** button next to Voice
auditions the selected engine/voice/speed with a fixed phrase; for Chatterbox it goes through the
same saved delivery settings a real book uses. Kokoro always narrates by sentence (never
paragraph mode): its gap detector was tuned on Chatterbox audio and buys nothing on a server this
fast (below). The chapter table, chapter summary and queue estimate all follow the selected
engine. Queue jobs persist `"engine"`; a job queued before this change has no such key and builds
as Chatterbox. `GeneralConfig` gained `openai_base_url` (`None` = today's `OPENAI_BASE_URL`
behaviour); `OpenAITTSProvider` now passes it to the `OpenAI` client explicitly, since a container
can have both `OPENAI_BASE_URL` (Chatterbox) and `KOKORO_BASE_URL` set at once and the ambient env
var must not win for a Kokoro-selected book. Voice-lab/Make-tab syncing and `add_voice`'s
Make-tab update are now engine-gated, so a Kokoro id can never land in the (Chatterbox-only)
Voice lab and a Chatterbox file name can never land in the Make-tab dropdown while Kokoro is
selected.

**pydub vs. Kokoro's streaming WAV header (verified, no workaround needed).** Kokoro's
`response_format: "wav"` sets the RIFF and data chunk sizes to the streaming placeholder
0xFFFFFFFF; Python's stdlib `wave` module would misreport the length from that (it trusts the
header literally). Checked with 6 real Kokoro WAV responses against `ffprobe` on the same bytes:
`AudioSegment.from_file(io.BytesIO(content), format="wav")` (the exact call the paced path
already uses) matched `ffprobe`'s duration to within 0.5 ms every time -- pydub does not take the
naive stdlib-`wave` fast path here, so the existing decode line needed no change for Kokoro.

**Kokoro estimate constants**, measured live 2026-09-28 against `af_heart` through the real paced
path (`paced_units` + `OpenAITTSProvider`, sentence pause 0.35s / paragraph pause 0.9s, speed
1.0), the same method as Chatterbox's own numbers: two invented paragraphs (6 sentences, 355
characters of speech) sent as 6 real paced requests took 1.22s wall time for 20.95s of audio
(`ffprobe`); the same text as one undivided request took 0.41s for 21.72s of audio -- 52.9x real
time, close to the ~0.09s-per-5s figure from the earlier live check. 355 / 20.95 = 16.9
characters/s (Chatterbox: 20.2). The paced run took 3.07x the wall time the bulk rate implies for
that much audio: Kokoro generates so fast that the fixed per-request overhead of many small
sentence requests dominates far more than it does for Chatterbox (14%), which is exactly why it
always narrates by sentence rather than adding paragraph mode's complexity for a saving that
would be swamped by that overhead anyway. Single small sample, one voice; not re-validated across
books or voices. `KOKORO_CHARS_PER_AUDIO_SECOND = 16.9`, `KOKORO_GENERATION_SPEED = 52.9`,
`KOKORO_PACED_GENERATION_OVERHEAD = 3.07` (`ui/chatterbox_ui.py`).

**Voice deletion (Voice lab).** A new "Delete a voice" section: a dropdown of the owner's own
voices (`.wav`/`.mp3` files directly in `TTS_VOICES_DIR` that aren't one of Chatterbox's 28
built-in voices -- `BUILT_IN_VOICES`, checked by a test against `chatterbox/voices/`), a "Delete
voice" button and a status line. The button's `js=` runs a browser `confirm()` first; cancelling
returns `null`, which the Python handler treats as a no-op. Refused (file untouched): a built-in
voice, a name that isn't a plain existing file directly inside `TTS_VOICES_DIR` (no path
separators or traversal), or a voice a queued/running Chatterbox job still uses (names the book).
After a successful delete, the Voice lab dropdown, the delete dropdown and the Make-tab dropdown
(only while Chatterbox is selected there) are refreshed, reselecting the default only if the
deleted voice was the one showing. Adding a voice also refreshes the delete dropdown now, and so
does page load.

Live-verified against the real `kokoro` container over `blackcat-net` (measurement + one smoke
run through the production `build_config` -> `get_tts_provider` -> `text_to_speech` path): 775
characters sent in total. Tests: 294 passing (was 241; +53), plus 19 for `chatterbox/` (unchanged).

## 32. Multiple EPUB chapters inside one text file (2026-09-30)

The owner reported that *Apex Prey: Polly* came through as one chapter. The library copy is named
`Lesley A. Camphouse/Apex Prey/01. Apex Prey (2025).epub`. Its main `c2T.xhtml` contains seven numbered
chapters plus "Self-Preservation", with all eight boundaries present in both the EPUB contents and
the headings. The parser previously returned one chapter per spine document, so chapter selection
picked one approximately 95,000-character story row titled "1 My Birth".

**Fix:** `EpubBookParser._chapter_sections` groups the contents links by document and splits documents
with at least two distinct resolved targets. It uses EbookLib's contents tree, which already excludes
page-list and landmark links and supports EPUB3 navigation and NCX-only navigation. Targets inside a
heading move to the start of that heading, preserving its number and title. Duplicate and missing
targets are ignored; sections follow document order even when contents links are out of order. A
preamble is retained, and documents without multiple usable targets retain the previous behavior.
All sections use the existing paragraph marking, text cleanup and title modes. No heading guesses,
new dependency or model request was added.

**Verification:**
- New synthetic regressions failed before the fix. They cover three sections inside one file, nested
  contents, anchors inside headings, encoded fragments, page-list exclusion, inline words, all three
  title modes, duplicate/broken/external links, preamble retention, legacy named anchors, and NCX-only
  contents with file-start links and unheaded sections.
- **Polly:** 7 parser rows became 14 (front matter included); automatic selection now picks all
  **8 story sections** with their proper headings. Comparing the combined text before and after,
  ignoring whitespace and paragraph markers, confirms no content was lost or duplicated.
- **Apex Prey 2:** all 18 parser rows and texts are unchanged; 10 story chapters selected.
- **The Reaping:** all 17 parser rows and texts are unchanged; 9 story chapters selected.
- **Robinson Crusoe:** its three contents-linked front-matter sections now separate, so 23 parser rows
  become 25. Its 20 story chapters remain 20. The integration expectation documents that change.
- **33 parser tests and all 702 application tests pass.** The full suite ran in the existing image
  against read-only working source, with temporary logs and networking disabled. `git diff --check`
  passes.

The earlier five cast fixes (§31) and this parser fix are committed separately. No services were
rebuilt or restarted, and existing saved casts, ebooks and audio were not rewritten. Books without
usable contents boundaries still use their original document boundaries.
