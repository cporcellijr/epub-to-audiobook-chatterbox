# Chatterbox edition: work log and findings

Covers 2026-09-25 to 2026-09-30. Written for the owner and for any agent reviewing or continuing
this project. Personal library details (book titles, authors) are deliberately left out.

Since 2026-09-27 this repo holds the whole stack: the audiobook app at the root and the Chatterbox
server in `chatterbox/` (upstream commit 915ae28 as a git subtree, then one commit of local patches),
deployed together by `docker-compose.chatterbox.yml` as two containers.

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

## 2. Chatterbox server (`chatterbox/`)

### 2.1 Behaviour this app depends on (verified)

- **One generation at a time.** Until 2026-09-28 synthesis blocked the server's event loop, so even
  `GET /v1/audio/voices` waited behind a running synthesis (it timed out at 5 s in testing). Now
  synthesis runs in a worker thread and one engine lock keeps generation serial, so other endpoints
  answer while a book is generating (F-10, F-54). The app still uses one worker and reads the voice
  list from the mounted voices folder.
- **`/v1/audio/speech` chunking.** Input is split at sentence boundaries into chunks of up to 500
  characters (Original; 400 for other models). Each chunk is generated independently and they are
  stitched with a 200 ms crossfaded pause (`SENTENCE_PAUSE_MS`).
- **1000-token cap per chunk.** The Original model stops at 1000 speech tokens (≈ 40 s of audio).
  A slow voice on a long chunk loses the end of the text; this happened to 4 of 629 requests
  before paced narration.
- **Delivery settings are live.** `generation_defaults` (exaggeration, cfg_weight, temperature,
  seed 888) in its `config.yaml` are read on every `/v1` request, and `POST /save_settings`
  changes them immediately. The request's `speed` is applied after generation (ffmpeg atempo).
  The `speed_factor` default in that config only affects the server's own `/tts` endpoint.
- **Voice cache key = (file path, file mtime, exaggeration).** Replacing a voice file is picked up
  without a restart.
- **How a reference clip is used:** the first 6 s become the speech-token prompt (150 tokens) that
  sets delivery, rhythm, pauses and accent; the first 10 s set timbre; the whole clip gives the
  speaker embedding. The Turbo model refuses clips of 5 s or less.

### 2.2 Performance (RTX 4070, Original, BF16)

- Autoregressive token sampling is 80–90% of the time: ~48–60 tokens/s against 25 tokens per second
  of audio, so ~1.6–1.8× real time overall. The GPU sits at only ~50–60% (kernel-launch bound).
- Fixed cost per request: ~0.35 s in the audio decoder, plus request setup.
- Tried and rejected: fp16/bf16 autocast on the token model (slower), checking for end-of-speech
  every 8 steps (no gain), `cfg_weight = 0` (breaks this fork's unconditional batch slice).
- Not yet tried: CUDA graphs or `torch.compile` with a static KV cache for the per-token step.

### 2.3 Local patches (the commit after the subtree add: `git log -- chatterbox`)

| Patch | Why | Result |
|---|---|---|
| `aac` and `flac` output formats | BookOrbit's mobile app requests AAC; the server returned 422 | Works |
| Speed via ffmpeg `atempo` instead of librosa's phase vocoder | Playback at 1.25× sounded metallic ("tin can") | Clean at 1.25× |
| Chunk size 500 for Original (was 120) | One generation pass per paragraph instead of several | 426-char paragraph: 13.0 s → 10.6 s |
| Only request attention weights when an alignment analyzer exists (`patches/apply_speed_patches.py`) | Requesting them forced the slow eager attention path | +13–23% tokens/s |
| `torch.clear_autocast_cache()` after each generation | GPU memory leak: +461 live CUDA tensors (+248 MB) over 240 requests; host RAM reached 12 GB over hours | Flat after the fix |
| `TTS_BF16=auto` | Faster Original model | Enabled |

Until 2026-09-27 these patches existed only as uncommitted edits in a local clone, deployed through
a thin image (`FROM` the full build, `COPY server.py`). They are now committed here and the image is
built from `chatterbox/`. The review fixes of 2026-09-28 added more local changes (section 10);
`git log -- chatterbox` lists them all.

## 3. Voice reference clips (findings)

- **Pauses in the reference become pauses in the speech.** A clip whose first 6 s was about 60%
  silence (short bursts of words, ~1 s gaps) made Chatterbox pause mid-sentence. Same 218-character
  text: that voice 15.6 s with 3 pauses (4.4 s total); a fluent stock voice 7.2 s with none; the
  same voice with its gaps removed about 12 s with none.
- Removing gaps: ffmpeg `silenceremove` (gaps over 0.3 s at −40 dB shortened to 0.15 s). All custom
  voices were cleaned this way, and the Voice lab now does it on upload by default.
- A one-word clip is not enough; 10–15 s of continuous speech is the target.
- A non-English clip still produces English, with that language's accent.

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

## 12. F-45 measured: a compiled token loop (2026-09-28)

Measured in a throwaway container from the production image, through the production engine (BF16,
autocast, voice cache) with the book settings (Elena, seed 888); the live server was not touched.
Scripts: `experiments/f45/`.

- **Baseline.** Token sampling is 5.09 s of a 5.55 s request (92%): 21.1 ms per token. The audio decode
  is 0.44 s.
- **Prototype.** The stock token loop, unchanged except that the per-token transformer step reads a
  static KV cache (transformers `StaticCache`, 2,048 positions) and runs through
  `torch.compile(mode="reduce-overhead")`, i.e. CUDA graphs.
- **Speed.** Token step 20-25 ms to 7.7-8.3 ms (about 2.6x). Whole requests (`engine.synthesize`, three
  texts, three runs each): 1.75x to 4.12x real time, **2.35x faster**. That is about 58% less
  generation time: a 10-hour book drops from about 6 hours to about 2.5. The review's guess of 15-30%
  was far too low. Compile is a one-time 17 s; peak VRAM in the test process was 3.2 GB.
- **Same model.** With the stock token sequence forced through both paths, the compiled path's top
  token matched stock at 94-97% of steps and was always in stock's top five (mean KL 0.0017). An
  uncompiled run on the same static cache differs from stock by the same amount, so the difference is
  bf16 rounding from the cache layout, not compilation. Sampled output drifts from stock after a few
  tokens (same distribution, different draws); durations stayed within about 5%.

Before building it for real: keep the compiled step and cache in `engine.py` for the life of the
process, for the Original model only, behind an on/off setting that falls back to the stock loop;
compile during model load (about 17 s more start-up); run synthesis on one dedicated thread, since the
server now synthesizes on a threadpool (F-10) and torch.compile's CUDA graphs are recorded per thread
(check); listen to A/B samples of the same text before switching it on.

**Built and validated (2026-09-28); on in the owner's deployment (`TTS_COMPILE=on`).** The
production version lives in `chatterbox/fast_t3.py` and is installed by `engine.py` behind
`TTS_COMPILE=on|off` (default off; compose passes `${TTS_COMPILE:-off}`), for the Original model with
`TTS_BF16` on and CUDA. It replaces `T3.inference` on the loaded model with the same sampling loop
over a compiled static-cache step, keeps the stock loop as the fallback for `cfg_weight=0`, over-long
prompts, other models and any raised error (one WARNING, then DEBUG), runs warm-up, every generation
and teardown on one dedicated worker thread (CUDA graphs are per thread), and is torn down on unload
and reinstalled on reload. `docs/chatterbox-edition/experiments/f45/validate_build.py` checks speed
(>= 2.0x), the teacher-forced match, recompiles, threads, memory over 240 requests, the fallback and
start-up time, and writes A/B WAVs; the setting stays off until it passes on the real machine.

GPU validation of the built version (`validate_build.py`, all 8 checks pass): start-up with warm-up
20.7 s; 2.37x faster whole requests (1.72x to 4.08x real time); teacher-forced agreement 96.5-100%,
always in stock's top 5, mean KL 0.0014-0.0018; no recompiles over 50 requests of varied length;
requests from 8 threads replay the same graphs with flat latency (0.92-0.96 s); zero growth in CUDA
tensors or allocated memory over 240 requests; a forced failure falls back to the stock loop with one
warning; peak VRAM 3.2 GB. Two bugs in the first version of the script itself (thread idents reused
by threads started one after another; a dead weakref proxy crashing the tensor count) were fixed
before this run. The owner's blind A/B listen preferred the compiled takes on 2 of 3.

Deploying it needed one compose change: with `runtime: nvidia` as well as the `deploy.resources` GPU
reservation, the container got the driver's `libcuda.so.1` only inside a WSL driver folder and no
`libcuda.so`, so Triton could not link its helper ("cannot find -lcuda") and the server fell back to
the stock loop, as designed. The reservation alone provides `/usr/lib/x86_64-linux-gnu/libcuda.so`,
the same as `docker run --gpus all` in the validation runs; `runtime: nvidia` is gone. Live afterwards:
warm-up 14.9 s; repeated 200-character requests over HTTP 3.5-3.7x real time; a paced 1,159-character
chapter through the app 18.7 s (the old estimate said 36 s). The app's estimate now assumes the
compiled loop: 3.87x real time for one long request, and sentence-sized requests take 1.26 times as long.


## 13. Multi-voice narration (2026-09-28)

Built from `docs/chatterbox-edition/MULTIVOICE_BUILD_BRIEF.md` on branch `feature/multivoice`, without a
GPU or an LLM: code, 110 unit tests and a validation script for the owner's machine. Nothing here has
run against a real LLM or Chatterbox yet; section 13.4 says what to run.

### 13.1 What it does

A **Voice mode** on the Make tab: *Single voice* (default; sends exactly the requests it sent before,
proven by a fixture test that pins the pre-change unit list), *Narrator + dialogue voice* (every quoted
line in a second voice, no LLM), and *Cast* (shown only when `LLM_BASE_URL` is set). Cast mode adds an
**Analyse cast** button that queues the LLM pass as a queue job of its own kind (`kind: "cast"`); the
queue shows its progress ("analysing cast · 3 of 12 chapters"), and books queued after it wait. When it
finishes the panel shows the cast table (character, lines, gender, age, voice, other names); clicking a
row opens a small editor (gender, voice dropdown for the job's engine, the existing Sample button) and
**Save** writes the change to the cast file. **Add to queue** validates the cast (finished; voices
belong to the engine), records the narrator voice in it and copies a snapshot into `queue_uploads/`
for the job, so later edits or a re-analysis never change a queued book.

### 13.2 Design

- **Dialogue splitting** (`core/dialogue.py`, no LLM): per chapter, the quote style is detected once
  (double, single, em-dash, or none; the style with more openings wins, and the other mark is plain
  text, so a single quote nested in double-quoted speech stays inside it). Straight quotes open only
  after a separator, so apostrophes never open speech; a single-quote mark followed by a letter is an
  apostrophe, and one straight after a letter is a possessive when the speech still has a closing mark
  later (`James' hat`). A quotation left open at a paragraph's end continues into a next paragraph that
  opens with speech; the continuation is a separate line marked `continues`, and attribution gives it the
  previous line's speaker without asking. Output: ordered `Segment(kind, line_id, text, continues)` per
  paragraph, line ids 1.. within the chapter; paragraph numbering matches `paced_units` exactly.
- **Units follow speakers** (`openai_tts_provider.py`): the sentence packer and the paragraph-mode
  packer were factored out (`_sentence_units`, `_paragraph_units`) and `paced_units` /
  `paragraph_mode_units` call them per paragraph as before. `voiced_units` / `voiced_paragraph_units`
  call them per *segment* and tag each unit with a voice, so a unit can never span two voices; the
  40/400/450-character rules apply inside a segment. Both paced paths now feed one `_speak_units` loop
  that requests each unit with its own voice; single voice mode still calls the untouched `paced_units`
  and the fixture test compares the resulting requests to the recorded pre-change list.
- **Per-unit voice rule** (`OpenAITTSProvider._voice_of`): narration -> `voice_name`; a quoted line ->
  its character's voice when the cast knows the speaker and the character has a voice, else
  `dialogue_voice`, else the narrator. Attributions are looked up by the chapter text's SHA-1, the same
  hash the chapter manifest (F-11) uses, so a different chapter selection or renumbering still finds
  them; an unanalysed chapter logs one warning and speaks its quotes with the dialogue voice. Tagging,
  M4B, resume and retry are untouched (the provider is built per chapter exactly as before).
- **Attribution** (`core/cast_llm.py`): windows of `WINDOW_LINES = 20` asked lines (or
  `WINDOW_MAX_CHARS = 6000` of text, whichever comes first) of consecutive paragraphs, preceded by
  `CONTEXT_PARAGRAPHS = 2` unmarked paragraphs; asked lines are rendered `[#12] "..."`. All prompt text is
  in `PROMPTS`. The reply must be a JSON object whose `speakers` cover exactly the asked ids (missing or
  invented id, non-string name, no JSON: `AttributionError`); code fences and `#12`/`[#12]` keys are
  tolerated, bad gender/age values become `unknown`. One retry, then the window's lines are unknown.
  `ChatClient` uses `response_format: json_object` until the server rejects it (HTTP 400), then goes on
  without. Temperature 0, one request at a time, 300 s timeout.
- **Aliases** (`Roster`): keys are the normalized first-seen name (titles like Mr./Mrs./Dr. stripped,
  unless that would leave nothing: "Mother" stays "Mother"); the display name grows to the fullest form
  seen. Merging is deliberately conservative: a first name and its fuller form merge (prefix, or a table
  of common English short forms so that Tom / Thomas, Bill / Will / William, Peggy / Margaret meet), two
  people of different known genders never merge, an ambiguous first name (two Annes) stays separate, and a
  bare surname is never merged by code ("Mrs. Marsh" next to "Ada Marsh" is usually the mother). The
  model is asked for aliases, which do merge. Reasoning: a wrong merge gives a main character the wrong
  voice for a whole book; a split shows two rows the owner can give the same voice.
- **Cast file** (`core/cast.py`): `casts/<key>.json` in the app data folder, key = SHA-1 of the EPUB's
  bytes (an upload and a library copy share one cast). Holds book title/author, engine, narrator voice,
  status (running/done/failed + error), progress, characters (name, aliases, gender, age, lines, voice),
  per-chapter `{text hash: {number, title, lines {id: key|null}, unknown}}` and stats (windows, first
  replies unusable, still unusable after retry, lines, unknown lines, seconds). Rewritten atomically after
  every chapter, so the UI can show progress and a crash keeps what was done.
- **Voice suggestions** (`suggest_voices`): characters by line count; each gets an unused voice of its
  gender (neutral fits anyone, unknown takes anything), never the narrator's; only when the suitable
  voices run out is one shared, least-used first. Kokoro genders come from the id prefix; Chatterbox
  genders from `voice_genders.json` (app data), set in the Voice lab's new "Voice gender" row; an unset
  voice is neutral. No gender is ever guessed from a name.
- **Unload / reload** (`core/chatterbox_control.py`, `core/cast_analysis.py`): the analysis process
  POSTs `/api/unload` when `LLM_UNLOAD_CHATTERBOX` is on (default), runs the pass, and in `finally`
  POSTs `/restart_server` (which blocks until the model is loaded) and then polls `/api/model-info`
  until `loaded` (up to 900 s: a cold cache downloads the model). The queue asks
  `ready_for_book(settings)` before starting a Chatterbox book: if `/api/model-info` says unloaded it
  refuses and kicks off one background reload (covers an analysis killed before its reload); an
  unreachable Chatterbox does not hold the queue (the book's own F-02 wait covers a restart). Kokoro
  books never wait.
- **Settings**: `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`, `LLM_UNLOAD_CHATTERBOX` through compose as
  `${VAR:-}` (`on` default for the last), documented in `.env.example` and the README, read per call.
- **Estimates**: the analysis job's queue time is `dialogue lines x ANALYSIS_SECONDS_PER_LINE (0.5)`,
  a placeholder until the validation script reports the real speed; `chapter_stats` gained a fourth
  element (dialogue lines) and older 3-element stats still work.

### 13.3 Build or borrow

Everything was built fresh; the licence stays MIT. prakharsr/audiobook-creator (GPL-3.0, checked) was
read for ideas only: its per-line attribution loop with a running character list and structured
outputs, its "insert / update / merge" character operations (here: `Roster.add` with alias merging), and
its lesson that small models drift from the schema (here: strict validation, one retry, then unknown).
Its code depends on pydantic-ai and a different text pipeline (whole-book JSONL of lines), so borrowing
would have meant heavy rework plus relicensing a public MIT repo for no saving.

### 13.4 Tests, and what is not verified

404 app tests pass (294 before; the 110 new ones are in `tests/audiobook_generator/dialogue_test.py`,
`multivoice_provider_test.py`, `cast_llm_test.py`, `cast_test.py`, `cast_analysis_test.py`, plus new
cases in `job_queue_test.py` and `chatterbox_ui_test.py`). They cover: quote splitting on invented
passages (straight/curly/single/dash, apostrophes, possessives, nesting, multi-paragraph speech, style
detection); units never spanning a voice change and the pre-change unit list for single voice; per-unit
voices through the provider (dialogue, cast, unknown speaker, unanalysed chapter, pauses); window
building by line count and character budget; reply parsing (good, fenced, broken, missing/invented ids);
alias merging rules; retry-then-unknown and continued lines; cast persistence and suggestions; the unload
-> analyse -> reload order including a failing analysis; the queue never starting a book while unloaded;
old jobs (no `kind`, no voice-mode keys) building as single-voice books; `queue_settings` validation and
the cast snapshot; the cast panel and editor.

## 14. Adaptive delivery (2026-09-28)

Dialogue lines that are whispered are read softer and quieter, shouted lines more excited and a
little louder, around a per-book baseline. Built on branch `feature/adaptive-delivery`, no GPU
needed for the code itself; the accuracy measurement (13.3 below) ran against the real Chatterbox
and the local `qwen2.5:14b` LLM.

### 14.1 Presets and the peak guard (`core/delivery.py`, new)

Three moods -- `soft`, `normal`, `excited` -- each a `(exaggeration, cfg_weight, temperature,
gain_db)` preset computed from the book's own baseline sliders `(b_exag, b_cfg, b_temp)`, approved
by ear at the owner's baseline (exaggeration 0.73, CFG 0.5, temperature 0.61):

```
soft:    exaggeration = max(0.25, round(b_exag * 0.48, 2))   cfg = max(0.2, b_cfg - 0.15)   temperature = max(0.3, b_temp - 0.11)   -6 dB
normal:  b_exag, b_cfg, b_temp unchanged                                                                                              0 dB
excited: exaggeration = min(1.0,  b_exag + 0.27)              cfg = max(0.2, b_cfg - 0.1)    temperature = min(0.9, b_temp + 0.09)    +1.5 dB
```

At the approved baseline these reproduce the approved numbers exactly (soft 0.35/0.35/0.5, excited
1.0/0.4/0.7) -- tested. Every unit spoken with adaptive delivery on gets a peak guard
(`delivery.peak_guard`, `PEAK_GUARD_DBFS = -1.0`): if its peak exceeds -1 dBFS after the mood's gain
it is turned down to exactly -1 dBFS. A +2 dB version of the excited preset was tried first and
audibly clipped on the owner's voice; -1 dB did not, hence the guard threshold.

`delivery.saved_chatterbox_defaults()` reads Chatterbox's own saved generation defaults
(`CHATTERBOX_CONFIG`'s `generation_defaults`, one retry after F-52's short delay) for whichever of
the three baseline sliders a book doesn't override, falling back to the approved baseline when the
file can't be read. It deliberately duplicates `ui.chatterbox_ui`'s read of that file rather than
importing it: `chatterbox_ui` already imports the TTS provider, so the reverse import would be
circular.

### 14.2 Rule-based mood cues (no LLM)

`delivery.mood_of(before, quote_text, after)` looks at the narration right before a dialogue line
(only when it is a lead-in speech tag, i.e. it ends in `,` or `:` -- "she whispered," but not an
unrelated sentence that happens to precede the quote), the narration right after it, and the
quotation's own text:

- soft cues: whispered, murmured, breathed, hissed, muttered, mumbled, and the adverb phrases
  softly / quietly / gently / under (his|her|their) breath / in a whisper / in a low voice;
- excited cues: shouted, yelled, screamed, shrieked, roared, bellowed, cried (out), exclaimed, and
  the adverbs loudly / angrily / furiously; a quotation whose last sentence ends in `!` is excited
  unless a soft cue applies -- soft always wins, including over `!`.

`delivery.segment_moods(paragraphs)` applies this per dialogue line of a chapter (mirroring
`speech_tags.tagged_speakers`'s paragraph walk): narration is always `normal`; a continued
quotation (`Segment.continues`) inherits the previous line's mood outright, ignoring its own
surrounding text entirely (tested with a line whose own narration would say "excited" on its own
but which must still inherit the previous line's "soft").

### 14.3 Cast mode: LLM moods measured, then turned off

The attribution prompt (`core/cast_llm.py`) can ask the model for an optional per-line mood
alongside the speaker (`PROMPTS["moods_shape"]`/`PROMPTS["moods_rule"]`, gated by the module
constant `ASK_LLM_FOR_MOODS`): `parse_reply` accepts a `"moods": {"N": "soft|normal|excited"}` for
the asked ids, and a missing or invalid value becomes `"normal"` rather than an error (unlike
`"speakers"`, which stays strict). `attribute_chapter` now returns `(lines, moods)`: a rule cue
(from `delivery.segment_moods`, recomputed for the whole chapter once) always overrides the LLM's
guess; a line never asked (speech-tag anchored, or continuing another line) gets the rule mood only.
Moods are saved per chapter in the cast file (`chapters[hash]["moods"]`) and the cast summary the UI
shows after analysis now adds a "N soft, M excited" count when either is non-zero.

**Measured against the labelled multivoice fixture** (`docs/chatterbox-edition/experiments/multivoice/fixture`,
270 dialogue lines, real `qwen2.5:14b` over `blackcat-net`, Chatterbox unloaded for the pass):
adding the moods clause to the prompt dropped speaker attribution from the current baseline
**221/270 to 198/270** -- well past the 219/270 floor this task set for reverting. `ASK_LLM_FOR_MOODS`
is therefore **`False`** in the shipped code: cast-mode moods come from the rule cues only (the
same `delivery.segment_moods` the non-LLM voice modes use), plus whatever the speech-tag anchors
and continuations resolve to. Rule cues on the fixture: 266 normal / 3 soft / 1 excited (it is
mostly plain dialogue, as expected of invented sample text); with `ASK_LLM_FOR_MOODS` on, the LLM
additionally reassigned 5 lines to soft and 5 to excited that the rules had called normal -- not
enough lines to justify a ~10-point accuracy cost. `parse_reply`/`attribute_chapter` still accept
and merge a `"moods"` reply correctly (tested) if the constant is ever switched back on after a
better-worded prompt is measured not to cost accuracy; the diagnostic script now also reports
`MOODS_FROM_RULES`/`MOODS_ADDED_BY_LLM` so a future attempt can be checked the same way.

### 14.4 Provider (`tts_providers/openai_tts_provider.py`)

New `GeneralConfig` fields: `adaptive_delivery` (bool) and `delivery_exaggeration` /
`delivery_cfg_weight` / `delivery_temperature` (`None` = Chatterbox's saved defaults, today's
behaviour). `_is_chatterbox_engine()` (`config.model_name == "chatterbox"`, exactly what
`chatterbox_ui.build_config` sets) gates everything below, so Kokoro and the real OpenAI API are
never affected regardless of what a config carries (tested directly, independent of
`build_config`'s own zeroing, as defence in depth).

- **A baseline with adaptive delivery off** sends the resolved baseline triple via the OpenAI SDK's
  `extra_body` on every request (today's unit building, unchanged) -- so a per-book "voice" works
  without the mood swings.
- **Adaptive delivery on** always builds units inside narration/dialogue segments (new
  `adaptive_units`/`adaptive_paragraph_units`, mirroring `voiced_units`/`voiced_paragraph_units`
  with an added per-segment mood), even in single voice mode (with the narrator voice for every
  segment, so a unit can still carry its segment's mood). Cast mode reads the cast's saved
  per-chapter moods for a dialogue line, with a rule cue detected fresh from the current text still
  overriding it, mirroring `attribute_chapter`'s own precedence; other modes are rules-only. Each
  unit's mood preset is sent via `extra_body`, and the preset's gain plus the peak guard are applied
  to the decoded audio before the configured pauses are joined. The mood is logged in the per-unit
  INFO line the same way `voice=` already is.
- **Neither applies**: byte-for-byte today's request (proven with a test that fails if the change
  is reverted: 6 of 9 new provider tests fail against the pre-change file, from a missing
  `extra_body` key to a wrong exported peak level).

### 14.5 UI

Make tab: an "Adaptive delivery" checkbox (default on, visible only while the engine is Chatterbox)
next to a live "Delivery for this book: exaggeration … · CFG … · temperature …, from the Voice lab"
line that follows the Voice lab sliders. `build_config`/`queue_settings` gained the four settings as
keyword arguments defaulting to off/`None`, so a job queued before this feature builds unchanged;
**Add to queue** captures the Voice lab sliders' *current* values as the book's baseline, so editing
the sliders afterwards never changes a queued or running book. Voice lab: a
"▶ Play soft / normal / excited" button plays the phrase three times through `/tts` at the three
presets around the current sliders (gain and peak guard applied, same as a real book), one clip with
~1 s gaps, reusing the existing preview temp-file handling.

### 14.6 Tests

473 app tests pass (421 before; 52 new, in `tests/audiobook_generator/core/delivery_test.py` (24),
`tests/audiobook_generator/tts_providers/delivery_provider_test.py` (9), and new cases in
`cast_llm_test.py` (8), `cast_analysis_test.py` (1) and `chatterbox_ui_test.py` (10)); 49 pass in
`chatterbox/` (44 before; 5 new, `chatterbox/tests/test_openai_speech_request.py`, pure Pydantic
validation, no GPU). They cover: the approved-baseline and relative presets (including both floor
clamps and both ceiling clamps); the peak guard (a loud clip guarded to exactly -1 dBFS, a quiet one
left alone, silence left alone); every mood cue (each soft/excited verb and adverb phrase, the
lead-in-must-end-in-,-or-: gate, soft winning over `!` and over an excited cue); continued lines
inheriting the previous line's mood regardless of their own text; reading Chatterbox's saved
defaults (full, partial, missing-then-retried); `OpenAISpeechRequest`'s new fields and their ranges;
LLM moods parsed and invalid ones defaulted to normal, rules overriding the LLM, lines never asked
getting rules only, and moods saved per chapter; the provider's baseline-via-extra_body with
adaptive off, mood presets and gain/peak-guard with it on (including the narrator-voice-for-every-
segment rule in single voice mode and the mood appearing in the log line), Kokoro's total exemption,
and adaptive-off-with-no-baseline sending exactly today's request kwargs; `build_config`/
`queue_settings` defaults, an explicit baseline travelling with a queued job, Kokoro zeroing the
baseline out, and old jobs without the new keys building with delivery off; the Make tab's baseline
line text, the engine toggle's checkbox/line visibility, the Play-soft/normal/excited button's three
presets and concatenated clip; and the cast summary's mood counts. Deploy: restart (bind-mounted
`./src`) plus, for the Chatterbox image, a rebuild (`chatterbox/server.py` changed).

Not verified without a GPU or an LLM: attribution quality of any real model, the `response_format`
fallback against a real server, the real unload/reload timing, how a two-voice book actually sounds
(short narration fragments such as "he said." are now their own requests, which Chatterbox may read
less steadily than a 40-character unit), and the analysis speed behind `ANALYSIS_SECONDS_PER_LINE`.
`experiments/multivoice/validate_multivoice.py` measures all of that: run it as its README says before
using cast mode on a book.

Measured on the owner's machine (Qwen2.5 14B through Ollama, the 270 labelled fixture lines): 207
right (76.7%) in the first build, with no unusable replies and Chatterbox unloaded and reloaded
around the pass. Every miss was the model's own answer, not alias merging. Qwen3 14B scored 74% at
about seven times the time, and asking the model to write its evidence per line first made it worse.
What helped (2026-09-28): `core/speech_tags.py` names the speaker of lines with a named speech tag
("said Tom", `Mrs. Marsh said, "..."`, and the other untagged quotations of such a paragraph)
without the LLM; those lines, and every line decided in earlier windows, are shown to the model as
`[Name] "..."` with six paragraphs of context, and it is asked only about the rest. Result: 221 right
(81.9%), misses down from 61 to 49; the tag rules named 23 fixture lines, all correctly (the fixture
is mostly untagged by design, so real books should gain more). Ten lines per request instead of 20
scored 218, so the window stays at 20.

## 15. Character profiles for picking voices (2026-09-28)

The owner asked for something like the KOReader X-Ray plugins to help choose cast voices.
[koreader-xray-plugin](https://github.com/0zd3m1r/koreader-xray-plugin) sends Gemini or ChatGPT only
the title and author and relies on the model having read the book; a local 14B model mostly hasn't,
and invents characters. [KoCharacters](https://github.com/nefelodamon/KoCharacters) sends page text,
which is the approach taken here. Two of its ideas are used (personality as lasting traits, not
events; a verbatim first-appearance quote). No code was borrowed.

### 15.1 What it does

`core/cast_profiles.py` (new) runs at the end of the cast job, after the last chapter's attribution,
while Chatterbox is still unloaded. The 15 most-spoken characters with at least 3 lines get one request
each. The request carries up to 6,000 characters of the paragraphs where the character speaks (shown
as `[Name] "..."`, as in the attribution windows) or where the narration names them (name, aliases,
first name; never a bare surname or a description's first word). It takes their first three passages,
then an even spread over the rest, with chapter headings and `[...]` for gaps. The model answers with
role, gender, age, description, relationships and a 4-12 word voice note; a gender or age the
attribution left unknown is filled in. Every character, however minor, gets the first line it speaks,
quoted by code. An unusable reply is asked again once, then skipped. Any other error (LLM down,
timeout) ends the stage and is noted in `profile_error`; the analysis still finishes as done.

Two corrections by code, both from the live run below:
- A voice-note clause naming an accent or origin the excerpts never mention is dropped. The model
  gave one character "a slight southern drawl"; no accent word appears anywhere in the book.
- A character with under a tenth of the most-spoken character's lines is "minor" (an antagonist stays
  one). The model sees one character's passages at a time, so it can't judge how big a part is.

UI: the cast table gains **Role** and **Sounds like** columns. Clicking a row shows the profile under
the editor: role, gender, age and line count, then the description, the voice note, relationships and
the first line. While the job runs, the status reads "Writing character profiles: n of m" after the
chapters. The summary counts the profiles and says if they stopped early. Casts analysed before this
have no profiles until they are analysed again (voice picks carry over).

### 15.2 Measured on the owner's machine

Run in the deployed container through the queue's own process target (`run_cast_analysis`: unload
Chatterbox, attribute, profile, reload) on scratch copies of the two real casts (a 3-chapter and a
4-chapter book, 218 and 291 dialogue lines, qwen2.5:14b through Ollama):
- All 10 profiles usable, no retries, no errors, in both runs. A profile took 2-6 s, against 15-20 s of
  attribution per chapter, so the profile stage adds well under a minute. Every existing voice pick carried over, and
  Chatterbox came back loaded (Docker shows it unhealthy for the minute it is unloaded).
- First prompt: one voice note was the prompt's own example word for word ("brisk older woman, dry
  and impatient"); one invented an accent; both 3-7 line characters came back "supporting"; one
  relationships field listed everyone else as "not mentioned in the excerpts". Now: the example is
  gone, the accent guard and minor rule are in, and "not mentioned" filler is dropped. On a rerun of
  the same casts none of the four recurred.
- A profile surfaced a real voice mismatch. "the doctor" had been given a male voice because the
  attribution left their gender unknown. The profile found she was a woman and set female.
  Voices already picked are never changed, so the table now shows the mismatch for the owner to fix.
- Spot-checked against the text: a family detail one profile gave is in the book; "southern drawl" was not.
- Remaining weaknesses: roles vary between books of the same series (one character is "protagonist"
  in one and "supporting" in the next); the notes lean on the book's register ("breathy" three times
  in one cast).

### 15.3 Tests

523 app tests pass (501 before; 22 new). `cast_profiles_test.py` (18) covers: name forms, passage
finding and rendering, the spread order and the budget, first lines, candidates, reply parsing
(role words, bad values, clipping, filler, unusable replies), the accent guard, the minor rule,
retry-then-skip and the error stop. `cast_analysis_test.py` (1) runs the whole job with profiles and
with a profile-stage LLM failure. `chatterbox_ui_test.py` (3) covers the new columns and summary, the
click-to-profile text and the progress line. Not verified by clicking through a browser: the deployed
page's config carries the new columns, and the same UI functions were run over the live casts
inside the container.


## 16. Voices matched to character profiles (2026-09-28)

Until now a cast's voices were the first free voice of the right gender, in alphabetical order
(one book's two leading women got Abigail and Alice). Now the profile says what kind of voice fits and each
Chatterbox voice is measured, so the suggestion is the closest measured voice.

### 16.1 What was tried first

Handing qwen2.5:14b the whole voice list (33 voices with measured descriptions) and the profiles,
and asking it to cast. Every reply was valid (genders respected, no voice shared, narrator avoided,
2-4 s), but the picks followed the list order. Reversing the list kept the same voice for 1 of 11
characters and the same pitch band for 5 of 11. The same order twice gave identical picks, so this
was position, not randomness. It also made up reasons ("clear" for a voice with no such tag).
Asking it instead for targets per character (no voice list) and choosing in code kept 7 of 11 voices
across orders, which is the design built.

### 16.2 Measuring voices (`core/voice_measure.py`, new)

Each voice speaks `MEASURE_TEXT` through Chatterbox (`/tts`, the saved delivery settings, as a book
would), and Praat (`praat-parselmouth==0.4.7`, a new pinned dependency) gives three numbers: median
pitch, pitch spread (10th to 90th percentile, in semitones) and harmonics-to-noise ratio. For
matching, each becomes a percentile among the measured voices of the same gender, and the Voice lab
shows them as words ("low for a woman, husky, even").

Generated speech is measured rather than the reference clip. On the 33 voices, clip and generated
pitch agree (r 0.96; same third of the range for 27 of 33, never the opposite third) but huskiness
doesn't (r 0.86, same third for only 19 of 33), and the generated speech is what a listener hears.

Measurements live in `voice_features.json` (app data) with each file's size and modification time,
so a replaced voice is measured again. **Measure voices** (Voice lab) measures every voice that
isn't; adding a voice measures it straight away, and a failure there (Chatterbox busy, unloaded or
unreachable) only defers it. Deleting a voice drops its measurement. Kokoro voices aren't measured
(Kokoro isn't deployed here) and are still matched by gender only.

### 16.3 Targets and matching

The profile reply gains `pitch` (low, medium, high), `quality` (husky, clear) and `delivery`
(expressive, even), relative to other voices of the same gender; "either", unknown or unrecognised
words become no target. Without a pitch target, a child wants high and an elderly character low.
`suggest_voices` keeps its rules (most-spoken first, distinct voices, narrator never, gender first)
and, among the voices those allow, takes the lowest `match_cost`. Pitch comes first: any voice in
the wanted third of the range beats any voice outside it. Then huskiness and liveliness count, then
closeness to the band's middle, so the most extreme voice isn't everyone's pick. An unmeasured
voice costs as much as one just outside the band. With no targets or no measurements, the list
order decides as before.

UI: an **Auto-pick suggested voices** checkbox (on) next to Analyse. With it, a re-analysis carries
over only the voices saved with **Save voice** (now marked `voice_picked`); the others get fresh
suggestions. **Suggest voices again** (browser confirm) does the same for an existing cast without
the LLM, for example after measuring voices. Clicking a character adds a line such as "**Voice
match:** wants medium pitch, clear, even · Jade is medium for a woman, even".

### 16.4 Measured on the owner's machine

- Measure voices: 32 voices in 71 s. The pitches match the investigation's measurements (Olivia
  172 Hz, Teen 230, Thomas 115). `Elf.wav` had been removed from the voices folder at 18:17, between
  the investigation and this run.
- Re-analysis with auto-pick on scratch copies of the two real casts: all 10 profiles usable, every
  one with targets. The first matching rule weighted pitch only twice as heavily, and it put 3 of 11
  characters a band higher than asked, for a voice that was clear and even as asked. Hence the
  pitch-first rule. After it, all 11 characters got a voice in the band asked for, and 2 matched on
  all three words (a young lead: Teen, high, clear, expressive; a crude minor character: Michael, low, husky, expressive).
- `Taylor.wav` is recorded as male but speaks at 196 Hz, the female range; it is the "highest man"
  and would go to young men. Worth a listen.

Limits: three coarse measurements can't hear warmth, age or acting, so these are better first picks,
not casting. The weights are set by reasoning, not tuned by ear. Casts saved before this have no
`voice_picked` marks, so auto-pick or Suggest voices again replaces every voice in them, including
earlier hand picks.

### 16.5 Tests

551 app tests pass (523 before; 28 new). `voice_measure_test.py` (8) runs Praat on synthetic
voices of known pitch, glide and noise, and covers saving, staleness, which voices need measuring,
within-gender percentiles and the words. `cast_test.py` (9) covers profile matching, the pitch-band
rule, distinctness, the age fallback, clearing unpicked voices and carrying only picked ones.
`cast_profiles_test.py` (1) covers target parsing. `cast_analysis_test.py` (1) covers auto-pick in a
real re-analysis. `chatterbox_ui_test.py` (9) covers measuring on add, Measure voices (including an
unreachable server and one bad voice), the Voice lab text, forgetting on delete, matched suggestions,
Suggest voices again, the Voice match line and the auto-pick setting.

## 17. Excited lines no longer clip; delivery per character and a narrator from the book's tone (2026-09-28)

### 17.1 Bug: a single excited word came out distorted

Reported by the owner. Chatterbox returns clips already peaking near -0.4 dBFS (every clip in a
36-clip probe did, whatever the preset). Adaptive delivery applied the excited preset's +1.5 dB
first, which pushed peaks past full scale, where pydub's `apply_gain` clips the waveform flat. The
peak guard ran after that, and turning a clip down can't undo clipping. A sentence only clips its
brief peaks; a one-word shout is loud from end to end, so it audibly distorted. The unit test missed
it because it used a square wave, which clipping leaves unchanged.

Fix: `delivery.guarded_gain` applies a mood's gain only as far as the headroom allows, so the peak
never passes -1 dBFS. The provider and the Voice lab's soft/normal/excited preview both use it.
Measured on the same 18 excited Chatterbox clips: 10 had 9-85 samples flattened at the top the old
way, none the new way; clips with headroom still get their +1.5 dB. The new test uses a sine wave and
checks the output is the input scaled. Since nearly every clip already peaks near full scale,
excited lines are now rarely louder than normal ones; the higher exaggeration carries the excitement.

### 17.2 The book's tone and the narrator

After the profiles, the cast job asks the LLM once about the book's narration
(`cast_profiles.describe_book`), from up to 6,000 characters of dialogue-free paragraphs. It returns:
- point of view, and the narrating character in a first-person book (matched to a cast key);
- tone in a few words, pace (slow, measured, brisk) and intensity (restrained, moderate, dramatic);
- the narrator voice that suits the book: gender, pitch, quality, delivery.

It is saved as `cast["book_tone"]`; a failure leaves none and never fails the analysis.
`cast.suggest_narrator` picks the measured voice that fits, taking a first-person narrator's gender
from the viewpoint character and never offering a voice the owner picked for a character.
`cast.narrator_delivery` nudges the owner's saved sliders: ±0.1 exaggeration for restrained or
dramatic narration, ±0.05 CFG for slow or brisk prose. Both are unmeasured by ear, kept small.

### 17.3 Delivery per character

With adaptive delivery in cast mode, each character's lines use the book's baseline shifted by up to
±0.12 exaggeration (`cast.exaggeration_offsets`), before the line's mood preset applies. A delivery
from the profile is centred on the cast's line-weighted average. The first live run showed why: the
model called 5 of 6 characters of a dramatic book "expressive", and uncentred that made nearly all
its dialogue louder instead of making characters differ. On that cast, centred, only the calm doctor
reads flatter (-0.12); a character with no profile reads as the book. A delivery the owner sets in
the editor applies as set. Units carry a voice, not a character, so the provider looks the offset up
by voice; the narrator's voice never has one.

### 17.4 Seamless by default

The owner's goal: automatic, with manual changes as advanced settings. In cast mode with
**Auto-pick suggested voices** on (the default), the first time a finished cast is shown on a page,
the Make tab's Voice and the Voice lab sliders are set to the suggested narrator and its delivery.
Add to queue already takes both from there, and a change the owner makes afterwards wins. Characters'
voices are suggested around that narrator. The cast editor (gender, voice, the new Delivery setting,
Sample, Save), **Suggest voices again** (which now also re-picks the narrator) and the auto-pick
switch moved into a collapsed **Adjust the cast (advanced)**. The table and the clicked character's
profile stay visible. Sample now speaks the character's own first line, in the editor's voice and
delivery.

### 17.5 Measured

Full analysis of a scratch copy of the second real cast on qwen2.5:14b. The tone came back as third
person, a three-word mood, brisk, dramatic, asking for a male narrator (medium,
husky, expressive). The narrator suggestion was Adrian at exaggeration 0.75, CFG 0.60, temperature
0.50, up from the saved 0.65 and 0.55. That replaces the owner's usual female narrator (Elena), a
real change the owner should hear. Characters were re-fitted around it; all 6 profiles were usable.

### 17.6 Tests

566 app tests pass (551 before; 15 new). They cover:
- `delivery_test.py` (2): the gain fix.
- `delivery_provider_test.py` (2): character offsets reach the request, and none without adaptive
  delivery.
- `cast_test.py` (5): owner-then-profile delivery, centring, lookup by voice, narrator sliders and
  narrator choice.
- `cast_profiles_test.py` (4): tone parsing, narration passages, describing the book with a retry,
  and a failure.
- `chatterbox_ui_test.py` (2): auto-applying the narrator once, and Sample with delivery.

Not tested in a browser: the collapsed section and the automatic Voice and slider updates. The live
page carries the new controls, and the same handlers ran over the live cast.

## 18. First-person books: the narrator reads the "I" character's lines (2026-09-28)

The usual audiobook convention is one performer for a first-person narrator, so their dialogue
should be in the narrator's voice rather than a voice of its own. About 28% of a 150-book sample of
the owner's library reads as first person: narration outside quotes using "I" more than 1.5 times as
often as "he" and "she".

`cast.narrating_character` is the book tone's viewpoint character (§17.2), unless the owner gave
that character a voice of their own in the advanced editor. That character:
- speaks their lines in whatever narrator voice the book is queued with (the provider's voice rule);
- gets no suggested voice, and one they held from an earlier suggestion is freed for others;
- has no delivery offset and doesn't count in the cast's delivery average (§17.3).

The narrator pick uses their gender and their own profile's voice targets before the tone's.
The cast table shows "(narrator's voice)". The editor offers "(the narrator's voice)" as the first
choice, and saving it switches back from an own voice. Sample plays the Make tab's narrator.

Live check on the owner's machine: a full analysis of the first three story chapters of a
first-person novel from the library, into a scratch cast.
- The tone came back as first person, "introspective melancholic", measured pace, moderate
  intensity, asking for a female narrator (medium, clear, even).
- The page set the narrator to Jade and kept the saved sliders (nothing to nudge). The narrating
  character (45 of 103 lines) showed "(narrator's voice)", and all 9 profiles were usable.
- The book never names its narrator, so attribution called them "I". The first summary said "I
  tells the story"; an unnamed narrator is now described as such.
- The same run showed two older rough edges, now fixed. The attribution model had listed "Unknown"
  as a character (0 lines, yet given a voice), and `parse_reply` now drops such names. Relationships
  like "Bea: unknown" are dropped as filler.

Tests: 573 pass (566 before; 7 new). They cover the narrating character and the owner override,
no suggestion and a freed voice, the narrator following their profile, no delivery offset, the
provider's voice rule, the table/editor/Sample flow, the unnamed narrator, and "Unknown" as a
character. Not measured by ear: whether one voice for narration and the narrator's dialogue sounds
right across a whole first-person book.

## 19. Chatterbox artifact hunt: short dialogue and quoted narration (2026-09-29)

Two completed audiobooks exposed defects that the existing three-second silence check missed.
In one, a short utterance ending in an ellipsis produced 0.78 seconds of damaged audio. A quoted
name embedded in prose was also split out as dialogue and stretched to 3.2 seconds. Speech-unit
construction now keeps a short, unpunctuated quoted term with its surrounding narration when no
speech tag is present. Cast line IDs stay unchanged. A short trailing-ellipsis clip under 0.95
seconds is retried with a new seed.

In a later book, alignment of the chapter's 243 speech units with the final M4B found three more
defects in one passage: a 19-character quote lasted about 7.1 seconds, a 60-character quote lasted
about 1.2 seconds, and a 20-character interrupted quote ran for over 20 seconds. The last had a
mismatched end quote in the source. The provider now repairs that punctuation before sending it to
Chatterbox and retries a quoted clip whose duration is implausible for its text length. Repeated
bad results fail the chapter instead of entering the final audiobook.

The first manual repair removed the large distortions, but review found two small residual
artifacts: one in a preserved narrator clip near the 10-second mark of the preview, and one inside
a newly generated interrupted line near 15–16 seconds. Waveform checks found no click at the
joins. Both takes were replaced, with short fades at the joins, in a second separate repair.
Duration and silence checks cannot reliably detect every brief phonetic artifact; a listening
preview remains useful for these cases. The original audiobooks were left unchanged.

A subsequent listening check found the second repair's short narrator take near 10 seconds
sounded worse, while its interrupted line near 15 seconds improved. The absence of a visible
splice spike did not predict listening quality. Three contextual comparisons varied only the
short narrator take. A calmer take was accepted as usable, though still imperfect, and selected
for the final separate repair; the improved later line was kept.

The audiobook test suite passed all 580 tests after the provider changes. The running app loaded
the updated code, and the repaired M4Bs retained their chapter markers, cover art and metadata.

## 20. Short-line delivery and clip diagnostics (2026-09-29)

The remaining brief phonetic defects sometimes occupy normally timed clips, so a waveform-only
rule cannot reliably choose the better of two plausible takes. A previous controlled comparison
had two plausible durations for the same short line; listening selected one that a duration check
could not distinguish reliably. The existing sentence packer already joins short sentences within
one uninterrupted voice span. Narration, dialogue, and attributed speaker boundaries stay separate.

Adaptive delivery now uses the book baseline for lines of at most 12 characters, including mood
settings, gain, and character delivery offsets. From 13 through 40 characters, those changes ease
in to the full preset. Longer lines retain their existing expression. Short Chatterbox requests
(up to 25 characters) now start with an independent, recorded seed. The existing bad-take retries
continue to use a different seed, and an implausibly long short narration tag now triggers one.
These changes do not assert that every normally timed phonetic defect can be detected.

Each completed Chatterbox chapter now gets a local clip map with a text hash and length, voice,
mood, settings, selected seed, attempts, and chapter-relative times. The map does not copy the
book's text into the audiobook library. The finished M4B gets a companion
`.clips.json` map with book-relative times adjusted to the measured chapter packet durations.
This supports targeted investigation of a reported timestamp while preserving skipped chapters'
maps. A failed map write does not fail audiobook generation. Map files receive the same output
ownership adjustment as the M4B so library deletion remains possible.

The full app suite passes 595 tests against the edited source in an isolated container.

A live, synthetic five-character line exposed a missed case: with the same voice and delivery
settings, one seed produced 37.42 seconds, while another produced 1.06 seconds. A matched-seed
expressive variant was also overlong. The short-dialogue and narration duration guards now include
one-word lines (previously they started at eight characters), so these takes are retried before
entering a book. This demonstrates a seed-dependent gross defect; it does not establish that
changing expression causes or cures the brief, normally timed defects.

Live provider verification repeated that synthetic first take through the edited app code: the
37.42-second output was rejected, a new seed produced 1.06 seconds, and the clip map recorded
the second seed, two attempts, and the baseline settings. The app container was rebuilt and
restarted with an empty queue; the page returned HTTP 200 and the running image contained the
one-word guard. The final full suite passes 595 tests. Book clip offsets also follow the
M4B's cumulative chapter rounding, avoiding drift over many chapters.

## 21. Softer excited delivery; deleted voices forget their gender (2026-09-29)

The owner heard slight distortion in Elena's loudest excited Voice lab take and asked for a lower
excited volume boost. A measurement showed that the boost was not the cause. At the saved sliders
(0.65/0.4/0.6), three normal and three excited takes of the preview phrase were requested as WAV.
Every take peaked at exactly -0.45 dBFS because Chatterbox's server scales any clip over 0.99 to
0.95 with a linear gain, not a limiter. No take had samples at full scale, whether raw or after
`guarded_gain` and an MP3 round trip (excited peaked at -0.85 dBFS). Because of this server
normalization, the peak guard cuts the excited +1.5 dB to about -0.5 dB. The boost therefore
rarely applies, and it cannot clip. The harshness comes from the model's own rendering at 0.92
exaggeration and CFG 0.3.

Excited now steps exaggeration by +0.18 (was +0.27) and CFG by -0.05 (was -0.10). Temperature
(+0.09) and gain (+1.5 dB) are unchanged. At the approved baseline, excited is now 0.91/0.45/0.7.
It remains above emphatic (0.85/0.46/0.64). The owner still needs to judge the result by ear.

Deleting a voice in the Voice lab previously forgot its measurement but kept its entry in
`voice_genders.json`, so entries for deleted voices accumulated. Deleting a voice now removes its
gender entry too. Files deleted outside the app (for example, in Explorer) are still not noticed.

## 22. Batch queuing, and first-person narrators per story (2026-09-29)

### 22.1 Pick books, analyse, press Start once

The owner's workflow is to pick a book and analyse it, then pick the next one while that analysis
still runs, and so on, and press Start only after choosing every book. Decisions:
- **Cast is the default voice mode** when an LLM is configured (without one it isn't offered).
- **With Auto-pick on, the book queues itself when its cast is ready**; nothing starts until
  **Start queued books**. At Analyse, the Make tab's book options go with the analysis job
  (`then_queue`). When the job finishes, `JobQueue.on_done` calls `queue_book_after_cast`: it sets
  the narrator and delivery from the book's tone, exactly as the page does, and queues the book.
  This runs in the queue rather than the page on purpose. Otherwise a book would only be queued if
  the page happened to show it when its analysis finished, so analysing book B while A runs would
  lose A. Anything that fails the usual checks is noted on the analysis row ("book not queued:
  …"). The output folder is checked at Analyse time, so a clash shows at once.
- **Add to queue is hidden** while Cast mode and Auto-pick are both on.
- **Analyses always run ahead of waiting books** (`tick`), so pressing Start early never lets a
  book generate before a later analysis. **Start** shows as soon as an analysis will bring a book,
  not only once a book is waiting. Analysing after Start joins the running batch; before, it put
  the queue back into preparing and held the remaining books.

Known gap: a book analysed earlier, with Auto-pick on, has no Add button. The owner unticks
Auto-pick to add it, or re-analyses.

### 22.2 Bug: the book tone named the wrong "I"

The first auto-picked book (a first-person novel) was narrated by Zoe. The book tone had named
Bettie as the "I", so her 263 lines were read in the narrator voice, and the narrator voice was
matched to her gender. The narrator is Oliver. The tone request (§17.2) sees only paragraphs
without dialogue. There Bettie is named on every page and Oliver never is: he is only named when
spoken to ("Hey, Oliver"). The attribution had it right: 73 of the 74 lines tagged "I said" / "I
told her" were Oliver's, and one was Jake's.

### 22.3 Point of view and narrator per chapter

- **Point of view by pronoun rate, with no LLM** (`chapter_point_of_view`): a chapter is first
  person when its narration (text outside quotes) has at least 20 I/me/my per 1,000 words. Measured
  rates were 80–120 in first-person chapters, 36 in the lowest seen, and 0 in every third-person
  chapter checked. Comparing with he/she didn't help: "she" was as frequent as "I" in the
  first-person chapters. A chapter with fewer than 150 narration words stays undecided and follows
  its neighbours.
- **The "I" comes from attribution** (`chapter_narrators`): a chapter's narrator is whoever its
  "I said" lines (`speech_tags.first_person_tagged`) were attributed to, if at least 2 of them and
  more than half agree. Each chapter votes on its own. At first, consecutive first-person chapters
  were pooled as one story, but in Greene Shorts Volume 2 two first-person stories sit side by
  side, and the pooled vote gave Irene's name to Nate's story (18 of 18 of its lines were Nate's). A
  chapter with too few votes follows the chapter before it (else after), then the run's pooled
  vote. Only a run with no "I said" lines at all falls back: to the book tone when it is the book's
  only first-person run, else to one tone request about that run alone.
- **The book's own point of view follows its chapters** (`apply_chapter_narrators`): first person
  when most of the narration words are, told by whoever narrates most of them. This overrides the
  tone's guess.
- **The generator reads each chapter's narrator** (`cast.chapter_narrator`). A third-person chapter
  has none. A cast analysed before this change falls back to the book-level narrator.
- **A first-person story told by someone other than the book's own "I" is narrated in that
  character's voice** (`cast.chapter_narrator_voice`), narration and lines alike. Otherwise one
  narrator voice, chosen from a mostly third-person book's tone, read a man's first-person story in
  a female voice (Volume 2: Gianna reading Nate). The teller's suggested voice already fits their
  gender and profile and is distinct from the others, so no new pick is needed. The owner changes it
  in the editor as for any character. The cast panel lists each story's teller and voice.
  First-person novels are unchanged: their "I" is the book narrator.

A pronoun scan of the library's 34 collection-like titles (out of 1,518 books) found 19 that mix
points of view. Several were true collections: Greene Shorts, Aberrations, Here Be Monsters. Others
were long series with a single first-person chapter, most likely an author's note. Such a chapter
costs at most one extra tone request and has no dialogue for a narrator to speak.

### 22.4 Aliases: names only, family words per chapter

The same runs showed characters collecting aliases that aren't names: Chris had "he", "honey" and
"child", and others had "Himself" and "Herself". In Greene Shorts Volume 1, every mother in six
stories merged into one "Terri" through "Mom", "Ma", "Mommy" and "Mama": 312 lines, "intimate with
Rob, Dennis, Henry, Jake, Scott", with her first line from another story.
- **Never aliases** (`usable_alias`): pronouns (reflexive ones too), pet names, generic words ("the
  woman", "child") and descriptions ("his mom"). A pronoun answered as a speaker counts as an
  unknown speaker.
- **Family words ("Mom", "Grandpa", "Step Mom", "little brother", in-laws) used as aliases name one
  person only within a chapter** (`Roster.chapter_aliases`, reset by `new_chapter`). A character
  whose only name is a family word is a different case; see §22.6. They are never saved with
  the character. Within one chapter they still merge ("said Mom"); across chapters only real names
  do. In a novel the cost is that a stray "said Mom" in a later chapter may become its own row. That
  follows the Roster's rule that splitting a character is cheaper than a wrong merge.
- **"I" stays an alias.** In a first-person book the model uses it for the narrator (Oliver had
  it). In Volume 2 it did not carry across stories: Nate had "I", yet Irene's story's lines went to
  Irene.

Real story boundaries, which would give each story its own character list, are not detected: the
EPUB parser follows the reading order and keeps no grouping.

### 22.5 Live checks

- **The first-person novel, re-analysed from scratch:** Oliver in all 8 chapters, narrator Gabriel
  (male), Bettie in her own voice (Zoe). The book queued itself and generated.
- **Home Temptation 4 (three third-person stories):** every story chapter came out third person, in
  line with the tone; no narrator.
- **Greene Shorts Volume 1:** chapter 5 narrated by Terri and chapter 7 by Scott, the rest third
  person. This was the run that exposed the "Mom" merge. It was deleted and not generated.
- **Greene Shorts Volume 2:** chapters 3 (Nate) and 4 (Irene) are first person, 5–8 third. The
  per-chapter vote was applied to the saved cast and to the queued job's snapshot without an LLM
  (backups in `data/cast_backups/`). Narration: Thomas (Nate), Elena (Irene), Gianna for the rest.

Not yet judged by ear: the per-story narrator voices. Elena is also that book's dialogue voice, so
the few unknown-speaker lines in Irene's story will sound like her.

### 22.6 Review fixes

An outside review of the last six commits found three problems in the decision logic:
- **An undecided chapter hid the book's narrator.** A chapter with too little narration to judge,
  and no judged neighbour to follow, was saved as "no narrator". That blocked the fallback to the
  book's known "I". Such a chapter now saves no chapter-level narrator, so `cast.chapter_narrator`
  falls back to the book's.
- **An untagged story inherited its neighbour's teller.** When a tagged first-person story was
  followed by an untagged one, the second took the first's narrator and the LLM was never asked. A
  chapter now borrows a neighbour's narrator (nearest first, then the run's pooled vote) only if
  that character speaks in it. In a first-person novel the "I" nearly always has lines in every
  chapter; in a collection's next story the previous teller doesn't appear. Otherwise the order
  is: the book tone's narrator (single-run books only, if they speak there), then a tone request
  about that chapter alone. That answer is lent to the following chapters like a vote, so a novel
  with no "I said" lines asks once, not once per chapter.
- **A character whose only name is a family word still merges across chapters.** §22.4 overstated
  this: chapter scoping covers family words used as aliases, not a character the model knows only
  as "Mom", whose name is global. This was left as it is on purpose. Scoping names per chapter would
  split every unnamed "Mom" of a novel into one row per chapter, each with its own voice. In a
  collection, unnamed mothers of different stories share a row and a voice, like any two stories'
  characters with the same name. A named mother's "Mom" alias stays within its chapter, which is the
  case seen live (§22.4).

A dry run of the new logic on the three real casts (the first-person novel, Greene Shorts Volume 2,
Home Temptation 4), with no LLM, gave the same narrators as saved.

### 22.7 Tests

623 pass (595 before; 28 new). They cover:
- the batch flow: queuing after the cast, analyses first, Start visibility, and the note on the
  analysis row;
- pronoun point of view, per-chapter votes (including side-by-side stories), the per-run LLM
  fallback and the book-level override;
- the chapter narrator and teller voice in the provider;
- alias filtering and chapter-scoped family words.

## 23. Bookmark investigation: brief artifacts between voices (2026-09-30)

Read the host's `AGENTS.md` and this log, then queried BookOrbit's database read-only. One finished
collection had 27 active bookmarks over its first 20m15s: 25 apparent artifact markers and two
explicit casting notes. The owner estimates clicking about half a second after the defect, so each
marker was investigated from 1.5 seconds before to 0.5 seconds after. The source's 677 first-chapter
units were reconstructed using the deployed provider and checked against every map text hash and
voice. Their logs identify the actual run starting at 01:18 UTC, rather than an earlier failed run.

Findings:
- 24 of 25 artifact windows contain a unit of at most 25 characters; 19 of 25 likely units, selected
  at bookmark minus 0.5 seconds, are that short. The heard portion contains 106 such units out of
  328 (32%). These are correlations and timestamp candidates, not exact word-level localization.
- 11 windows include short narrator fragments, eight between quotations. At 7m59s the sequence is
  Gianna -> a nine-character Thomas narration tag -> Jeremiah. Other marked tags sit between two
  Gianna quotations (10m52s, 15m38s, 20m15s). Different dialogue speakers are therefore not necessary;
  isolated tiny requests are the shared risk. Do not combine a tag into another performer's line.
- All 49 nearby candidate units were accepted on attempt one, with no quality flag. Their ordinary
  durations evade the existing silence/length checks. An unrelated nearby long quote did retry,
  confirming those checks were active. Most candidates use baseline 0.78 exaggeration, 0.45 CFG,
  0.6 temperature: the short-line easing removes mood/character offsets but retains the high baseline.
- Measured candidate boundary sample jumps are small (maximum 0.00263 full scale), so the data does
  not support a splice click as the common cause. AAC decoding has two isolated samples above full
  scale, which does not establish clipping in generation. A few late sound islands after silence
  warrant listening; waveform rules cannot label them reliably as unwanted speech.
- Two reference-mel/token-length warnings during the passage are away from the bookmarked units;
  the installed model already truncates reference tokens for that warning. No generation error or
  compiled-path fallback was found in the investigated interval.

Created five-second original-audio windows and three same-seed A/B pairs for harmless narrator tags
at 10m52s, 15m38s and 20m15s. All three baseline replays exactly reproduced the recorded raw
durations (1.20, 0.68 and 1.32 s). Only exaggeration changes in the second take, from 0.78 to 0.5;
CFG, temperature, voice and seed are held. At this stage the comparisons had not been judged by
ear and no production preset was changed; the listening result and subsequent fix are below.

Recommended next steps: audition a lower expression cap specifically for isolated narration tags;
reuse the existing sentence packer across adjacent spans only when paragraph, voice, mood and
speaker match; investigate a conservative suspicious-tail retry using the saved windows, rather
than blindly trimming tails or retrying every clip. Neither packing nor a tail rule covers all
normally timed phonetic defects. Production code was unchanged at this stage; the original book
remains unchanged. Preserve the existing unrelated UI edit.

Private report, CSV, timestamp windows, source reconstruction and A/B WAVs are under the host's
`data/diagnostics/volume2_bookmarks_2026-09-30/`. They contain personal library material and stay
outside the repository. No software tests were needed for the documentation change; the diagnostic
scripts include source/map/log consistency assertions and completed successfully.

### 23.1 Listening result and deployed short-tag cap

The owner clarified the combined comparison: the first two pairs were garble followed by a clear
"she said"; the last pair was clear both times. Exact PCM verification confirmed the file's order
is baseline/calm, baseline/calm, baseline/calm, with only digital silence inserted. This supports
the lower expression setting on two marked tags, not a universal cure for short-clip artifacts.

`OpenAITTSProvider._delivery_extra_body` now caps exaggeration at 0.5 for recognized narration tags
of at most 25 characters when adaptive delivery is on. It reuses `has_speech_tag`; quoted dialogue,
longer narration, lower existing exaggeration, CFG, temperature, adaptive-off jobs and other engines
keep their existing behavior. The shared method covers both sentence and paragraph generation.
No additional retry or packing change was added.

125 relevant tests pass. One new provider test exercises both packing modes and checks short tags,
quoted tags, ordinary prose and an already lower baseline. Existing assertions were updated to
select dialogue explicitly and to expect the cap on a short named narration tag. A live provider
run using all three recorded seeds produced PCM byte-for-byte identical to the calmer comparison
takes; clip maps recorded 0.5/0.45/0.6, the correct seeds and one attempt each.

With the queue empty and paused, rebuilt and recreated only the app container. The page returned
HTTP 200 and a check inside the running container confirmed the short tag receives 0.5 while longer
prose retains 0.78. This applies to future generation; the existing audiobook was not regenerated.
The original unrelated Voice lab slider edit remains intact. Private evidence includes
`comparison_order.json`, `listening_feedback.json` and `provider_fix_validation.json`.

## 24. First short-book regeneration after the tag cap (2026-09-30)

Reviewed the owner's next run to completion (app log 09:05–09:25 Eastern; Chatterbox logs are UTC,
four hours ahead). Four chapters, 726 clips, final duration 3596.828 seconds. Reconstructed all
source units and verified every finished clip hash and voice, four chapter markers and agreement
between the M4B duration and its map. The chapter working folder is removed after completion;
the final companion map is the durable record. BookOrbit indexed the M4B.

12 units needed retries: 10 succeeded on attempt two, two on attempt three. The 14 discarded takes
comprised 11 near-silent takes, one too-short ellipsis, one overlong short narration and one
truncated dialogue. No retained unit has a quality flag, and no repeated near-silence failed the
book. One backend 1000-token-cap warning belongs to a rejected near-silent take. There were no
reported server errors or compiled-path fallbacks. The 38 reference mel/token-length warnings
are the existing model's reference adjustment; the local model's price warning is also expected.

All 76 recognized short narration tags have exaggeration at most 0.5, confirming the new cap in
real generation. Eleven clips of at most 25 characters still last over three seconds: nine are
recognized narration tags in the custom narrator voice, two are short dialogue clips in Gabriel.
They are listening candidates, not confirmed phonetic defects; the duration guard deliberately
allows at least six seconds before labelling a short take overlong. Leave that guard and the voice
reference unchanged until the owner's listening/bookmarks identify whether these takes are bad.

Private evidence and timestamp shortlist: `data/diagnostics/short_book_2026-09-30/`, including
`FINDINGS.md`, `run_review.json`, `book_metadata.json`, the job snapshot and source reconstruction.
No production code or settings changed in this review.

## 25. Original first-chapter regeneration and remaining tiny-quote artifacts (2026-09-30)

The owner selected the first chapter of the original collection as a more representative check.
Reviewed the 09:32–09:46 Eastern run to completion: 677 clips, 2701.674 seconds, one M4B chapter.
All source text hashes match the original chapter. The original 0.78/0.45/0.6 baseline is retained,
and all 59 recognized short narration tags receive the new exaggeration cap of 0.5. Two rejected
first takes (truncated dialogue and too-short ellipsis) succeeded on attempt two; no retained clip
is flagged. No server error, token cap or compiled-path fallback was reported. Twenty reference
mel/token-length warnings are the existing automatic reference adjustment.

The newly analysed cast changes 508 voices, including the narrator from Thomas to Michael, and
short clips have new randomized seeds. Identical text makes this useful for listening, but it is
not a controlled comparison of the production change alone. The original audiobook/database
bookmarks were removed by the owner's regeneration workflow. Their private bookmark snapshots,
candidate maps, original audio windows and source reconstruction remain available; recovered the
complete original first-chapter timing/settings records from logs and remapped all 25 original
artifact passages to new times using clip numbers and text hashes.

The owner then listened while watching the text and reported mostly single-word quotations. Read
20 new BookOrbit bookmarks: 19 artifact markers, one explicit wrong-speaker note. Using the same
1.5-seconds-before/0.5-seconds-after windows, every artifact window contains one tiny quote of
1–3 words and 5–17 characters: twelve single-word, five two-word, two three-word. There are only
24 single-word quote clips among the 363 clips in the heard portion. All 19 tiny quotes were kept
on attempt one, use 0.78/0.45/0.6, and last 0.68–2.28 seconds, escaping the silence/length checks.
Eighteen use Gianna, one Michael. No representative marker is a recognized short narration tag;
three windows contain a neighboring tag. Eight quote chunks were also original artifact candidates.

The first cap deliberately targeted narrator tags, leaving these quotes at the higher baseline.
Prepared three same-seed pairs for harmless bookmarked single-word quotes ("Oh!", "Nothing!",
"Oh,"): same input, Gianna voice, recorded seed, CFG and temperature; exaggeration alone changes
from 0.78 to 0.5. Baseline replay durations match the recorded raw durations within 1 ms. Verified
the six PCM segments and their order exactly. Asked the owner to judge clarity before extending
the cap; do not infer improvement from duration or combine a quote with another speaker's prose.
The wrong-speaker note maps to a saved cast attribution assigning the wife's voice to a line the
owner identifies as the husband's; record this separately from synthesis artifacts.

Private evidence, timestamp tables, final map snapshot and listening pairs are under
`data/diagnostics/volume2_first_chapter_2026-09-30/`. Read-only diagnostic assertions passed. No
additional production code, settings, voice assignments or audiobook changed during this review.

### 25.1 Short-quote exaggeration trial rejected; quotation delimiters tested next

The owner reports that all six takes in the short-quote comparison were garbled, with nothing
clear. Lowering exaggeration from 0.78 to 0.5 did not solve these three quoted words; do not extend
the narration-tag cap to dialogue on this evidence. The earlier tag cap retains its separate
positive listening evidence.

Traced the deployed API's chunker, synthesis wrapper and installed `chatterbox.tts.punc_norm`.
The local API/engine/fast-loop/utils sources match the running container byte-for-byte. Tiny quotes
reach the model intact; punctuation normalization converts curly quotes to straight quotes, then
adds a full stop because the closing quote is not considered an ending punctuation character.
For example, the quoted exclamation becomes `"Oh!".`; without delimiters it remains `Oh!`.
This is a concrete difference, not proof that the added punctuation causes garble.

Prepared a second controlled comparison of the same three marked words. Reused the exact original
baseline WAVs and made only three new requests with the outer quote delimiters removed. Word,
punctuation inside the quote, voice, recorded seed, exaggeration 0.78, CFG 0.45 and temperature 0.6
stay fixed. `quote_delimiter_comparisons.wav` plays original/quotes-removed in each pair. Audition
before choosing a production fix; duration changes alone do not establish clarity. Feedback and
request metadata remain in the private diagnostic folder. No production change was made.

### 25.2 Separate listening files after ambiguous paired playback

The owner found the paired takes too close to decipher: possibly clearer "Oh" and "Nothing"
after quotation delimiters were removed, but each seemed merged with its original garbled take.
Both third takes remained garbled and possibly sounded like "money". Verified the third input is
exactly `Oh,`; that perceived extra word was not in the request. The first two are tentative
positive listening evidence, and the third remains unresolved. Do not call the trial a general fix.

Repackaged the existing samples into separate original-only and quotes-removed-only WAVs, with
four seconds of exact digital silence between words and two seconds before/after the sequence.
Also saved individually padded modified takes. No new synthesis or production change. A small
standard-library script verifies exact PCM preservation, exported file contents and silent gaps;
all assertions pass. Feedback and the spacing manifest remain in the private diagnostic folder.

### 25.3 Quotation removal remains unreliable; decoder/precision controls

On separated playback, the owner reports that the original three takes are all garbled (the last
still suggests an unwanted "money" sound). Removing delimiters gives a slight leading artifact
then clear "Oh" in the first take, a garbled onset and unclear remainder in "Nothing", and an
unrecognizable final word. Do not treat punctuation removal as a complete fix or deploy it alone.

Reviewed the compiled T3 sampling flow against the installed stock decoder. Both follow the same
CFG/repetition/temperature/min-p/top-p/EOS sequence; no obvious short-input control-flow omission
was found. Ran a separate process inside the Chatterbox container, leaving the live service's
model and configuration intact. Loaded Original with compilation and BF16 off, generated the
three original quoted inputs with their recorded seeds and 0.78/0.45/0.6, plus an invented longer
control sentence in Gianna. Then converted that isolated T3 to BF16 and repeated the same four
cases on the stock decoder. FP32 runs precede conversion to preserve the full original weights.

All eight takes completed with finite audio and no synthesis error. Ran the matching longer
control through the existing compiled API as well. Saved original/compiled, stock BF16 and stock
FP32 listening files, each containing the three tiny quotes followed by the control, separated by
four seconds of exact digital silence. Verified exact PCM preservation and file contents. Stock
and compiled samples differ, but waveform differences/durations do not establish improved clarity;
owner listening is required. The live model-info endpoint confirms Original remains loaded with
the compiled decoder, and the isolated process released its GPU memory after completion.

Private evidence includes `spaced_quote_listening_feedback.json`, `decoder_trial.py`, its log and
metadata, `decoder_waveform_comparisons.json` and `decoder_spacing_manifest.json`. No production
change or new dependency. The full-precision listening control is the next audition.

### 25.4 Full precision does not fix tiny quotes; context trial and prior quote repair audit

The owner reports garble before the first and second words in the stock FP32 trial; the third
still resembles an unwanted "money" sound. The longer control sentence is perfectly clear.
Turning off compilation and BF16 is therefore insufficient for these samples. Preserve the live
decoder settings; this supports testing short-input context, without establishing a single cause.

At the owner's reminder, checked the earlier quotation fixes and all 19 new artifact candidates.
The prior misplaced closing-quote repair and quoted-term narration packing remain in place.
All 19 current quotes have balanced curly opening/closing marks and none matches the prior
malformed interrupted-quote pattern. Closing delimiters still affect the installed model's
punctuation normalization, but the preceding delimiter-removal trial remains only partly helpful.

Prepared three bounded live-API requests with the same quoted words, voices, recorded seeds and
0.78/0.45/0.6 settings, adding the harmless same-voice lead-in "The room was quiet, and the window
was open." These are diagnostic sentences only: do not insert the carrier into a book or trim it
into production without establishing clear target speech and a safe extraction boundary. Saved
`quotes_with_context_spaced.wav`, with four seconds of digital silence between sentences. All
three requests completed; exact PCM preservation, exported content and silent-gap assertions
passed. Owner listening is pending. No production code or settings changed.

Private evidence includes `fp32_decoder_listening_feedback.json`,
`short_quote_punctuation_audit.json`, `compare_quote_context.py` and `quote_context_trial.json`.

### 25.5 BookOrbit direct-stream control sounds clean so far

The owner streamed the same EPUB directly through BookOrbit/Chatterbox and reported no audible
artifacts so far; listening to the previously marked passage is still in progress. Read the live
logs, BookOrbit's runtime provider and saved preferences, Chatterbox's saved generation getters
and live model-info. BookOrbit reaches the same Original/compiled service and the same OpenAI
speech endpoint as book generation, using Abigail throughout and speed 1, with AAC responses.
Its provider sends no seed or delivery overrides, so requests use the server defaults:
exaggeration 0.68, CFG 0.4, temperature 0.6 and seed 888 (incremented for internal chunks).
Bookmarked generated quotes used mostly Gianna, 0.78/0.45/0.6 and randomized short-unit seeds.

The 11:28–11:38 Eastern snapshot contains 54 Chatterbox requests from BookOrbit, all completed,
with matching request IDs and no synthesis failure. Request lengths range from 9 to 692 characters,
median 164; only two are at most 25 characters. The backend request-length multiset matches after
excluding the three separately identified diagnostic context requests. No backend error, token
cap or compiled-path fallback was logged; one reference mel/token adjustment is the known model
warning. An earlier failed request to a different provider is excluded from this Chatterbox test.

This is useful evidence that the same service can produce clear speech for this book, but input
length, voice, expression/CFG and seeds differ together. Do not claim a proven short-input cause
or deploy a new packing/voice/settings change from this comparison. The pending same-voice,
same-seed context trial isolates text context more directly. No production change or new synthesis
was performed during this read-only streaming review. Private snapshots and the structured review
are `bookorbit_stream_*.log` and `bookorbit_stream_log_review.json` in the first-chapter folder.

### 25.6 Same-seed context trial confirmed clear; word-only extraction audition

The owner listened to `quotes_with_context_spaced.wav` and reports every word clear with no
garble or artifact. This confirms an improvement for these three samples when only text context
changes; voice, recorded seed, quote delimiters and delivery parameters remain fixed. It does not
establish a universal cause or validate an arbitrary trimming rule for other inputs.

Reused the existing `_silence_runs` and `_silent_core` helpers on these saved WAVs. Each has exactly
one true-silence core of at least 80 ms in its latter half: 160, 120 and 290 ms, respectively.
Cut within those cores, preserving all following PCM without fades or other speech edits. The
resulting target-word samples are 950, 930 and 870 ms. Core levels are below -59 dBFS; boundaries,
PCM preservation, exports and four-second silent gaps all pass assertions. This is an audition
of three diagnostic crops, not a general production extractor. No new synthesis or production
change. `context_words_only_spaced.wav` is awaiting the owner's clarity and word-boundary check.

Private evidence: `context_listening_feedback.json`, `extract_context_words.py` and
`context_word_extraction.json`. Preserve character voice assignments and original book text when
developing any eventual context optimization; the diagnostic carrier must not reach an audiobook.

### 25.7 Word-only listening confirmed; narrow context fix deployed

The owner confirms the extracted three-word file is clear all the way through. Implemented the
confirmed technique in the shared `_speak_take` path used by sentence and paragraph packing:
Chatterbox English single-word double-quoted units of at most 25 characters receive the tested
same-voice lead-in during synthesis. The selection permits one alphabetic word/contraction with
optional terminal comma, period, question mark or exclamation mark. Multi-word phrases, ellipses,
interrupted lines, ordinary prose, other languages and other engines retain their existing path.
Voice routing, quote repairs and delivery calculations continue to use the original book unit.

Reused the existing silence helpers. Accept only one true-silence core of at least 80 ms in the
audio's latter half, cutting inside it and retaining 250–2000 ms afterward. Preserve the remaining
PCM, including quiet word onsets. Missing/ambiguous boundaries use the existing three-attempt
new-seed retry loop. Such unseparated takes never enter the retained-take list; if no safe take is
available the chapter fails with an explicit context-separation error. This remains a narrowly
validated silence heuristic, not word alignment; do not broaden it to phrases without evidence.
Record `context_trim_ms` in the chapter map and log each cut/seed; the existing M4B map merger
copies that field through. Source hashes, lengths and clip times refer to the intended book text
and extracted audio, not the lead-in. The earlier short-narration-tag cap remains in place.

99 relevant tests pass, including new checks for both packing modes, unchanged character voice,
original text hashes and output timing, exact target PCM/quiet-onset preservation, scope limits,
new-seed retries and refusal to export inseparable context. Uniform voice/delivery test fixtures
explicitly isolate routing/settings; the new integration checks exercise the real context path.
Git whitespace validation passes. Replayed all 11 bookmarked single-word quotes matching the new
selection against the live backend: every extraction succeeded on attempt one, with 660–970 ms
target audio and exact raw PCM suffix verification. The original three reproduce the approved
speech PCM exactly when applying their original whole-sample gain. The other eight have boundary
validation, not a new owner listening verdict; gain is still applied normally after extraction.

With no queued or running jobs, rebuilt/recreated only the audiobook app. Its page returns HTTP
200; deployed provider SHA-256 matches the working source, and its boundary helper reproduces the
three approved cut points. Chatterbox remains healthy, Original loaded and compiled, without a
restart or model/config change. BookOrbit streaming is unaffected. This applies to fresh generation;
no existing audiobook was overwritten or automatically regenerated. Preserve the owner's unrelated
Voice lab slider step edit. No dependency was added and no commit/push was requested.

Private evidence: `context_fix_tests.log`, `validate_context_provider.py`,
`context_provider_validation.json`, its log, eleven extracted WAVs, `context_fix_build.log` and
`context_fix_deployment.json`. Next listening check is a freshly generated first chapter.

### 25.8 Owner's next first-chapter generation is using the fix

The owner started a fresh run after deployment. Read the new queue/log snapshot: cast preparation
finished and audiobook job `acc4e597638e` is running, with 677 units, cast/sentence mode,
adaptive delivery and the original 0.78/0.45/0.6 baseline. The log confirms context removal on
eligible quotes. At the later snapshot the run reached unit 471. Two ambiguous context cuts
(units 129 and 258) were discarded and successfully separated on subsequent attempts. The
existing dialogue-length guard also discarded two truncated takes for unit 260. No error was
logged in the reviewed portion. Generation/listening are still pending; do not present this as
a completed quality verdict. No further production change. Private early-run summary is
`regeneration_context_early_check.json` in the first-chapter diagnostic folder.

## 26. After the context fix: remaining marks are two-word quotes; Whisper as a garble check (2026-09-30)

Reviewed the run from §25.8 to completion: 677 clips, 2692.458 s. 26 single-word quotes used the carrier,
21 of them in the heard part. Attempts: 673 once, 3 twice, 1 three times. The owner bookmarked the first ~19
minutes as before (9 active marks) and said a mark may trail the defect by about a second, so each window runs
from 3.5 s before to 0.7 s after.

**The single-word fix works; the gap is its scope.** Every window contains a tiny quote. In 8 of 9 it is a
two-word quote the fix excludes by design ("Kiss me!", "Love you!", "Oh god!", "Mm-hmm.", "Okay, okay!",
"Mmm, yummy,", "About us.", "I do,"). The ninth is a carrier-trimmed "Oof!", which stays unexplained. In the
heard part, 2 of 21 single-word quotes with the carrier were marked; in the previous run, 12 of 24 single-word
quotes were marked without it. Two-word quotes: 8 of 17 marked. Three or more words: 1 of 26 (ambiguous).
Longer clips: 0 of 213. Seven of the nine are Gianna, but Michael's "I do," failed the same way, and Gianna
simply speaks most of this book's interjections.

**Ruled out.** Chatterbox 0.1.6 creates `AlignmentStreamAnalyzer` only for the multilingual model, so the
English model runs without it. Even when enabled, it detects a false start but only forces an end on long tails
and repetition, so enabling it would not fix these onsets. Reference-clip endings are not the cause: Gianna's
ends the cleanest of the voices checked.

**Whisper hears the defect.** faster-whisper (already installed in the BookBridge container with cached models)
transcribed 113 clips from the M4B, read-only on CPU. The transcripts show invented syllables, mostly before
the words: "Kiss me!" became "Isn't it, Lee?", and "Mm-hmm." became "Do not tell!". Unmarked short tags show
it too: "I said." became "If I said.", and "she asked." became "But she asked.". Seven of the 8 clearly marked
clips rank among the 11 worst matches. Checked against the 17 takes the owner judged by ear in §25, Whisper
*small* matches every verdict. It flags all 9 garbled takes and hears "going in the money" in the "Oh," take
where the owner heard "money". The 3 clear context words and 2 clear controls pass. *Medium* passed two
garbled takes as "Oh" and "Nothing", so small is the better detector. It took 0.6 s per clip on 8 CPU threads.

**Same-seed trial of the 8 two-word quotes.** Replays reproduce every book take's duration exactly. With the
carrier and nothing else changed, small and medium transcribe all 8 correctly. The deployed silence rule
separates 7 of 8. "Oh god!" has a second gap between its words, so the rule would retry, and three ambiguous
takes would fail the chapter. Whisper's word times put the boundary in the right gap: the carrier's "open."
ends at 2.40 s and "Oh," starts at 2.84 s. This is machine evidence; owner listening is pending on
`two_word_book_takes_spaced.wav` and `two_word_with_carrier_spaced.wav`.

**Recommended next step, not implemented:**
- Choose carrier units by length (quotes up to about 3 words or 16 characters), not by the one-word pattern.
- Place the cut in the silence after the carrier's last word, as Whisper word timestamps locate it.
- Use a Whisper small check on every Chatterbox unit of at most 25 characters. A take whose transcript doesn't
  match the text is retried with a new seed, and the best match is kept. This is the phonetic check the
  duration and silence guards could never provide (§19, §20).
- Cost: roughly 100–150 short units per 45-minute chapter, at 0.6 s each for Whisper plus about 1.5 s of
  carrier synthesis each.
- It adds faster-whisper and a ~500 MB model to the app image: the owner's call.

No production code, settings, casts or audiobooks changed. Private evidence, listening files and scripts are
in `data/diagnostics/volume2_context_fix_run_2026-09-30/`.

### 26.1 Built: the lead-in for every short unit, and Whisper checks each take

The owner listened to the two-word files: the book takes were garbled and the takes with the lead-in
"perfect". They asked for the recommended fix and said a machine listening beats bookmarking 25
minutes by ear each time.

- **`core/speech_check.py` (new).** It loads Whisper small once per job process (faster-whisper, CPU, int8,
  up to 8 threads) from `SPEECH_CHECK_MODEL`. The compose default is `/app/models/faster-whisper-small`,
  and a missing model is downloaded there once. On this host the model was copied from BookBridge's
  cache (same SHA-256), so nothing was downloaded. `match` scores a transcript against the text,
  0-1. `PASS_SCORE` is 0.70, chosen from §26's data: it caught all 25 known-bad takes and passed all
  known-clear ones except "Mm-hmm." heard as "Hmm", which the filler rule below fixes.
- **Filler words.** Whisper leaves out or respells hesitation sounds. A filler-only text ("Um...",
  "Mm-hmm.") heard as nothing can't be judged; heard as any filler, it matches. A garbled take comes
  out as real words instead ("Do not tell!", "Oh, did you get gum?"). Text with digits isn't judged,
  because Whisper may spell numbers out.
- **Scope.** With the check on, every Chatterbox English unit of at most 25 characters (quotes and short
  narration tags) gets the lead-in. The single-word rule of §25.7 still applies when the check is off
  or unavailable.
- **The cut.** Single-word quotes keep the proven silence rule. Everything else uses Whisper's word
  times: the cut goes inside the true silence between the lead-in's last word and the unit's first.
  If Whisper hears nothing after the lead-in, it goes in the first silence after it. The silence
  rule is not used for longer units, because a quote's own pause can be the one silence it finds.
  A wrong cut there could drop a first word and still pass the match ("me about it" for "Tell me
  about it").
- **The check.** The cut take is transcribed. Below 0.70 it is retried with a new seed within the
  existing three attempts. If none passes, the best match is kept and flagged "speech mismatch". A
  plausible length that Whisper doubts ranks above a length that is surely wrong. A checker error
  costs only the check.
- **Changed: no more chapter failure over one unseparable unit.** Before, three takes with no clear
  boundary failed the chapter. At the ~7% per-attempt rate seen in the context-fix run, that risked
  about one book in four. Now the unit is spoken without the lead-in, up to three more times while the
  check doubts it, and is flagged unless a take passes. The lead-in still never reaches the book.
  Three near-silent plain takes still fail the chapter.
- **Clip map.** Each checked clip records `match`. The map still holds no book text; the heard text is
  logged at DEBUG only.
- **Tests.** 641 pass (14 new, 1 changed). The suite pops `SPEECH_CHECK_MODEL` in `tests/__init__.py`,
  so it never loads Whisper inside the live image. `faster-whisper==1.2.1` was added to requirements.

**Live check.** Run through the real provider with real Whisper and the live Chatterbox, starting from
each unit's recorded book seed:
- The first build got 17 of 18 units right.
- "Um..." failed to separate: Whisper had left the word out, so there was no boundary to find. The plain
  fallback it fell back to was garbled. That led to the no-following-word cut and the retried plain
  fallback.
- "Mm-hmm." was heard as "Hmm", which led to the filler match.
- After both fixes, all 18 pass on the first attempt with match 1.0: the 9 marked windows plus 7
  unmarked units Whisper had flagged.
- Whisper took 0.65 s per transcription. A short unit now takes about 3.3 s instead of about 1.5 s:
  roughly 5 more minutes on this dialogue-heavy 45-minute chapter (~170 short units).
- The owner deleted the book in BookOrbit for the next test, which also removed its bookmarks. Seeds and
  voices were rebuilt from the run's log and matched the saved map exactly.

The app container was rebuilt and recreated three times, each after checking that the queue was idle.
The page returns HTTP 200, the deployed source hashes match, and Chatterbox was not restarted. The next
check is the owner's fresh first-chapter run. The clip map now doubles as the machine listening report:
clips with `flagged` or a low `match` are the ones to hear.

### 26.2 Review fixes

An outside review of §26.1 found three problems. All three were confirmed and fixed.

- **P1: the cut could keep lead-in speech.** `_carrier_cut` accepted any silence that merely overlapped
  Whisper's rough boundary window, and chose the one nearest its middle. If Whisper ended "open." early,
  a pause *before* "open" could win, and the book would start with "open.". The match would not catch it:
  "Open. Kiss me!" scores 0.71 against "Kiss me!".
  - Measured on the 8 saved lead-in takes: Whisper small ends "open." 100–170 ms *before* the real
    silence starts on every one. The unit's first word lands within about ±80 ms of where the silence ends.
  - Now the silence must *start* after Whisper's end of "open." and before the unit's first word plus
    150 ms, and the earliest such silence is used. A pause before "open" (about 300 ms earlier) can't
    qualify. A late word time finds no silence, and the take is retried.
  - Added `speech_check.leaked`: a cut take whose transcript starts with a lead-in word the text doesn't
    start with counts as unseparated and is retried. The take is heard before the length checks, so a
    take kept for the least-wrong length is leak-checked too.
- **P2: the plain fallback skipped the length guards.** Takes spoken without the lead-in now go through the
  same verdict as any take (`_verdict`: length checks, then the speech check). With the check off, a
  runaway take was kept unchecked before.
- **P3: an exhausted fallback under-reported attempts.** It now reports every request made: 3 with the
  lead-in plus 3 without, so 6.

645 tests pass (4 new: a pause before the lead-in's last word, a leaked cut retried, the plain fallback's
length checks and attempt count, and `leaked`). Rebuilt and recreated the app with an idle queue; the
deployed source hashes match. The live replay of the 18 units passed each on the first attempt, with the
same cut points as before the fix.

### 26.3 First run with the speech check; the time estimate after app restarts

The owner regenerated the same first chapter (14:29–14:53 Eastern): 677 clips, 44:45. There were 203
short units: 200 got the lead-in and Whisper checked 202 of them.
- **Attempts.** 648 units passed first time, 21 needed two attempts and 5 needed three.
- **The lead-in could not be separated** on 27 attempts, and a leak was caught twice. All of these were
  recovered: three units (two "she said"-type tags and "N-- no, Ma,") were spoken without it, and each
  passed the check.
- **The pipeline flagged 2 clips**:
  - "Jeeesuss..." at 6:47: Whisper heard "Jesus. Jesus.", score 0.62. Two earlier takes were too short
    for an ellipsis line.
  - "And then..." at 24:04: too short for an ellipsis line, although Whisper hears it correctly.
- **Independent listen.** Whisper small went over all 203 short clips of the finished M4B, the method
  that matched the owner's marks in §26:
  - 2 suspects: the flagged "Jeeesuss..." and "Oh," at 15:05, heard as "Ho!" in the final audio but "Oh"
    by the pipeline's own check.
  - No lead-in leaks.
  - On the previous run's first 19 minutes, the same method doubted 12 of 96 short clips, 8 of them the
    owner's marks. On this run it doubts 2 of 97.
- **Time.** Generation took 24 minutes, against 16 without the check: every short unit is now a longer
  request plus one or two transcriptions, and there were about 40 retries. Unseparated attempts
  (27 of 200 units) are the main cost worth tuning later; the strict start rule of §26.2 is the likely
  source.

The owner's listening verdict is in §26.4.

**Bug: the dashboard's time estimate was 0.** The chapter stats behind the estimates live in the page's
Gradio session. The four app restarts of §26.1–26.2 emptied them while the owner's open page still showed
its chapter table, so the next Analyse queued both the cast (normally 144 s) and the book (760 s) with an
estimate of 0 ("less than a minute left"). Generation itself was unaffected: the job's settings matched the
previous run's. `stats_for_estimate` now re-reads the chapters from the book at queue time whenever the
page's stats don't cover the ticked chapters. It takes a fraction of a second, and a read failure only logs
a warning and still queues. Tested in the live app with empty stats: 760 s and 144 s again. 646 tests pass.
Deployed with the queue idle, after the run finished. The formula itself is unchanged and now about half
the real time for a dialogue-heavy chapter; recalibrating it for the speech check is open.

### 26.4 Listening verdict, and a second book

**The owner listened through the first 19 minutes and bookmarked 2 spots.** The previous three runs of this
chapter had 25, 19 and 9 artifact marks.
- **6:47, "Jeeesuss..." said twice.** This is the clip the pipeline flagged. The owner judged it not an issue.
- **10:31, one artifact.** It is clip 170, a five-word question in Gianna's voice. The text is 27
  characters with its quote marks, just over the 25-character limit, so the clip got no lead-in and no check.
  Whisper small hears it and the clip before it correctly, so this is a sound artifact that speech
  recognition can't hear, like "Oof!" in §26, not wrong words. Widening the limit would not have flagged it.
  Whether the lead-in would have prevented it is unknown.

None of the three places the review pointed to ("Jeeesuss...", "Oh," at 15:05, "And then...") was a
problem, apart from the doubled word. The owner called the result good enough to commit.

**A second, different book: chapter 6 of a narration-heavy novel.**
- **Size and time.** 1,153 clips, 90:21 of audio, 11 voices; the narrator reads 1,008 units. It took 36
  minutes against the fixed estimate of 25.2, which was no longer 0 after §26.3.
- **Short units.** 145 of them (13% of units, against 30% in the first chapter): 137 got the lead-in and
  138 were checked.
- **Attempts.** 1,121 clips passed first time, and the pipeline flagged none.
- **Unit check.** The review rebuilt every unit with the deployed provider, with no synthesis
  (`rebuild_units.py`), and matched all 1,153 text hashes and voices against the clip map.
- **Independent listen.** 1 suspect of 145 short clips, a false alarm: "Three," was transcribed as "3.", and
  `match` only declines to judge when the *text* has digits. No lead-in leaks. The owner's listening is
  pending.

**Open.**
- **Separation failures.** The lead-in failed to separate on 27 attempts (first chapter) and 49 (second
  book). The retries recovered them, but 3 and 8 units were then spoken without it (all passed the check).
  The strict start rule of §26.2 is the likely cause and the main time cost to tune.
- **Digits in transcripts.** Whisper writes "3" for "Three"; this probably caused some of the second book's
  6 mismatch retries. Normalising numbers before `match` would fix it.
- **The time estimate** runs 30–50% low with the check on.
- **Sound artifacts** that a transcript can't reveal remain possible, but are now rare: one in 19 minutes of
  dialogue-heavy audio.

Private evidence is in `data/diagnostics/volume2_context_fix_run_2026-09-30/` and
`data/diagnostics/second_book_2026-09-30/` (`rebuild_units.py`, `review_book_run.py`, `review.private.json`).

### 26.5 Fewer lead-in retries, a dropped-first-word bug, and numbers in transcripts

The owner asked to look into the two costs from §26.4.

**Which units failed.** The logs of both runs had 76 unseparated attempts spread over 49 units. Nearly all
were short narration tags such as "she said." and "he snarls.". Single-word quotes, which use the silence
rule, failed on 1 of 43 units. In the second book, the narrator's voice (Elena) failed on 25 of its 116
short units.

**Why.** 97 fresh lead-in takes reproduced it: the 49 failing units plus 49 same-voice controls, analysed
and saved (`leadin_tuning_2026-09-30/repro_takes.py`).
- **Controls:** 0 of 48 unseparated. **Failing units:** 13 of 49.
- In every failure the model paused only 0–70 ms of true silence after "open.", against a median of
  200 ms in the takes that separated; the cut rule wanted 80 ms. Whisper still heard a full stop in 11 of
  the 13.
- The worst cases run the two together as one plausible sentence: "the window was open so much", "open
  she said".

**Changes to the cut:**
- **30 ms of true silence is enough where Whisper locates the gap** (`_CARRIER_GAP_MS`). The single-word
  silence rule keeps its 80 ms. In no take did a true silence start between Whisper's end of "open." and
  the real gap, so a brief closure inside "open" can't be taken for it.
- **The gap must start by Whisper's start of the unit plus 50 ms** (`_UNIT_ONSET_SLACK_MS`, down from
  150). Measured on 150 takes, the real gap starts between 470 ms before and 30 ms after that point, and a
  pause inside the unit's first word no earlier than 90 ms after it.
  - **This fixes a bug in §26.2's rule.** Its 150 ms allowance cut one take of "she bites back." to
    "Bites back.", which scores 0.83 and would pass the check.
  - The independent transcripts of both finished books show no dropped first word; the only hits were
    homophones.
- **`speech_check.clipped`**: a cut take whose transcript lacks the unit's first word is retried, just like
  a leak. Spelling variants, stutters, fillers and one-sound homophones ("Eye" for "I", "C." for "See!",
  "Ho!" for "Oh,") still count as the word.
- **A lowercase unit is capitalised after the lead-in** (`_new_sentence`). Chatterbox's `punc_norm`
  capitalises a request's first letter, so before the lead-in existed every tag was already spoken as
  "She said.". On 32 same-seed pairs of the hardest tags, the unseparated count fell from 11 to 8 under the
  new rules, and pauses under 30 ms fell from 7 to 4.

**Offline check** of the new rules on all 169 saved takes: the 8 owner-approved cuts are unchanged, no
existing cut moved, and every new cut was heard correctly. The new checks rejected four cuts:
- one wrong in-word cut ("Edges me on");
- two genuinely bad takes;
- one homophone ("See!"), since fixed.

**Live check.** The working-tree `_speak_take` ran against live Chatterbox on the 49 units that failed in
the two books.
- Unseparated attempts fell from 76 to 20, and units spoken without the lead-in from 11 to 4.
- Two stubborn units were flagged "speech mismatch": "N-- no, Ma," and "So much!".
- "So much!" shows the remaining limit: the model reads "…the window was open so much!" as one sentence.
  Its first plain retry came out garbled ("In college!", scored 0.12), and the check caught it.
- Changing the lead-in text itself could fix the run-on, but would need the owner's ear again.

**Numbers.** A clear "Three," heard as "3." scored 0.00 and was retried twice. `match` now spells digits in
the transcript as words: cardinals, ordinals like "21st", and thousands separators. Text that contains
digits is still not judged, because a narrator may read "1990" several ways.

**Side finding, left unchanged.** `speech_tags.SPEECH_VERBS` is almost all past tense. In a present-tense
book, "he murmurs." is not recognised as a tag, so it misses the calmer short-tag cap of §23.1; "panted"
and "sobbed" are missing too. This list also feeds cast attribution, so any change belongs in its own
piece of work.

649 tests pass. Deployed with the queue idle; the deployed source hashes match.

## 27. Speech tags point one way; present-tense tags (2026-09-30)

An outside review of the first chapter's saved cast found four misattributed spots in its first 26
minutes. The owner asked to take on its first recommendation, the speech-tag rules, together with §26.5's
side finding (a verb list that was almost all past tense), since both live in `core/speech_tags.py`.

### 27.1 What went wrong

At 1:29 the husband's brother asks a question, and the narration after it reads "<the wife> answered
without any hesitation:", introducing her answer. The after-tag rule read that narration as the
question's own tag. Even without that, the paragraph rule would have lent her to the question, as the
paragraph's only named speaker. Tagged lines are *anchors*: they are never asked, so the model could not
correct it. The review's other spots are the model's own errors, and are not addressed here.

`SPEECH_VERBS` knew only "says" and "asks" in the present tense, so a present-tense book lost its named
tags, the short-tag exaggeration cap (§23.1) and its "I ask" narrator votes (§22.3).

### 27.2 Changes

- **Direction.** Narration whose first sentence runs on to a colon introduces the next quotation and is
  never the previous one's tag. A before-tag may carry a short phrase that names no one else before its
  colon or comma ("answered without any hesitation:").
- **Contradictions.** A quotation named one way before it and another way after it gets no anchor; the
  model decides.
- **Lending.** An untagged quotation now inherits a speaker only by going forward: after that speaker's
  line in the same paragraph, when nothing between them ends in a colon or names anyone else. "She
  smiled." keeps the speaker; "Tom smiled." and "Ada looked up:" don't. As before, the
  paragraph's tags must name only one person. A first, stricter version, which broke the run at any
  narration, released 18 correct anchors in the real chapter; this one releases 13. All 13 are cases
  where the narration names someone else or a pronoun tag is now recognised.
- **Verbs.**
  - The present tense of every verb, plus voiced tags (murmurs/murmured, sobs, panted, gasped, snarls…).
  - Base forms ("I ask", "they whisper") count for "I" and pronoun tags only, because after a name,
    "tell" or "call" is rarely a tag.
  - `has_speech_tag` still reports a tag whichever quotation it belongs to (delivery cues, quote
    packing).

### 27.3 Evidence

- **Anchors on the benchmark.** Two invented passages were added to the labelled benchmark
  (`experiments/multivoice/fixture/05, 06`) with the real book's patterns. Across all six passages the old
  rules locked 39 lines, 3 of them wrongly, all the 1:29 pattern. The new rules lock 41 with none wrong.
  Passages 01–04 are unchanged.
- **Benchmark accuracy** (qwen2.5:14b, 329 lines, before → after):
  - Overall: 82.4% → 83.3%.
  - Passage 05: 81.1% → 91.9%. All three formerly locked lines went to the right speaker once asked, and
    a character the model had split off merged back.
  - Passage 06: 90.9% → 86.4%, one line: the model gave a vocative "Dom." to Dom.
  - Passages 01–04: identical.
  - Requests: 17 windows either way, with no unusable replies.
- **The real chapter**, run on a fresh roster; the owner's saved cast was only read.
  - Anchors fell from 79 to 70.
  - The 1:29 question now goes to the brother (saved: the wife).
  - An "Um..." now goes to the wife (saved: her father); the next sentence is hers.
  - Two "I said" lines came back unknown in this chapter-only run. They are never anchored under either
    rule set, and the full cast job resolves the narrator.
  - Every other line matches the saved cast.
- **The second book's chapter:** anchors went from 3 to 5, both new ones present-tense tags and correct.

Tests: 659 pass. They include the review's cases sanitised (a colon introduction, two speakers introduced
in one paragraph, interrupted speech, continuation through the speaker's own action, an earlier
quotation, narration naming someone else, contradictions, present and base-form tags) and an
attribution-flow check: the question is asked, and the introduced answer stays anchored.

Not addressed: the review's second recommendation (a targeted review pass for the model's own errors at
17:32 and 21:34 and the three unknowns at 12:52), stricter character merging, and a stray opening quote
at 15:47. The saved cast keeps its old attribution until the book is re-analysed. Private evidence is in
`data/diagnostics/tag_direction_2026-09-30/`.

## 28. A second look at the lines the text contradicts (2026-09-30)

The outside review's second recommendation: attribution asks each window once and keeps any well-formed
answer, even one the text contradicts. The owner asked for it after §27.

### 28.1 What the errors had in common

The review's remaining spots in the first chapter, mapped to lines:
- **12:52.** The narrator's "Damn!" I said. / "You mean he wasn't…?" and the wife's reply were all unknown.
  The model had answered "Narrator", which names no one, because it was never told who the "I" is. So
  they got the fallback dialogue voice.
- **17:32.** "So, Dad," she said after a sip, "…knows about us." The first half went to the narrator: a
  "she said" tag on a man, and one sentence split between two speakers.
- **21:34.** Midway through the wife's story, "<the wife> looked at me." and then "Isn't that sweet?". The
  line went to the narrator ("me"), though the narration names only her.

These are things the text itself shows without understanding the story.

### 28.2 Design

`core/cast_review.py` (new, no LLM) flags lines on five signals:
- no speaker;
- a he/she tag against the assigned character's gender;
- an "I"-tagged line not given to the chapter's clear narrator (at least 2 "I said" votes and a majority,
  as `chapter_narrators` decides);
- a speaker change between two quotations of one paragraph where the narration between ends in no colon
  and names no one else, and the second quotation has no tag of its own; or where one sentence is split
  by a tag ("…a good girl," he said, "but…");
- contradictory tags (§27).

Tag-anchored and continued lines are never asked.

`cast_llm.review_lines` runs once per chapter, after the windows and before lines are counted, so
profiles, narrator detection and voice matching see the corrections.
- **Grouping.** Flagged lines within 3 paragraphs share one request, up to 12 lines. Each request shows 4
  paragraphs before and 2 after.
- **What the model sees.** Tag-named lines are shown as [Name] (certain), the first pass's other answers
  as [Name?] (a guess that may be wrong), and the asked lines as [#N].
- **Narrator.** It is named when the "I said" votes establish one, and described when their character is
  only "I".
- **Rules in the prompt:** "she said" is a woman; a quotation split by a tag is one speaker's; a paragraph
  usually holds one speaker; someone addressed by name is usually not the speaker.
- **Answers.** A named answer replaces the first one, "unknown" keeps it, and an unusable reply changes
  nothing and is counted. Continued lines follow their corrected first part.
- **Reporting.** Stats go into the cast and the analysis log.

### 28.3 Evidence

- **Labelled benchmark.** 83.3% (after §27) became 83.9%: 276/329 lines from 2 review requests for 4
  flagged lines, both corrected. No passage got worse, and the first-pass windows stayed at 17.
- **The first real chapter**, attributed with §27 and §28 on a fresh roster (the saved cast was only read):
  - 5 review requests for 11 lines, 8 changed, 0 unknown lines (the §27-only run had 5 unknown).
  - Fixed: the 12:52 narrator lines, 17:32 and 21:34.
  - Also fixed, beyond the review's list: the sentence split by "he said" (its second half had gone to the
    wife), and the two "I said" lines §27's run left unknown.
  - Still wrong: the wife's 12:52 reply went to her mother, who only appears later, where it had been
    unknown. A second run gave identical results.
- **The second book's chapter** (first-person, present tense): 2 requests for 2 lines, 0 unknown. Both
  changed lines were "I"-tagged and moved to the chapter's narrator entry (a chapter-only run had split
  the narrator into "I" and their name). Nothing else differed from the saved cast.
- **Tried and dropped: a line calling its own speaker by name** (the benchmark's "Dom." given to Dom).
  It flagged 2 more benchmark lines, but asked again the model kept its answers: no gain for one more
  request.

Limits: the same model reviews its own answers. The gain comes from wider context, the narrator's name
and the explicit contradictions, and a second answer is no guarantee. Merging (the review's fourth
recommendation) and the stray-quote paragraph are still open.

Tests: 673 pass. They cover each signal, grouping and rendering, the narrator vote, the review flow on
the three sanitised cases (asked, corrected, counted), an "unknown" review keeping the first answer, an
unusable review changing nothing, and a consistent chapter asking nothing more. Private evidence is in
`data/diagnostics/tag_direction_2026-09-30/`.

## 29. The narrator reads what the cast can't place; more careful merging (2026-09-30)

The outside review's third and fourth recommendations.

**Fallback voice.** In cast mode, a line with no speaker, or a character with no voice yet, was read by the
Dialogue voice setting. The owner's default there was a voice belonging to no character, so the unknown
lines at 12:52 came out in a third voice between the husband's and the wife's, exactly where attribution
had failed.
- The Dialogue voice list now starts with "(the narrator's voice)", and that is the default in cast mode.
  The provider already read an absent dialogue voice as the narrator's, so the change is in the UI and in
  what gets queued. In a collection's first-person story that means the teller's voice.
- Picking a real voice still works. "Narrator + dialogue voice" mode still needs one, or it would be
  single voice.
- The cast summary names the fallback: "N with no speaker found (read by the Dialogue voice setting: the
  narrator's voice unless you pick another)".
- With §28, unknown lines are rarer to begin with: the first chapter now has none.

**Merging.** Both of the review's reproductions held:
- "Mr. Smith" and "Mrs. Smith" became one character even with recorded genders of male and female.
  Keys drop titles, and an exact name match skipped the gender check.
- "Ann" and "Anna" merged by prefix, as did "Paul" and "Paula" until their genders were known.

Changes:
- A gendered title counts as gender evidence (`cast.title_gender`): Mr/Sir/Lord/Uncle… male, Mrs/Ms/
  Miss/Lady/Aunt… female.
- An exact name match is refused when the genders conflict, recorded or implied by title. The second
  person then gets a key that keeps the title ("mrs smith"), where later mentions find them.
- A first name merges with a longer one only when that is at least two letters longer (Ben/Benjamin,
  Chris/Christopher, Ann/Annie), or through the nickname table (Tom/Thomas).

Checks:
- **Benchmark:** unchanged at 83.9%; its passages are full of titles (Master, Lady, Constable, Captain,
  Aunt, Dr., Nurse).
- **The first real chapter's cast:** the same 5 characters with the same genders and aliases (the wife's
  short name still merges with her full one), and the same attributions as §28.

679 tests pass. They cover title-kept keys in either order, the gender check on an exact name, one-letter
names kept apart while real short forms merge, the family-word alias, the narrator fallback in the queue
and the provider, and the dialogue mode still needing its own voice.

## 30. A whole collection re-cast: family words, pairs and one person under two names (2026-09-30)

The owner asked for a full cast of the first collection (6 stories, 1,521 lines) to verify §27–§29. It
was run with the app's own chapter selection, settings and cast job (`run_cast_analysis`); the saved cast
was backed up before each run. Compared with the old full-book cast from before §22's per-story fixes:
- **Better:** no unknown lines (was 4), and story 1's narrator is its own "I" (the old cast had story 2's
  narrator). Every story-1 correction of §27–§28 held.
- **Three new problems, which this section fixes:**

**1. "Dad" was two men.** In story 2 the narrator's husband is their son's "Daddy", and the model aliased
him "Dad" for that chapter. The narration's `"…," Dad said` (the narrator's *father*) is a tag, so its
line was anchored to "Dad", which resolved to the husband: all 25 of the father's lines. The old cast only
avoided this because that run's model happened to create a separate father.
- A tag naming only a family word ("Dad said", "said Mother") now tags but anchors nothing, and lends
  nothing. Whose "Dad" it is depends on who is telling it, so the model decides, as for a pronoun tag.
- A name behind a title still anchors ("said Aunt Ruth").
- Asked, the model created "<narrator>'s father" and gave him all 25 lines.

**2. A pair became a name.** The model answered "<twin> and <twin>" for one character, and the roster's
"the fuller name becomes the display name" rule showed one twin as the pair. A speaker or character
answered as "X and Y" is now X, and joint aliases are dropped.

**3. One doctor, two characters, two voices.** Story 5's doctor says "please call me <first name>". The
model gave that very line to the first name, so her earlier lines stayed "Dr. <surname>" (18) and the rest
went to the first name (61). The old cast only merged them because the model had listed the alias early.
- **An introduction line merges the name into its speaker.** When a line says "call me X", "my name is
  X" or "the name's X" (not denied: "don't call me X"), and X is a separate character who first appeared in
  this chapter, X is folded into the line's speaker. An X not met yet becomes the speaker's alias.
- **When the line was given to X itself, the model is asked one question.** This applies when the
  introduction is X's first line: "Is X a new person, or another name for one of these characters who
  spoke just before?" A "new" answer (a newcomer introducing themselves), an unusable reply or a name
  outside the candidates changes nothing.
- On the real story that is one question, answered correctly: 79 lines, one character.
- **The editor gains Merge.** "Adjust the cast" has a "Same person as" list and a Merge button, with a
  confirmation. The selected character's lines in every chapter, name, aliases and narrator role move to
  the chosen one, which keeps its own voice (`cast.merge_characters`).

**Results.** The third whole-book run on the final code, compared with the first of the day:

| | first | now |
|---|---|---|
| story 2, the father's 25 lines | the husband | the father |
| story 5, the doctor | 2 characters (18 + 63 lines) | 1 (79) |
| a twin's display name | "<twin> and <twin>" | the twin |
| unknown lines / review requests | 0 / 11 | 0 / 11 (+1 identity question) |

- Analysis time was about 7½ minutes per run.
- The benchmark is unchanged at 83.9%.
- One spelling split remains: the author spells a twin's name two ways (26 lines and 1). It was merged
  with the new editor action, the same code the button runs. The saved cast now has 25 characters.

**Shortfalls, not fixed:**
- **One story-1 line is still wrong.** The wife's reply at 12:52 goes to her mother (§28), and the review's
  second answer repeats the mistake.
- **A name spelled two ways by the book is not merged automatically.** Merging near-identical names is
  the loose matching §29 removed (Ann/Anna, Andrea/Andrew), so this is left to the editor's Merge.
- **An editor merge doesn't survive a re-analysis.** Re-analysing rebuilds the characters, carrying over
  only picked voices, so a merged pair can split again unless the model or an introduction line joins
  them.
- **The identity question needs an introduction line.** A character called by title in one part and by
  first name in another, with no "call me …", can still split.
- **The model can still confuse a vocative with the speaker.** The benchmark's "Dom." given to Dom
  (§28) stays wrong.
- **Voices are not picked by the analysis itself.** They are suggested when the cast is first shown on
  the page, or when its book is queued.

**Deployment slip.** One rebuild did not complete, and `compose up` left the old container running. The
deployed-source hash check caught it before any result was trusted. The analysis started on the stale
code was stopped, and Chatterbox, which that job had unloaded, was reloaded by hand.

687 tests pass. They cover family-word tags (asked, and the answer joining the character given that
alias), pairs, the introduction merge (by the speaker, given to the new name with "same as" and "new"
answers, denied names, someone from an earlier chapter, an alias for a name not met yet), `Roster.merge`,
`cast.merge_characters` (lines across chapters, narrator and point-of-view roles, voice kept) and the
editor's Merge. Private evidence and the four whole-book casts (the old one, runs 1–3) are in
`data/diagnostics/tag_direction_2026-09-30/` and `data/cast_backups/`.
