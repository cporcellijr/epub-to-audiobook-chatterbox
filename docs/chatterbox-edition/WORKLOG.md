# Chatterbox edition: work log and findings

Covers 2026-09-25 to 2026-10-01. Written for the owner and for any agent reviewing or continuing
this project. Later investigations name the affected books where needed to trace the evidence.

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

## 31. Five cast review fixes, tested between phases (2026-09-30)

The owner asked to fix each review finding in a separate phase, testing before proceeding. CodeGraph
was used to trace the shared functions and their callers. Each new regression reproduced the bug on
the old code before the fix was applied.

| Phase | Fix | Passing checks before the next phase |
|---|---|---|
| 1 | Self-introductions only match direct sentence openings, optionally "And" / "please". "Don't ever call me Beth", indirect reports and nested quoted introductions no longer merge another character or move their lines. The existing Dr. Hale / Lena introduction still merges. | 138 cast, attribution, speech-tag and review tests |
| 2 | Title periods are ignored when finding the first sentence of a colon introduction. "Mrs. Marsh answered: …" tags the following quotation, without claiming the question before it. Real sentence endings still separate tags. | 139 tests in the same set |
| 3 | An incompatible exact-name match falls through to compatible existing candidates instead of immediately returning "new". Repeated male/female "Alex" mentions reuse two entries; titled matches also check gender. Saved rosters retain this behavior. | 140 tests in the same set |
| 4 | The default dialogue fallback is selected after the chapter narrator. Unknown lines and characters without a voice use that chapter's narrator; an explicitly selected fallback still wins. | 158 tests, including the actual voice provider with mocked speech responses |
| 5 | A voice transferred by the editor's Merge carries its `voice_picked` flag. Picked voices survive Suggest again and voice carry-over during re-analysis; suggestions stay suggestions. A target with its own voice retains its voice and flag. | 159 tests in the combined set |

Final review added a regression for a reported quotation at the very start of a dialogue line
("'Call me Beth,' John said."). Only the outer quotation marks are removed before checking for
nested speech, so that case also stays unmerged. The 141 core checks passed again after this adjustment.

**Final validation:** 698 tests pass with the full application discovery command from §8, using the
existing `epub_to_audiobook:local` image and the working source mounted read-only. Networking was
disabled. The first full run had seven UI errors because the read-only mount prevented the UI from
creating log files; rerunning with a temporary in-container `/src/logs` folder passed all 698.
`git diff --check` also passes. Five regression test methods were added to existing test files;
no dependency or additional model request was added.

**Scope and limit:** only source, tests and this log were changed. These fixes have not been deployed,
and no saved cast or audio was regenerated. Introduction matching is intentionally conservative:
mixed quoted speech and introductions outside the recognized direct sentence forms stay unmerged.
The remaining attribution and re-analysis limitations in §30 still apply; these tests do not establish
a new live-book accuracy score.

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

## 33. §31–§32 checked against §26–§30 (2026-09-30)

The owner asked whether §31's fixes break any of the earlier work. Each change was checked against the
code it touches, then on real books, the benchmark and the full suite.

**§31, change by change:**
- **Lead-in and speech check (§26): untouched.** Phase 4 only moves the line in `voice_of` that picks the
  default dialogue voice. `_speak_take`, the carrier cut and the checker are unchanged.
- **Narrator fallback (§29): consistent.** "The narrator's voice" now means a collection story's own
  teller, which is what §29 meant for a first-person story.
- **Colon introductions (§27): consistent.** Phase 2 only stops a title's period from ending the
  sentence.
- **Roster gender check (§29): kept, and a gap in it closed.** Before, every later mention of a name
  shared by a man and a woman made yet another character.
- **Introduction merge (§30): kept, one loss fixed here.** Every dialogue line naming its speaker in the
  12 saved casts' books (22 lines) was run through both versions:
  - the collection's doctor ("…And please call me <name>.") still merges;
  - §31 fixed a bug in §30: the "don't call me" check knew only the straight apostrophe, so "Don’t call
    me Lady <name>" was read as an introduction; "My name’s …" with a curly apostrophe now counts too;
  - §31 lost "You can call me <name>." **Fixed:** "you can", "you may" or "just", and a comma after
    "please", may come before "call me". "You can't call me X" and "Don't just call me X" still name no
    one.
- **Benchmark: unchanged** at 83.9% (276/329), the same on every passage, 17 windows, 2 review
  requests.
- **703 tests pass** (§8's command): §32's 702 plus the new introduction-forms test, which fails on
  §31's pattern.

**§32 and saved casts.** Casts find a chapter by the hash of its text, so every saved cast was checked
against the parser before and after §32:
- **9 of 12 unchanged**, including both collections from §22–§30.
- **The one-chapter book §32 re-split** now reads as 8 stories. Its one-chapter cast (183 lines) covers
  none of them, as expected.
- **Two older collections changed too, correctly.** Each story's title page now stands apart from its
  copyright page, and the "other stories by the author" list now stands apart from the story's last
  chapter. That uncovers 2 analysed chapters in one cast and 1 in the other (92 lines each), and the
  first cast's later chapters are renumbered.
- **The editor warns about this.** `cast_coverage_gaps` reports the uncovered chapters when the book is
  queued, where those chapters would otherwise get the dialogue voice. Re-analysing the three books fixes
  them.

**Deployment.** The running image was built at 20:14, a minute after §32's commit, so §31 and §32 are
live, although both sections say nothing was rebuilt. This section's fix is not deployed. Private
evidence is in `data/diagnostics/review31_2026-09-30/`.

## 34. Tone matching: each voice turned down to its own clip (2026-10-01)

The owner asked about raising the bitrate for better sound. On questioning, the complaint turned out
to be specific: the treble sounds turned up and there is a faint hiss. Neither came from the encoder.

### 34.1 What it was

- **The bitrate wasn't it.** On 24 kHz mono speech, ffmpeg's AAC encoder keeps the full band at 64k: a
  clean speech test kept 98% of the 10–12 kHz energy at 64k, 96k and 128k. A higher rate only lowers the
  coding noise (4–12 kHz noise-to-signal 14.1 → 22.2 → 28.0 dB). 128k only reaches ~107 kb/s on this
  material.
- **First comparison against the wrong clip.** The first excerpt was compared with Adrian's clip (the
  book's narrator), and showed 11 dB too much at 10–12 kHz. The owner heard that the excerpt was Emera,
  voiced by Teen. Teen's clip matched that excerpt's top band, so that comparison proved nothing.
- **The fair test:** the same two lines per voice, lossless, with the book's settings and a fixed seed,
  each compared with its own clip (dB, generated minus clip):

  | voice | 4–6k | 6–8k | 8–10k | 10–11k |
  |---|---|---|---|---|
  | Teen | +6.8 | +11.9 | +7.3 | +12.7 |
  | Everett | -3.3 | -2.1 | +11.4 | +6.5 |
  | Adrian | 0.0 | 0.0 | +3.8 | +14.2 |
  | Maya | +2.3 | +4.0 | -3.1 | +0.3 |
  | Gianna | -1.5 | -3.3 | -1.3 | -6.1 |

  Every take's background was quieter than its clip's (2.9–13 dB). The "hiss" is fizz on the voice,
  not noise in the pauses.
- **Ruled out:** the Perth watermark (it adds -55 to -67 dB of signal above 2 kHz and no change in
  level), the server's post-processing (DC filter off, peak normalisation only) and the AAC encode.
- **One clip is the problem itself:** `love poem.wav` (Charra, 520 lines in the book being made) is
  extremely bright, with 6–8 kHz at -1 dB against the speech core where clips usually sit 15–30 dB lower.
  It is also the hissiest custom clip. Matching keeps a voice like its clip, so this one needs a better
  clip.

### 34.2 Choices

- **Per-voice matching over one fixed filter.** A fixed cut (fizz above 10.5 kHz, -3.5 dB around
  3.5 kHz) helped Teen but would dull Gianna, who already comes out darker than her clip. The owner
  compared now / fixed / matched for Teen, Adrian, Everett and Maya and said matching "made a huge
  difference".
- **No pause gate.** By ear, "fixed filter" and "fixed filter plus quieter pauses" were the same.
  Measured, the gate took 1.5 dB off the pauses, and a stronger gate would clip word endings.
- **Measured from the book itself, not a calibration pass.** A calibration pass would cost GPU time for
  every voice in every book. The book's own audio costs nothing extra and reflects its delivery
  settings. The measurement accumulates per voice over the whole run, and a voice is left as generated
  until 8 s of it has been heard. A voice with only a line or two in the book is never touched.
- **How:** frames within 30 dB of a take's peak, third-octave bands from 1.6 to 11.8 kHz, each against
  the 300–1600 Hz core. Cuts only, at most 12 dB, smoothed across neighbouring bands. The filter is
  zero-phase FFT, padded so nothing wraps around a take's ends. The clip is read through ffmpeg:
  pydub's resampler would fold a 44.1 kHz clip's top octave into the bands being compared.
- **96k for lossy chapters and M4B re-encodes**, one constant (`m4b.LOSSY_BITRATE`) instead of two
  hard-coded "64k".
- **Peak guard on every unit of a lossy chapter** (-1 dBFS, `delivery.PEAK_GUARD_DBFS`), not only
  adaptive ones. Decoded finished books went past full scale: The Sofa reached +1.53 dBFS (1,181
  events in 7.1 h) and Apex Prey 2 reached +1.21 dBFS (18 events in 3.5 h). A 16-bit decode clips those.

### 34.3 Evidence

- **Live chapter** (working tree in a scratch container against the running Chatterbox): Adrian
  narrating, Teen speaking the quotes, the book's settings, 22.6 s and 29.6 s of speech. dB against
  each voice's clip:
  - Adrian at 10.9 kHz: +13.1 before, +4.3 after (the 12 dB cap plus smoothing). 1.8–3.6 kHz within
    1 dB, untouched.
  - Teen at 5.7–7.2 kHz: +3.2/+6.1 before, +0.3/+1.2 after. Every band within 3.4 dB.
  - Old pipeline (64k, unmatched, unguarded): 63.5 kb/s, decoded peak -0.80 dBFS. New: 86.1 kb/s,
    -1.13 dBFS. Neither had samples over full scale in this one-minute test.
- **Tests:** 713 pass (§8's command): 697 at the previous commit by the same command (§33 recorded 703),
  plus 8 tone-match unit tests, 7 provider tests (matched
  and unmatched voices, the switch, Kokoro, carry-over between chapters, peak guard on lossy and not on
  lossless output) and 1 queue test (old jobs default to on).

### 34.4 Not verified

- **Not yet run on a whole book.** Each chapter logs `Tone match <voice>: up to N dB from F kHz`.
- **A book resumed across this change mixes matched and unmatched chapters.** Goblin Stepsister
  Obsession has chapters 1–42 from before it. Each chapter's `.clips.json` records every clip's voice
  and timing, so finished chapters could be matched afterwards. That is not built.
- **The clip and the book say different words**, so a voice's first chapter measures it on less
  speech than later chapters do. The cut can shift by about a dB as the book goes on.
- **Listening samples and measurements:** `data/diagnostics/treble_hiss_samples/`.

**Deployment.** Deployed 2026-10-01 03:17 with the queue empty; the five changed files in `/app_src`
match the working tree, and the app starts with tone matching on and 96k. Not yet committed.

## 35. Breeze TTS 2 replaces Chatterbox: a batched engine (2026-10-01)

The owner asked whether newer models would help. Qwen3.8 27B for the cast LLM was dropped: its
4-bit build (18 GB) doesn't fit the 12 GB card. OmniVoice (k2-fsa, 0.6B) was weighed and set aside
for Breeze TTS 2 (BreezeBlue, 3B), which leads the open-weight models in Artificial Analysis's
blind-vote arena (1,215 Elo, #6 of 100+ overall).

### 35.1 Bake-off on this machine

Same lines, voices and book settings on both engines; listening page in the session, samples in
`data/diagnostics/bakeoff/`.
- **Words:** Whisper heard every word from both engines on the test lines, whispers and shouts.
- **Tone:** Breeze is also brighter than the clips (Adrian +4 to +9 dB from 2.9 kHz, Maya +5 to
  +12 dB above 5 kHz), so §34's tone matching applies to it too.
- **Designed voices:** from the cast profiles' own words, Breeze made Emera a child (354 Hz;
  Teen.mp3, the voice she had, is 218 Hz), Charra the huskiest of three (HNR 9.2), Marielle
  medium-low (172 Hz). Chatterbox cloned the designed clips without trouble. The owner: "the
  generated voices are better than my uploaded", and Breeze's speech "sounds sooo much better".
- **Speed, one request at a time:** 0.43x real time (eager path, 8.3 GiB). Its CUDA-graph fast path
  needs 14.4 GiB; the decode-only graphs that fit gained under 10% (0.45-0.48x). Chatterbox: 2.5x.
- **Speed, batched** through the model's own `generate()` with a list of requests, one voice:
  1 = 0.39x, 4 = 1.38x, 8 = 1.88x, 16 = 3.94x, 32 = 7.14x real time, peak 8.2 GiB. All 24 batched
  takes checked were word-perfect by Whisper. This is what made Breeze viable.
- **Not usable as a clip:** Teen.mp3. Whisper hears only "Hello. Hello. …" in it, and Breeze needs a
  clip's exact words.

### 35.2 What was built

- **`breeze/`: the server** (FastAPI, its own container, model volume `breeze-models`).
  - It starts with no model and loads on demand (~26 s) or on `POST /api/load`; `POST /api/unload`
    frees the GPU.
  - `POST /v1/batch` groups items by template and cfg, sorts them by length and runs chunks of up to
    32. Each chunk gets its own seed. Runaway length is capped at 3x the expected length (12.5 codec
    frames/s, read from the tokenizer config). An out-of-memory chunk is split in half and retried.
  - `repetition_penalty` is not passed: in `generate()` it trips a CUDA device-side assert.
  - Breeze's code is pinned at commit 58ec70c. flash-attn isn't built: inference uses eager attention.
- **App: engine `breeze`** (shown and made the default when `BREEZE_BASE_URL` is set).
  - `_speak_units` now only assembles a chapter. `_unit_takes` produces the audio: per unit for
    Chatterbox and Kokoro, unchanged, or in batches of 32 for Breeze (`_breeze_takes`).
  - A worker thread checks one batch (near-silence, then `_verdict` with Whisper, for every take,
    not only short ones) while the next batch generates.
  - Failed units are sent again together with a new seed for up to two more rounds, and the best
    take is kept, as `_speak_take` does. A unit fails the chapter only if the server never returned
    audio for it.
  - Tone matching and the peak guard apply to Breeze. No adaptive delivery yet: moods are to become
    spoken instructions (phase 4).
- **`voice_transcripts.py`:** each clip's words, transcribed once by the speech check's Whisper and
  cached by file signature.
- **`engine_gpu.py`:** the queue's readiness check never blocks. A background thread unloads
  Chatterbox and loads Breeze before a Breeze book (or the reverse), and a cast analysis now unloads
  Breeze as well as Chatterbox.
- **Retired:** Chatterbox moved to the compose profile `chatterbox` and was stopped (image kept).
  The app no longer depends on it at startup. Kokoro was commented out of `.env`.
- **Written by two Sonnet subagents** from specs, reviewed here. Their open choices, kept:
  - a unit whose every take is near-silent is kept and flagged (ranked last) rather than failing
    the chapter;
  - Breeze's unload before the LLM follows `LLM_UNLOAD_CHATTERBOX`;
  - a book waits while the Breeze server is unreachable.

### 35.3 Live evidence

- **Server, one mixed batch:** 32 Adrian and Maya units ran at 4.98x real time; a designed voice
  alone at cfg 4 ran at 0.22x. Order and format were right, with no errors.
- **A real chapter**, Goblin Stepsister Obsession Chapter 1, made by the app's own generator in cast
  mode (scratch cast):
  - 105 units in 4 batches, all passing on the first attempt (lowest Whisper match 0.86);
  - tone matching cut Maya by up to 7.4 dB from 3.6 kHz and Abigail by up to 3.6 dB;
  - an M4B in 206 s for 11.1 min of audio;
  - the same voices and unit count as Chatterbox's version of the chapter, so cast handling is
    unchanged.
- **Completeness**, whole chapter by Whisper against the book text: Breeze 94.9% of 1,658 words,
  Chatterbox 95.4%. Neither drops a run of 4+ words.
- **Pace:** Breeze reads it in 11.1 min where Chatterbox took 15.8 min, about 30% faster speech.
  The owner's ear decides whether that's too quick; the Speed slider slows a book.
- **Tests:** 772 pass (§34's 713, plus 59 for Breeze); the server's 19 pass on CPU.

### 35.4 Not verified, and what still needs Chatterbox

- **Not run yet:** a whole book through the queue (the handover, the speed estimate of 4x real
  time, retries at scale), and a cast analysis with Breeze loaded.
- **Still on Chatterbox, so broken while it is stopped:**
  - voice measuring for the automatic voice picks (it speaks a test line through Chatterbox);
  - the Voice lab's play and tune;
  - adaptive delivery.
  Those, designed voices for unmatched characters, and a usable clip for Teen are phases 3-4.
- **Long clips:** "andor request.wav" and "good morning.wav" run 25 s, so every request using them
  carries a long reference. Trimming isn't measured.

## 36. Designed voices, and the first real Breeze chapter (2026-10-01)

### 36.1 The owner's test chapter on Breeze

The owner picked the Greene Shorts chapter that gave Chatterbox the most trouble (garbled words,
random artifacts): 677 units, 49 min of audio, cast mode, through the queue.
- **Handover:** the cast analysis unloaded Breeze for the LLM. When Start was pressed, the queue
  loaded it again (136.7 s from a cold disk) and the book started.
- **Takes:** 664 passed first time, 8 on the second round, 5 on the third. One was kept with a flag
  (6:23, "Mm-hmm … Jeeesuss", a stretched interjection).
- **Whole chapter by Whisper:** 94.8% of 8,978 words. The four other 4-word gaps are Whisper's
  mishearings or the text's paragraph marks. No garbled runs, no near-silent takes.
- **Speed:** 19.5 min for 49 min of audio, 2.5x real time, against 4-5x on Goblin Chapter 1.
  Suspected cause, not yet measured: a batch pads every request to its longest voice reference, and
  this cast uses "andor request.wav" (26 s). Next step: trim references to about 10 s.

### 36.2 Designed voices (phase 3; written by a Sonnet subagent, reviewed here)

- **Who gets one:** in a Breeze cast book, a main character (profiled, with at least
  `PROFILE_MIN_LINES` lines) whose suggested voice fits poorly. That means the voice is of the
  other gender, is shared with a bigger or owner-picked character, or misses the pitch band
  (`match_cost >= OUT_OF_BAND_COST`). The narrating character and owner picks are never designed
  over, and there are at most 6 per cast.
- **What happens:** the character is marked `voice_design: pending` with `describe(character)`
  (gender, age, voice targets and the profile's voice words). The matcher's voice stays as a
  fallback.
- **When:** at book start, in the book's process, while Breeze is already loaded
  (`AudiobookGenerator._design_cast_voices`). Each design is a fixed ~10 s sample whose words are
  its transcript, checked for length, near-silence, Whisper match and a rough pitch for the gender
  (male <= 180 Hz, female >= 150 Hz, child >= 250 Hz), with up to 3 seeds.
- **Saved as:** "<Name> (designed).wav", with its transcript, gender and measurement recorded. It
  goes into both the job's cast and the saved cast, so later books reuse it. A failure keeps the
  fallback voice and is not retried.
- **Advanced:** "Design a new voice" in the cast editor (editable description), and
  "Design starter voices" in the Voice lab (24 varied descriptions).
- **Voice measuring** now speaks `MEASURE_TEXT` through Breeze when it is configured.
- **Live:** Emera (Goblin Stepsister Obsession) was designed from her profile, "A young girl with a
  high, clear voice. Youthful, playful, sharp, with a lively, expressive delivery." It passed on the
  first attempt in 42 s at 337 Hz. Breeze then cloned it for a new line, and Whisper matched it 1.00.
- **Tests:** 828 pass (§35's 772, plus 56).
- **Not verified live:** a book start with a pending design, the starter-voices run (about 15 min),
  and the pitch limits on male and low female voices.

## 37. Moods as spoken direction for Breeze; three clips archived (2026-10-01)

### 37.1 Adaptive delivery on Breeze (phase 4; written by a Sonnet subagent, reviewed here)

- A dialogue unit whose mood isn't normal goes to Breeze with an instruction and still clones the
  character's voice; normal units stay plain. The server batches directed units apart from plain
  ones (cfg 4 against 1), and they cost about twice as much.
- **Wording:** `delivery.breeze_instruction(mood, text, cue)`. Each mood has a default:
  - soft: "Say this softly and quietly, close to a whisper."
  - excited: "…loudly and with intense emotion, as if shouting."
  - emphatic: "…with emphasis and energy."
  A rule cue's verb picks a closer one: whisper, hiss, mutter or mumble, murmur, breathed,
  scream or shriek, roar or bellow, yell, shout, cried, exclaim, furiously, angrily.
- **How the verb gets there:** `segment_moods_and_cues` carries the speech-tag verb through to
  `CuedMood`, a str subclass. The cast LLM's moods have no verb and get the default.
- **No gain for Breeze:** the model sets loudness, and the peak guard still caps it. Every directed
  unit is directed whatever its length, and the clip map records the mood and instruction.
- **Live**, 7 lines, plain against directed, Maya and Michael. Whisper matched 1.00 on all 14.
  Directed loudness against plain:
  - whispered -11 dB, muttered -15 dB, hissed unchanged;
  - shouted +1 dB, screamed +10 dB, roared +18 dB;
  - a plain "!" -5 dB.
  The directed roar stretches its line from 3.4 to 7.1 s. Left to the owner's ear: how quiet the
  whispers are (about -37 dB against about -25 dB for plain speech), and the long roar.
- **Tests:** 846 pass (§36's 828, plus 18).

### 37.2 Archived clips

At the owner's request, love poem.wav, andor request.wav and Teen.mp3 moved to
`C:\Server\stacks\chatterbox\voices_archive` (out of the library, not deleted), and their
measurements and genders were dropped.
- 19 characters in 11 saved casts used them; the casts were backed up first to
  `data/cast_backups/before-voice-archive-2026-10-01`.
- Emera got `Emera (designed).wav`. Every other character got a fallback suggestion plus a pending
  design, except Andrew (Apex Prey 2), who has no profile to describe.
- Forbidden Fire 2's saved narrator voice was one of the clips.
- The owner is re-analysing every cast anyway.

### 37.3 The owner's ear: only quiet lines are directed

The owner compared the 14 takes:
- **Directed was better** for the whisper and the hiss.
- **Plain was better** for the mutter, the shout and the "!".
- **The scream and roar were weak either way,** and plain was preferred. The directed scream
  "changes the voice weird and it gets distorted", and the directed roar "sounds like he's trying to
  be a lion".

So `breeze_instruction` now directs soft speech only: whispered, hissed, murmured, breathed, under
the breath, and the soft default. Muttered and mumbled lines, and every excited and emphatic line,
are spoken plain. The tests changed to match; 846 pass.

## 38. The Docker crash, the LLM left on the GPU, and the Voice lab under Breeze (2026-10-01)

### 38.1 What crashed

At 10:30 the whole Docker VM died, one minute into a Breeze book that had started two minutes after its
cast analysis. Every container went with it, and `docker` answered HTTP 500. The VM's own log just stops.
Docker's host monitor log has the reason at 10:30:23: "Insufficient system resources exist to complete
the requested service". Windows had run out of memory. Two things fed it:
- **The cast LLM was still on the GPU.** Ollama keeps a model loaded for its keep-alive (`OLLAMA_KEEP_ALIVE=5m`)
  after the last request. qwen2.5:14b (~9 GB) was still loaded when the book loaded Breeze (~8 GB) on the
  12 GB card, with Whisper checking takes beside them. Under WSL an overfull GPU spills into Windows'
  RAM instead of failing. `engine_gpu`'s own docstring says one model at a time, but `_to_breeze` only
  unloaded Chatterbox.
- **The VM keeps Windows' RAM as file cache.** On the 31.7 GB host, `memory=24GB` let `vmmemWSL` reach
  20.7 GB while Linux used 6.7 GB. The other 17 GB was cache (model files), and Windows had 0.8 GB
  free. WSL's automatic reclaim waits for the VM to go idle, and it never does: BookBridge
  (`abs_kosync_enhanced`) keeps a steady ~20% CPU. Breeze loads of 105-121 s before the crash (30 s
  normally) show Windows was already paging.

Two earlier runs that day had the same LLM overlap and survived, so the overlap was the trigger, not
the whole cause. The Voice lab change the owner suspected (2465bab) was not involved: nothing used it
in the crashed session.

### 38.2 Fixes

- **`engine_gpu.unload_llm()`**, called by `_to_breeze` before Breeze loads (book handovers and
  standalone Voice lab requests alike). It lists Ollama's loaded models (`/api/ps`) and drops each one
  (`/api/generate` with `keep_alive: 0`). It is best effort:
  - Another LLM server answers 404 and is left alone.
  - An unreachable one is logged and Breeze loads anyway, since it holds no GPU memory to wait for.
  - Rejected: unloading at the end of the cast analysis instead. It would miss a Voice lab request
    within the keep-alive window, and the handover is where the "one model at a time" rule lives.
- **WSL** (`C:\Users\cporc\.wslconfig`, backup `.wslconfig.bak-2026-10-01`): `memory=16GB` (was 24GB),
  plus `[experimental] autoMemoryReclaim=gradual`.
  - The cap does the work: Linux evicts its own cache instead of taking it from Windows.
  - Gradual reclaim only helps when the VM goes idle. Microsoft's reference lists `dropCache` as the
    default already, and it wasn't triggering either.
  - Measured peaks for the cap: Linux's used memory reached 6.9 GB while Breeze loaded from a cold
    cache, and 7.05 GB while qwen2.5:14b loaded. Both models go into the page cache, which the cap
    can always evict.

### 38.3 The Voice lab under Breeze

The custom voice creator itself worked live (69 s while a book was generating). What failed was the
next step: the lab selects the new voice, and ▶ Play, soft/normal/excited and Save for books all post
to Chatterbox, which has been stopped since §35 ("Could not reach Chatterbox").
- **▶ Play** now goes through `play_lab`: with Breeze as the engine, it speaks the Phrase box with Breeze
  (`_breeze_sample` takes the phrase now) at the Make tab's speed.
- **Under Breeze, the delivery sliders, soft/normal/excited, Save and Reset are hidden**, because Breeze
  has no such settings. The intro says so. Under Chatterbox, nothing changed.
- **The creator's estimate** said "about 40 s". It now says "about a minute, longer while a book is
  generating": a cold Breeze load alone takes 30-50 s, and a request waits behind the book's batch.

### 38.4 Live evidence

- **Handover:** with Breeze unloaded and qwen2.5:14b warmed (GPU at 10.4 GB), `prepare_breeze()`
  unloaded the LLM at 15:23:31, and Breeze began loading at 15:23:32. `ollama ps` was then empty.
- **Voice lab:** the live config shows the sliders' row and the three Chatterbox buttons hidden.
  Through the UI's own endpoints, creating a voice took 57 s, then ▶ Play on the new voice gave 5.3 s
  of audio from Breeze in 18 s. The test voice and its records were deleted.
- **WSL:** `drop_caches` brought `vmmemWSL` from 20.7 to 9.6 GB, and Windows from 0.8 to 12.6 GB free.
  After the restart with the new settings, the VM has 16 GB, all 20 containers came back, and Windows
  had 17.6 GB free.
- **The book:** "Apex Prey: The Reaping" was stopped and deleted at the owner's request (wrong
  characters, which Codex's attribution commits address). That removed the queue job and the 5
  partial chapters in the library.
- **Not verified yet:** a whole book under the 16 GB cap (model reloads after switching may take
  30-50 s instead of 17 s), and gradual reclaim ever triggering on this always-busy VM.
- **Tests:** 869 pass, including 4 new ones here.

## 39. Breeze implementation review and plain-English voice creation (2026-10-01)

This records the review and fixes completed before the cast audit below. Findings were handled in
separate design, implementation, test and commit phases, using CodeGraph to trace the shared paths
and cheaper-model assistance where appropriate.

| Commit | Finding and implemented change |
|---|---|
| `9b643e2` | Cast reanalysis could discard Breeze voice choices. Preserve the existing choices through reanalysis. |
| `ffd380d` | Failed batch allocations could remain alive during the out-of-memory retry. Release them before splitting and retrying. |
| `7738d64` | Standalone Breeze requests could overlap the queue's GPU work. Coordinate their ownership through the shared GPU path. |
| `eb6b132` | Default Compose service discovery could leave Breeze unavailable without an explicit URL. Discover the default Breeze service. |
| `81fef98` | A pending automatic voice design could overwrite an owner's newer cast edits. Preserve those edits when saving the design result. |
| `c3d4496` | Breeze voice previews ignored the selected speed. Apply the selected speed to the preview. |

At that review checkpoint, **860 app unit tests and 20 Breeze server tests passed**. These are
historical suite counts, not a claim that the entire suite was rerun after every later change.

**Live verification:** evidence is in `data/live_verification/2026-10-01` under the stack directory.
The four-item mixed batch returned its items in order, with Whisper matches of 0.974-1.00. Preview
audio at speeds 1.0, 2.0 and 0.5 lasted 7.680, 3.837 and 15.336 seconds respectively; all three
matched the requested words at 1.00. A designed voice and its saved records were also checked.
These checks establish those paths, rather than whole-book speaker accuracy or every retry case.

The owner expected a plain-English custom voice maker in the Voice lab. **`2465bab`** added the
Name / Describe / Create flow, saved the generated voice and metadata, refreshed the selectors and
provided a preview. **270 related tests passed.** The live `/design_custom` check produced an
8.72-second preview in 56.14 seconds, Whisper match 1.00, and updated the dropdowns. The temporary
test voice was removed. `custom_maker_report.json` records the result. The separate problem with
playing that voice through the stopped Chatterbox service was subsequently fixed by Clough in
`035cc1a`, documented in §38.

## 40. Apex Prey 2: narrator switched at the chapter boundary (2026-10-01)

### 40.1 Diagnosis and owner intent

The owner heard the narrator change about 20 minutes into the finished book. BookOrbit bookmark
**66**, book **6186**, at **1,205 seconds** marks the change; the first chapter ends at 1,204.532
seconds. The model had assigned six first-person narrator quotes to the minor dermatologist.
Chapter narrator voting then chose the dermatologist's Elena voice for the first chapter and
Polly's selected Gianna voice for the remaining chapters.

The owner confirmed that Polly is the narrator and the dermatologist has only one or two lines.
Either suitable female voice chosen during evaluation was acceptable, provided the narrator stayed
consistent. **The owner explicitly requested prevention for future books, with no redo of this book.**

### 40.2 Implemented prevention and verification

**`bae3df5`** requires independent, narration-only model confirmation before a chapter switches away
from the single first-person book narrator. An unconfirmed vote uses the book narrator. Genuine
chapter POV changes can still be confirmed. Applying the resolved narrator also repairs explicit
first-person speech tags and their quote continuations, updates speaker maps and counts, and is
idempotent.

**192 related tests passed at this checkpoint.** A real independent narration-only model request
identified I / Polly. A private check using the running app's deployed correction path examined all
10 chapters and 800 narration/tagged passages: narrator routing was Gianna throughout, while the
dermatologist's actual dialogue remained separate. No audio was generated and the existing cast
was unchanged. Evidence: `data/diagnostics/apex_narrator_2026-10-01`, especially `diagnosis.json`,
`independent_narration.json` and `future_routing_report.json`.

The original finished output was 156,423,229 bytes, approximately 4.02 hours and 10 chapters. It was
not replaced. This routing check does not retroactively change the audio already being listened to.

## 41. Apex Prey 3 cast audit, failed private rerun, and Astra handoff (2026-10-01)

**Current outcome: speaker attribution remains unreliable. The owner stopped further algorithm
work and requested an independent Astra audit before authorizing more fixes.** Passing regression
tests and keeping the chapter narrator consistent did not make this cast correct. This section
supersedes the earlier pending-validation status in the diagnostic `review.md` and any implication
in §38 that the attribution commits fully resolved the book's cast problems.

### 41.1 Original cast and text audit

The test book is *Apex Prey: The Reaping* (Apex Prey Trilogy, Book 3), cast key
**`8c7f3e6ca0040c1a`**. Its nine selected EPUB documents, numbered **7-15**, correspond to story
chapters **1-9**, with **269 dialogue lines**. The original analysis kept Polly / Emily as narrator
in all nine chapters. Voice-feature comparisons matched the saved gender and pitch targets;
Emily had the lowest measured matching cost for Polly's target. This was a metadata comparison,
not a fresh listening test, and did not establish that the assigned speakers were correct.

The first manual audit identified 14 wrong assignments:

| EPUB document / story chapter | Dialogue lines | Original assignment | Text-based correction |
|---|---|---|---|
| 9 / 3 | 2, 4 | Polly | Unnamed girl with needle phobia |
| 9 / 3 | 7 | Jimmy | Unnamed male group patient |
| 12 / 6 | 18 | Jimmy | Gus, Jimmy's companion, who addresses Jimmy and later gives his own name |
| 14 / 8 | 6, 10, 11, 15, 20, 25, 26 | Andrew | Polly |
| 14 / 8 | 18, 19 | Andrew | Alan |
| 14 / 8 | 23 | Alan | Polly, explicitly tagged "I read aloud" |

Andrew's throat had already been severed; the scene explicitly says he cannot shout. Jimmy and
"Jimmy's companion" had also collapsed into one roster entry. Nine unknown lines in EPUB document
11 were Polly's, whose narrator fallback already used Emily correctly. The displayed unknown count
was stale: 16 reported versus 9 remaining after earlier correction.

**Reference correction discovered later:** EPUB document 9, dialogue line 24 is Stuart answering
Polly's preceding question. The original reference incorrectly said Polly. `expected_speakers.json`
and the rerun comparison were corrected; the historical original report still records the first
14 findings. The manual reference needs independent review and must not be treated as infallible.

### 41.2 Four completed fix phases

| Commit | Implemented change and intended effect |
|---|---|
| `b11699b` | Keep possessive relationship names separate from their owners, including both registration orders, curly/ASCII apostrophes, supplied aliases and saved-roster restoration. Prevent Jimmy from becoming Jimmy's companion. |
| `c04a520` | Recognize missing first-person speech verbs such as respond, read, finish and sneer; carry the narrator through same-paragraph quote continuations until another speaker or introduction intervenes. |
| `cb8f7e6` | Resolve narrator attributions before building voice profiles, and recompute chapter/book unknown counts from the corrected assignments. Avoid profiling the wrong speaker's lines. |
| `fcea8c4` | Strengthen attribution and review prompts around who is present and able to speak, addressed/mentioned names, and consistently identified unnamed speakers. |

**200 relevant local tests passed** across cast analysis, LLM attribution, speech tags, profiles,
review and storage. The profile-order test runs the analysis on a generated EPUB with controlled
model replies; it verifies ordering and counts, not live-model accuracy. Applying only deterministic
corrections to a private copy fixed three of the original 14 errors and three additional unknown
Polly lines, with no unexpected assignment changes. The other 11 needed real-model validation.

Docker failed during this work; the owner assigned recovery to Clough. The cast investigation
continued offline without taking over recovery. Clough's `035cc1a` GPU/Voice lab fixes are separate
from these four cast changes. After recovery, runtime hashes confirmed the cast modules matched
the committed local versions before the private rerun.

### 41.3 Actual private rerun: still failed

A fresh analysis through `analyse_book` used **qwen2.5:14b**, without carrying the saved cast into
the new analysis or writing results to the book. It made **42 model requests in 190.28 seconds**:
20 base attribution windows, six review requests (26 lines reviewed, 25 changed), plus narrator/tone
and 13 profile requests. Measured voice suggestions were then added to the private result. This
tested analysis and suggestion components; it did not enqueue a book, run pending voice designs,
or regenerate audio through the UI. The harness checked that the queue was idle and unloaded the
LLM at the end.

Polly / Emily remained the narrator for all nine chapters; the final unknown count was zero.
Andrew was no longer assigned dialogue, which is an improvement. Nevertheless:

- The needle-phobic girl's lines were still assigned to Polly.
- Polly's dialogue was split into an "unnamed female" entry, aliased as "the narrator", with Layla
  suggested. Fourteen of that entry's 15 lines were Polly's in the reference; one was Alan's.
- Gemma was split into Gemma / Lucy and "young woman" / Gianna, giving the same character two voices.
- "One of the guys" represented both an unnamed group patient and Gus in different scenes.
- Opening Jimmy/Jose assignments changed. These need an independent reading of the surrounding
  narration; they are comparison leads rather than unquestionable ground truth.

The current comparison records **241/269 reference matches and 28 speaker/identity discrepancies**:
five of the original 14 findings fixed, nine remaining, plus 19 new discrepancies. These numbers
include identity splits and depend on a corrected but fallible manual reference. **They are not
a validated accuracy score.** Zero unknowns likewise does not imply correct attribution.

The original saved cast `data/casts/8c7f3e6ca0040c1a.json` was unchanged, reconfirmed while writing
this entry with SHA-256 `01dfe6ecdc990aa0933dfe3aaae7f0870410db0785afc78dd2964bc6bd61971c`.
The original queue snapshot `upload_xqmwp1pu.json` was removed by the separate queue/recovery
cleanup described in §38; this audit did not edit or delete it. Do not rely on the older diagnostic
report's statement that that snapshot still exists. Apex Prey 2 was not redone.

### 41.4 Unshipped prompt experiments

Two private request replays produced mixed results and were **not implemented or committed**:

- `test_grounding.py` / `grounding_experiment.json` tried narrator aliases and evidence after the
  speaker list. Its broad string replacement also changed occurrences of Polly inside quoted book
  text, invalidating it as a production-equivalent experiment. Some identity improvements came
  with malformed names and persistent wrong assignments.
- `test_evidence_first.py` / `evidence_first_experiment.json` asked for evidence before speakers.
  Some girl/Gemma assignments improved, but other speakers became wrong or unknown and the
  "unnamed female" split remained. This did not establish a reliable fix.

No further algorithm code changes followed the failed rerun. No larger-book test was performed.

### 41.5 Evidence and next authorized action

All diagnostic paths above are relative to **`C:\Server\stacks\epub-to-audiobook`**, outside the
`src` Git repository. The Apex Prey 3 evidence directory is:
`C:\Server\stacks\epub-to-audiobook\data\diagnostics\apex3_2026-10-01`.

- Original audit: `review.md`, `verification.json`, `roster.json`, `audit_input_summary.json`,
  `voice_fit.json`, and `chapter_7.txt` through `chapter_15.txt`. The chapter annotations are
  original predictions, not ground truth; `review.md` has historical live-validation status.
- Deterministic-only preview: `offline_corrected_preview.json`; it is not a fully repaired cast.
- Revised reference: `expected_speakers.json`, including the Stuart correction above.
- Actual rerun: `reanalysis_1.json`, `reanalysis_1_with_voices.json`,
  `reanalysis_1_comparison.json`, `reanalysis_1_requests.json`, `reanalysis_1.log`, and
  `rerun_analysis.py`. Requests/replies allow attribution and review decisions to be inspected.
- Unshipped experiments: the two scripts and result files named in §41.4.

**Next action is an independent, read-only Astra audit.** Its prompt must require reading this
worklog first, then tracing the full cast-to-voice flow with CodeGraph and checking predictions
against the book text. Return ranked, evidenced findings and a minimal phased implementation/test
plan, including which existing changes to retain, revise or revert. Do not edit code, deploy,
change casts or queues, regenerate audio, or launch more model experiments during that audit.
The owner will provide the fixes to implement afterward. Algorithm work remains stopped.

## 42. Astra audit implemented: identity fixes and an offline Apex Prey 3 replay (2026-10-01)

The Astra audit (§41.5) returned seven ranked findings and a phased plan. The owner authorized the
plan. Codex implemented the first two phases and part of the rest, then ran out of budget; Claude
reviewed that work, had cheaper-model agents finish it, and checked each result.

### 42.1 Commits

| Commit | Phase | Change |
|---|---|---|
| `621a9b3` | 0 | Offline scorer against a source-linked 269-line Apex Prey 3 reference (kept outside the repository, see below) (`cast_audit_eval.py`) reporting wrong speakers, splits, false merges, unresolved lines and voice routes separately. |
| `b049841` | 1 | "I", "narrator" and "the narrator" mean one person per chapter (an anthology's next "I" is someone else); the resolved narrator reaches attribution and review. |
| `42feb41` | 2-4 | Descriptions are chapter-local identities; declared description aliases bind; answers to "what's your name?" and short "I'm X" introduce the speaker; review answers that the line's own tag contradicts are rejected. |
| `d8e410b` | 1, 4-6 | One unnamed narrator per first-person run; final reconciliation before profiles (pronoun-contradicted lines go to no speaker and are listed); advisory issues; profiles rebuilt after identity changes; voice picks carry by real names only; embedded documents and backmatter excluded from narrator evidence. |
| `5c7a44c` | tooling | Private production-path runner with exact request logging; dry voice-routing regression test. |

### 42.2 What the review changed in Codex's unfinished work, and why

- **Name matching had been switched off entirely** ("Tom" and "Thomas Baker" became two people, 7
  tests failing). The audit called it a latent risk, not a defect; it was restored and only kept
  away from descriptions.
- **A never-named narrator got a new identity in every chapter**: in a first-person book whose "I"
  is never named, that meant a voice change per chapter (the Apex Prey 2 bug class). One unnamed
  narrator now serves a whole first-person run and becomes the book's POV character.
- **Codex's readiness gate blocked automatic voices and queueing on any unknown line or unclosed
  "I open the letter".** Nearly every book has unknown lines, and nothing in the UI could clear an
  issue, so cast mode would have stopped being automatic. Replaced by repair-and-report: a
  contradicted line is not accepted (it is read in the dialogue voice) and is listed in the summary.
- **Unclosed document frames** excluded the rest of the chapter from narrator evidence; now only a
  frame opened and closed within 8 paragraphs is excluded.
- **Direction of self-introductions**: the replay below showed the narrator's own "Call me ..." line
  (tagged "I say") folding the named narrator into the anonymous "The Narrator". A description or
  unnamed "I" who gives a name now becomes that named person.

### 42.3 Offline replay on Apex Prey 3

No model was called. `data/diagnostics/apex3_2026-10-01/replay/replay_apex3.py` runs the current
production `analyse_book` on the real EPUB and answers every attribution and review request with the
answer qwen2.5:14b gave for the same lines in the failed rerun (§41.3). Run against the rerun's own
code (`32631d1`), it reproduced that rerun exactly: same chapter hashes, same 26 requests, same
scorer output. So the comparison holds the model's answers fixed and isolates the code.

| Scorer result (reference of §41.1, corrected) | Rerun code | Current code |
|---|---|---|
| Lines whose speaker label differs | 29 | 6 |
| Narrator identities | Polly + "unnamed female" | Polly only, all 9 chapters |
| Gemma | 2 characters (Gianna / Lucy) | 1 |
| Hospital patient vs Jimmy's companion | one character | two |
| Alan's "he cries out" line | female speaker | no speaker, listed as an issue |

The 6 remaining: the opening Jimmy lines given to Jose and the needle-phobic girl's two lines given
to Polly are model errors no code change here touches; the hospital patient's line is the right
person under another description; Jimmy's companion joins Gus only if the model answers the new
identity question ("same as one of the guys?"), which the replay can only answer "new". Pair counts
(splits 1977 → 18, false merges 297 → 310) move with those same lines.

### 42.4 Tests and what is not verified

- 915 app tests pass in the container (full discovery, including the UI modules).
- The owner then authorized the live rerun, the history rewrite and the deploy (§42.5).
- The reference contains the book's text, so it is not in the repository: Codex's original
  commit (which added it under `tests/fixtures`) was rewritten before any push. The reference is at
  `data/diagnostics/apex3_2026-10-01/expected_speakers_source_linked.json`; `cast_audit_eval.py`
  reads it from there by default (or `--reference` / `CAST_AUDIT_REFERENCE`), and its unit tests
  use an invented reference. The pre-rewrite history is kept on the local branch
  `backup/before-fixture-move`, which must never be pushed.

### 42.5 Live private reruns with qwen2.5:14b

Four private reruns of the 9 selected chapters ran through `evaluate_cast.py` in a throwaway
container (`reanalysis_2` to `_5` beside the earlier evidence, each with `run.py`, `cast.json`, the
exact `requests.jsonl`, `analysis.log` and the scorer's `eval.json`). Each refused to start with a
queued job, unloaded Breeze first and the LLM afterwards, and wrote nothing outside its folder.

| Run | Code | Wrong speakers (real) | Narrator |
|---|---|---|---|
| failed rerun (§41.3) | `32631d1` | 28 discrepancies | Polly + "unnamed female" (Layla voice) |
| 2 | `b537a78` | 7, Alan's line unassigned | Polly, 9 chapters |
| 3 | 2 + "I said" lines not pinned when the narrator is unnamed | 12 | 5 Polly lines to Andrew |
| 4 | 2 + "sputters" | narrator split | Polly ch 1-2, "The Narrator" ch 3-9 |
| 5 | `73b9afb` (deployed) | 7, Alan's line unassigned | Polly, 9 chapters |

"Real" ignores label-only differences ("the girl" for the needle-phobic girl, "the realtor" for the
male realtor, "one of the guys" for the patient), which are the right people.

- **Run 3** tried showing the model unnamed-narrator "I said" lines instead of pinning them as `[I]`
  (in run 2 the bare `[I]` seemed to cut the link between "I" and Polly in the first chapter). It
  fixed that chapter but sent five of Polly's lines and two of Alan's to Andrew, who cannot speak;
  reverted. With this model, prompt-level changes move errors around rather than remove them.
- **Run 4 found a real bug**, not model noise: the book-tone question was offered the roster's
  "The Narrator" placeholder and picked it over Polly; chapters 3-9, whose "I" the model never named,
  then kept that placeholder as their narrator, so the narrator voice would have changed at chapter
  3 (the Apex Prey 2 symptom). `73b9afb` never offers or accepts the placeholder there, and an
  unnamed "I" chapter takes the nearest named narrator of its first-person run.
- **Run 5 (deployed code)**: one narrator, Polly, in every chapter; Gemma one character; the
  needle-phobic girl separate from Polly; the patient and Jimmy's companion separate, and the
  companion joined to Gus by the new identity question; "Stuart sputters" now settles that line;
  Alan's "he cries out" line unassigned and listed. Still wrong: D7 lines 1, 6, 16 (the opening
  Jimmy/Jose scene and the doctor's "Well, Polly, ..."), D9 line 6 and D11 line 22 (Polly's lines to
  Stuart and the realtor), D12 line 15 (Jimmy's line under Gus), D15 line 7 (Gemma's "Absolutely!"
  to Polly). The text settles every one of them; these are the model's errors.
- **Release gate**: not fully met. The audit asked for no remaining errors in the known scenes, and
  D7 line 1 is still wrong. The owner chose to deploy because every identity failure the audit found
  is fixed with the real model and narrator continuity held in every run but the one whose bug is
  now fixed. Before a larger book: the remaining errors need a better prompt or model, judged with
  this scorer over repeated runs, not single runs.

### 42.6 History rewrite

Codex's commit that added the reference under `tests/fixtures` was rewritten before any push:
`621a9b3` replaces it with the scorer and invented-reference tests, and the later commits were
replayed unchanged (new hashes in §42.1). The tree differs from the pre-rewrite history only by the
fixture and the scorer's tests. The old history is on the local branch `backup/before-fixture-move`
(like `backup/before-scrub`, never to be pushed).

- Tests after `73b9afb`: 242 host cast tests pass; the full container run is in §42.7.

### 42.7 Deploy

- 916 app tests pass in the container on `73b9afb` (1 skipped: the private reference check, whose
  file lives outside the repository). 242 host cast tests pass.
- With no job queued or running, `docker compose up -d --build epub-to-audiobook` recreated the app.
  The container's `/app_src` copies of the seven changed modules match the commit, the page answers
  200 and the log shows no errors. The owner's saved Apex Prey 3 cast (made before these changes)
  loads and summarises unchanged; no cast, queue or audio was changed. Breeze was left unloaded by
  the reruns and loads again when a book starts.
- README: a line whose tag contradicts its speaker reads in the dialogue voice and is listed.

## 43. A larger cast: Six Wakes test chapters (2026-10-01)

The owner chose *Six Wakes* (Mur Lafferty) to tune attribution on a bigger cast: six crew members,
the ship's AI, clones, and per-character flashback chapters, all in the third person.

### 43.1 Test set and reference

- **Chapters** (parser numbers): 5, 17, 21 and 28; 512 dialogue lines. Chosen for a crowded waking
  scene, a 7-speaker scene, a flashback with few tags, and long untagged two-person exchanges.
  Evidence: `data/diagnostics/six_wakes_2026-10-01` (chapter exports without predictions, the
  reference, every run).
- **Reference**: two Sonnet labelers worked independently from the text and agreed on all 512 lines.
  Agreement was not taken as proof: Claude read every line where the local model disagreed, and the
  labelers were both wrong on ch5 lines 31-32 (the speaker asks "Do you remember anything?" and
  Joanna answers "No": Maria speaks). Excluded from the score: 7 terms quoted inside narration
  (not speech) and 1 genuinely ambiguous line, leaving 504 scored lines.

### 43.2 Findings and fixes (`2cdd6ce`)

The baseline (deployed code) had 35 wrong lines, and 31 of them were identity splits, not model
mistakes: one man as "Hiro", "Akihiro Sato", "Akihiro Sato (the clone)" and "Akihiro Sato, Ninth of
the line"; a detective as "Detective Natalie Lo" and "Lo". In audio, up to three voices per person.

- A note the model adds after a name, in brackets or after a comma, is dropped.
- A nickname of at least 4 letters that ends a first name ("Hiro" / "Akihiro") is a candidate for the
  same person, under the existing one-candidate and gender checks.
- A bare surname joins the one character with that last name; a titled form ("Mrs. Marsh") or a
  shared surname still does not.
- A term quoted mid-sentence in narration is not speech: never asked, no speaker, narrator's voice.
  Over the whole book this flags 28 lines, all terms or reported speech the narrator voice suits;
  none in Apex Prey 3.

### 43.3 Live results (qwen2.5:14b)

| Run | Six Wakes, 504 lines | Apex Prey 3 (held out) |
|---|---|---|
| Baseline (`00d08b7`) | 35 wrong | 7 wrong + 1 unassigned (§42.5) |
| Identity + quoted-term fixes | 7 wrong | identical to §42.5 |
| + comma qualifiers (`2cdd6ce`) | **2 wrong** (ch28 lines 24-25 swapped) | identical to §42.5 |

Each Six Wakes character now has one identity. Three model errors in the baseline (ch17 lines 71
and 90, ch21 line 87) came out right after the fixes, but only because the prompts changed when the
quoted terms stopped being asked; they are not claimed as fixed.

### 43.4 What it means for prompt tuning

These four chapters are now near the ceiling (2 wrong), so they cannot show whether a prompt change
helps. Prompt tuning needs harder chapters: the low-tag ones are 29 ("Wolfgang's Story", 42 lines,
1 tagged), 30 ("Breakdowns", 71/8), 20 ("Yadokari", 62/15) and 26 ("Paul's Story", 71/16).
Apex Prey 3's remaining 7 errors stay the held-out check.

- Tests: 926 app tests pass in the container (1 skipped: private reference check).
- Not deployed yet.

## 44. Library scan and a story collection: one narrator per story (2026-10-01)

### 44.1 Library scan

`data/diagnostics/library_scan_2026-10-01/scan.py` parses all 1,518 library EPUBs as the analysis does
(read-only; 4 could not be parsed) and records dialogue lines, tagged share, first-person tags and
recurring speakers per book and chapter (`scan.jsonl`). It ranked candidate test books by difficulty:
big third-person casts with few tags (The Stand, Imajica), first-person books with big casts (Pierce
Brown), and collections or multi-POV books (You Like It Darker, Hyperion). The owner chose *You Like
It Darker* to test different first-person narrators side by side.

### 44.2 Test set and reference

Chapters 12 ("Red Screen", third person), 15-16 ("Rattlesnakes", first person, Vic Trenton) and 17
("The Dreamers", first person, William Davis): 923 lines. Two independent Sonnet labelers agreed on
919; three differences were spelling only, and Claude left ch17 line 172 out as ambiguous. Excluded
from scoring: 12 quoted titles and terms that are not speech. 910 lines scored, each chapter's
narrator recorded. Scored with `score.py` there, which maps the model's other names for the same
people ("Allie Bell" for Alita Bell, "Officer Zane" for Preston Zane) so only real errors count.

### 44.3 What failed and the fix (`21c85fc`)

The deployed code made Allie Bell, a woman in the story, narrator of chapter 15 (the book-tone guess,
used because the model had left that chapter's "I" unnamed), and gave chapter 17 the previous story's
narrator Vic: the model had answered "Vic Trenton" for The Dreamers' "I said" lines.

- **Tried and reverted:** asking the model about each unnamed-"I" chapter with only that chapter's
  people listed. On Apex Prey 3 it named whoever it was offered (Gemma, Alan) as narrator and the same
  answer "confirmed" itself: Polly's narration would have changed voice twice. Asking again is the
  fragile step.
- **Kept:** a deterministic "same story" test -- a chapter belongs to a narrator's story when,
  besides the narrator, it shares someone with that narrator's chapters (speakers or names in the
  text). An unnamed-"I" chapter takes the nearest named narrator that passes it. A chapter "narrated"
  by someone with at least 3 other named people, none shared, and the narrator's name nowhere in it
  gets its own unnamed narrator. Apex Prey 3's chapters all share people (Jimmy, Jose, Stuart,
  Alan), so its narrator stays one.

### 44.4 Results (live, qwen2.5:14b)

| Run | Ch 15 | Ch 16 | Ch 17 | Wrong of 910 | Apex Prey 3 |
|---|---|---|---|---|---|
| Baseline (deployed `2cdd6ce`) | Allie Bell | Vic | Vic | 237 | as §42.5 |
| Unnamed-"I" chapters asked again | Vic | Allie Bell | Vic | 191 | narrator split (Gemma, Alan) |
| Story-aware fold | Vic | Vic | Vic | 169 | as §42.5 |
| + story-break split (`21c85fc`) | Vic | Vic | own narrator | 93 | as §42.5 |

The remaining 93 are model errors, led by 24 of Andy Pelley's lines given to Vic when Pelley says
"..., Vic?" (an addressed name taken for the speaker), and Elgin's lines the model gave to "Vic" in
ch17, which now sit with ch17's narrator. That error class is the next prompt-tuning target. The
model never learns that ch17's narrator is William Davis, so his voice is chosen without a name.

- Tests: 928 app tests pass in the container. Not deployed yet.

## 45. The remaining errors: what code could fix, and a narrator risk (2026-10-01)

The 103 wrong lines left across the three test sets (You Like It Darker 93, Apex Prey 3 8, Six Wakes 2)
were sorted by cause from the text and the model's raw replies. Evidence:
`data/diagnostics/*/runs/{phaseA,phaseAB,w16_*}`.

### 45.1 Kept

- **`7545c2c`, "I told her..." is a reply.** A paragraph opening with a first-person speech verb after
  someone's quotation had handed that quotation to the narrator. Over the three answer keys this was
  right 2 times and wrong 12, so the rule is gone; '"...," I said, and he repeated "..."' no longer
  gives "..." to the narrator either.
- **`44792d3`, turn-taking.** Runs of one-quotation paragraphs between two people, anchored by a
  tagged line, are reassigned by strict alternation when the model's answers break it. Measured
  offline on two saved runs of each set: 5 and 8 lines fixed, none broken (one early version broke
  two lines because a next paragraph that opens with a beat was taken as the next turn; fixed).

### 45.2 Measured and not done

- **A name in the line is not its speaker.** About 1 line in 10 that names someone is spoken by them
  (self-introductions: "Polly," to "what's your name?", "Vic, please"), so it is used only to block
  turn-taking changes, never as a rule of its own.
- **The narrator's "twin."** In Pelley's interview the model used both "The Narrator" and "Vic Trenton";
  of 70 lines under "Vic Trenton" 51 really are Vic's, so the two labels cannot be told apart.
- **Long untagged interviews** (Pelley, Elgin) have no tagged line to anchor turn-taking and King
  breaks strict alternation often; they remain the model's.

### 45.3 Results and run-to-run spread

| You Like It Darker run | Wrong of 910 |
|---|---|
| Before (`21c85fc` code, §44) | 93 (88 with turn-taking offline) |
| Phase A live | 108; phases A+B live 100 (= offline estimate) |

Removing a rule changes what the model sees, and the model's errors move: phase A fixed 25 lines and
broke 40 elsewhere (Pelley's interview collapsed in another stretch). Single paired runs differ by
+/-15 lines for this reason, so a change must beat that margin on all three sets to count. Apex Prey 3
and Six Wakes were unchanged (8, 2).

### 45.4 Narrator risk found (not fixed)

Two further runs with 16 lines per request instead of 20 (a deliberate perturbation, both rule
variants) made **Alita Bell, a woman in Rattlesnakes, narrator of both its chapters**: 292 and 382 of
910 wrong. When no chapter of a first-person story gets a named "I said" vote, the book-tone guess
still decides, and it can name someone the narrator talks to. It happened in 3 of the 6 runs of this
book (the baseline's chapter 15 too). Apex Prey 3 has not shown it because the model names Polly.
Checked and rejected as fixes: "she converses with the I" (Polly does too, as the narrator continues
into the next paragraph) and "most addressed person" per chapter (Polly addresses her victims). Summed
over a story the most-addressed person points the right way (Vic 36, Polly 9 vs Jimmy 5) but with
thin margins; this needs its own design and measurement before anything ships.

- Tests: 933 app tests pass in the container. Not deployed.

## 46. Narrator choice: nobody the narration names, and an unnamed "I" stays unnamed (2026-10-01)

Phases A+B (§45.1) were deployed first; the container's code matched `302f075`. Evidence for this
section, outside git: `data/diagnostics/narrator_choice_2026-10-01` (tools and replayed casts).

### 46.1 What went wrong in §45.4

The failed runs' logs show the book-wide guess (one model answer from narration excerpts) overruling
chapter votes, not only filling gaps:

- An unnamed "I said" vote can never be confirmed by the chapter-only question ("The Narrator" is
  never offered as an answer), so in a single first-person run it was always replaced by the guess:
  Alita Bell for chapters 15, 16 and 17 of one window-16 run.
- A named vote (Vic, chapter 16 of the other) was overruled when the chapter-only answer differed.

A chapter told by anyone but the book's own "I" is read in that teller's own voice, so her voice read
part of his story.

### 46.2 The rule: nobody the narration names is its "I" (`could_say_i`)

First-person narration calls its teller "I". Over 153 first-person chapters of eight books (the test
sets and five of the owner's saved casts) real narrators were named in 0-3 narration paragraphs of
their chapter, 125 chapters in none. The 18 wrong narrators found in saved casts were named in 9-54:
Alita Bell, and "Ruby" in Goblin Stepsister Obsession, a stepsister the narration talks about (checked
in the text: the narration describes her to "me"). Named in more than 4, a person is not that
chapter's narrator, whether the "I said" vote, a neighbour's, the pooled vote, the book-wide guess or
the chapter-only answer proposes them. Message labels ("Name: hey"; a texting chapter wrote its
narrator's name 12 times that way) and framed documents don't count. Checked and rejected first:
speech tags alone ("Allie said") named her once in chapter 15 and never in 16.

### 46.3 Option 1: an unnamed "I" stays unnamed

A story whose "I said" lines only ever went to "The Narrator" keeps one unnamed narrator: the
book-wide guess no longer names it, and its untagged chapters join it. A story the model names
anywhere still folds its unnamed chapters into that name (§42). The cost: the narrator's lines the
model gave to their name (when addressed, say) keep that name's voice. The Apex Prey 2 safeguard (§40:
a named vote for the dermatologist overruled by the book narrator after the chapter-only question)
stands, unless the narration names the book narrator.

Unnamed tellers also split at story breaks, on the signs `_split_story_breaks` uses for named ones
(at least 3 other named people, none shared). Without it a replayed run gave The Dreamers the same
unnamed narrator as Rattlesnakes.

### 46.4 Measured

Exact replay: the pipeline runs on each saved run's requests, answered with the model's saved
replies (HEAD reproduces all ten runs exactly; older runs used older code and can't be replayed).
The scorer judges an unnamed narrator as whoever most of its reference lines belong to: one voice,
one person.

| Run (wrong of the reference lines) | HEAD | New |
|---|---|---|
| You Like It Darker, window 16, rule on | 292 (Alita Bell narrates 15-16, Vic 17) | 104 (Vic, Vic, unnamed 17) |
| You Like It Darker, window 16, rule off | 294 (Alita Bell narrates 15-16) | 100 (unnamed 15-16, its own unnamed 17) |
| You Like It Darker phaseAB, phaseA | 100, 100 | 100, 100 |
| Apex Prey 3 phaseAB, phaseA | 8, 8 (Polly throughout) | 8, 8 |
| Six Wakes fix2, fix3, phaseA, phaseAB | 2 each | 2 each |

The owner's eight saved first-person casts, rerun offline with a stand-in model that confirms every
chapter vote (the worst case): the old code makes Ruby narrator of 14 Goblin chapters, the new code
keeps Rakos; nothing else changes (Apex Prey 2's dermatologist chapter, the Greene collections' five
tellers, Troy, Oliver).

One live run of the You Like It Darker chapters with the new code: 100 wrong, as before; narrators
Vic, Vic and The Dreamers' own unnamed narrator (profiled male). The model named Vic in that run, so
it shows an ordinary run is unchanged; the replays above show the fix.

### 46.5 Option 2 measured, not shipped

Per story (first-person runs split at story breaks), the person others address most by name among
those the narration doesn't name, needing 3 addresses and twice the next person: 9 stories right, 4
abstained, none wrong; by chapter 35 right, 66 abstained, 1 wrong. Abstentions: one person split into
two cast keys (Goblin "Onii-chan" 79 / Rakos 53, Mom's Guidance Troy 55 / "Troy" 47), The Dreamers
(William Davis is never addressed) and two merged Greene stories. The wrong chapter: in a cast where
the model had already given The Dreamers' "I said" lines to Vic, the story break could not be seen
and Vic was named. Looser thresholds (2 addresses, 1.5x) start naming wrong people (Robin for a Greene
story). It would name an unnamed story's narrator (Rattlesnakes: Vic) and could replace the
book-wide guess as the safeguard's reference; not done until the owner decides.

- Tests: 936 app tests pass in the container. Not deployed.

## 47. Option 2: the person the others call by name tells the story (2026-10-01)

§46 (`5d57315`) was deployed first; the container's code matched the commit.

### 47.1 What it does (`addressed_tellers`)

A run of first-person chapters is split into stories where a chapter's text names at least 3 people
and none the story so far named. Within a story, the person others address by name most often
("..., Vic?") among those the narration never names (`could_say_i`) is its "I", if addressed at least
3 times and twice as often as anyone else. That person:

- names a story whose "I said" lines only ever went to "The Narrator" (§46.3 left it unnamed);
- stands in for the book-wide guess as the reference that overrules an unconfirmed chapter vote (the
  Apex Prey 2 safeguard, §40) and as the last resort for an untagged chapter. The guess is used only
  where a story has no teller.

The story split counts only the people a chapter's text names. Counting who the model said speaks,
as `_same_story` does, joined The Dreamers to Rattlesnakes in every replay: lines of The Dreamers had
gone to Vic. The address check now compiles one pattern per person (`_address_pattern`).

### 47.2 Measured

- **Per chapter on nine casts** (the test sets and five of the owner's books): 35 chapters named
  right, 67 left alone, none wrong. Left alone: two books where one person is split into two cast
  entries and both are addressed (79/53 and 55/47), The Dreamers (its teller is never addressed by
  name), and two collection stories the split could not tell apart. The measurement in §46.5 had one
  wrong chapter (The Dreamers named Vic); the text-only split removed it. 1.4 s for a 55-chapter book.
- **Exact replays, ten runs:** scores unchanged (You Like It Darker 100, 100, 104, 100; Apex Prey 3
  8, 8; Six Wakes 2 x4). In the window-16 run whose story stayed unnamed under §46, Rattlesnakes is now
  told by Vic, not "The Narrator". Logged: "chapters 15-16 told by vic trenton, addressed by name 27
  times (next 1)"; Apex Prey 3: Polly 6 (next 0).
- **The owner's saved casts:** no narrator changes against §46, with or without a model; their votes
  already agree with the teller.

- **One live run** of the You Like It Darker chapters: 100 wrong, as before; logged "chapters 15-16 told
  by vic trenton, addressed by name 27 times (next 1)"; narrators Vic, Vic and The Dreamers' own
  unnamed narrator, all profiled male.

### 47.3 Fixed before commit

A run whose only "I said" votes were unnamed, where one story got a teller and a later story had no
vote and no teller, raised `min()` of an empty list while lending an unnamed narrator; that chapter now
takes the usual path. Covered by a test.

- Tests: 940 app tests pass in the container. Not deployed.

## 48. Breeze slowdown guard (2026-10-02)

### 48.1 What happened

The owner expected the overnight queue to finish by morning. One book, "Whores Versus Sex Robots"
(2026-10-01 18:13-23:46 EDT, 5.3 h of audio), took 5.6 h, about 3.5 h more than Breeze's normal speed
allows. From the Breeze log (full chunks = 24-32 sentences):

- Healthy books, before and after it (488 full chunks): median 3.11x real time, 5% under 1.77x.
- That book (112 full chunks): median 1.21x, 75% under 1.43x. 32 of its 86 chunks of 32 took over 2 min
  each (up to 640 s; normal is about 40 s), 150 min in all.
- Its first five chunks ran at normal speed; then everything slowed and stayed slow until the book
  ended and the app unloaded Breeze for the next cast analyses. Trad Wife, the next book after a fresh
  load, ran at normal speed.

Ruled out: the text (1.4% of units rejected and retried, against 1.2-1.8% for the next three books);
Ollama (no requests during the book, and qwen2.5:14b had expired 6 min before Breeze loaded); Whisper
(runs on the CPU). Not found: the cause. Docker's host monitor log had already rotated past that night.
The GPU stood at 11.9 of 12.3 GB used during this morning's book, so GPU memory spilling into shared
system memory under WSL fits the pattern but is unproven. Neither the RAM work (§38) nor the cast work
(§40-§47) changed Breeze's speed: full chunks ran at 3.3-3.7x before both and 2.7-3.7x after.

### 48.2 The guard (`breeze/server.py` `SlowdownGuard`)

The server times every chunk already. A chunk of at least 3/4 of `BREEZE_MAX_BATCH` counts as full;
small chunks run under real time by nature (1 sentence 0.2-0.4x). When the median of the last 4 full
chunks is under 1.5x (`BREEZE_SLOW_RTF`, 0 = off), the server unloads and loads the model at the start
of its next request, logging GPU memory before and after so the next slowdown shows whether memory was
the cause. A reload that doesn't help logs `still slow`, and the next reload waits 30 min. An unload
from the app drops a pending reload; a fresh load starts a fresh window.

Rule chosen by replaying the Breeze log: "median of 4 under 1.5x" never fired in a healthy book (longest
healthy run of full chunks under 1.6x: 2) and fired 24 min into the slow book. "3 in a row under 1.5x"
fired an hour in; "median of 5 under 1.8x" fired once in a healthy book. Replaying the committed
`SlowdownGuard` itself over the whole log: 0 reloads in healthy books, 8 in the slow one (first at
22:37 UTC), at most ~5 min of reload time if reloading does nothing.

Rejected: reloading from the app between chapters (the app would need the server's per-chunk timings,
and a chapter is several requests, so the server reacts sooner); a speed baseline learned after each
load (it would have worked here, since the first five chunks were normal, but a load into an
already-slow GPU would learn the slow speed; 1.5x is measured on this GPU and set by an env var).

### 48.3 Not yet known

Whether a reload restores the speed: the only evidence is that a fresh load 45 min later (with cast
analyses in between) ran normally. The next `slowed down` line in `docker logs breeze` answers it.

- Tests: 26 Breeze server tests pass (6 new). App untouched.
- Deployed 2026-10-02 08:10 EDT with the queue paused between books (Breeze container only;
  `/opt/breeze-infer/server.py` matches the commit). Live: one Adrian sentence, 2.6 s, after a 41.7 s
  load; then unloaded. Under WSL, `torch.cuda.mem_get_info` from a second process showed 11.1 of 12.3
  GB free with the model loaded, so the logged "free" may understate use; the reserved/allocated
  figures are the server's own. The Windows counter `\GPU Process Memory(*)\Shared Usage` shows the
  spill into system memory (1.6 GB during a normal book on 2026-10-02).

## 49. Review follow-ups (2026-10-02)

§47 (`47db4bd`) was deployed after its entry was written; the container's code matched `e9b461b`.

### 49.1 The fold ignored the narration check

An outside review found that `_share_anonymous_narrator` folded a chapter's unnamed "I" into the
nearest named narrator whose story it fits without asking `could_say_i`: a chapter whose narration
names Oliver in six paragraphs was rejected for him by `chapter_narrators`, then handed to him anyway
with its "I said" lines (reproduced). In a novel alternating between two first-person tellers, where
the model names one and leaves the other "The Narrator", that gives the second teller's chapters to
the first. The fold now only picks a narrator who could be the chapter's "I"; otherwise the chapter
keeps its own unnamed narrator. Test covers the whole flow. The ten exact replays (§46.4) score the
same.

### 49.2 A reproducible benchmark

The review asked for runs that say what made them and for the replay tooling in one place.

- `evaluate_cast.py` writes `cast.manifest.json` beside each run: commit (read from `.git`, since the
  container has no git) plus a hash of every app source file, model name and Ollama digest, the
  input's hash, settings, the answer key's hash (`--reference`), calls, time and status (also when the
  run fails).
- `replay_cast.py` (moved in from the session scratchpad) replays a saved run through the runner, so
  replays get manifests too; it counts the requests the saved run never made and stops when the code
  asks different attribution questions than the code that made the run.
- `cast_audit_eval.py` takes a book's alias file (kept outside git with its answer key; the three
  test books have one now), judges an unnamed "The Narrator" as whoever most of its lines belong to,
  checks chapter narrators where the key lists them, and prints a one-line `--summary`. On the ten
  replayed casts it matches the session's scratch scorer, with unresolved lines now counted apart
  (Apex Prey 3: 7 wrong + 1 unresolved, previously 8 wrong), and it flags the window-16 runs before
  §46 with 3 and 2 of 4 chapter narrators wrong, none after.
- The README of `experiments/multivoice` says how to use them: replay for code that asks nothing new,
  three or more live runs per variant for prompt changes.

### 49.3 A narrator only described, and point-of-view chapters

The books analysed overnight on the §47 code were checked chapter by chapter. The Ugly Love of
Monster Girls was wrong where it matters most. Its "I" is Markus (others call him by name, the
narration never does), but the model labelled him "Man" in most chapters: 461 lines under "Man" (read
in the narrator voice) and 370 under "Markus" (Gabriel), with chapters 29-39 narrated by "Markus" and
so read in Gabriel's voice. The book marks its point-of-view chapters ("Nora's PoV:" chapter 6,
"Yuki's PoV:" chapter 7, a "Selina PoV" section inside chapter 39), and the narrator choice also gave
chapter 8 (whose narration names Yuki 20 times) to Yuki through the fold bug of §49.1, and 9 and 53 to
Nora and a description. Three causes, all fixed:

- **"Oh man!" was an address to "Man".** The address check ignored case; a name in direct address is
  written with its capital. Names now match as written (the "hey"/"oh" before them in any case). The
  ten replays and turn-taking are unchanged.
- **A description could be a teller or the book narrator.** "Man", "the girl" and "The Narrator" are
  labels, not names: never a teller, never the book narrator that overrules votes.
- **Point-of-view chapters blocked the main teller.** Markus is named in the narration of Nora's and
  Yuki's chapters, so he failed "never named in the story's narration". A chapter whose "I said" lines
  went to another named person is now that person's: it neither counts for nor against a teller, nor
  gets one. A chapter whose "I" is unnamed or only described takes the teller, with that label's lines
  in the chapter.

Rerun offline on the saved cast (no model; its "I said" lines already carry the chosen narrators):
the deployed code gives every chapter to "Man"; the new code gives chapter 6 to Nora, 7 to Yuki and the
other 53 to Markus, matching the book's headings; the Selina section inside chapter 39 stays Markus's
(narrators are chosen per chapter). Master of Bodies (first-person chapters in a third-person book) is
unchanged. Over 157 first-person chapters of ten books, the teller now names 95 right, leaves 62 alone
and none wrong (Mom's Guidance now names Troy: its chapters' votes go to him, so a silent female "Troy"
twin no longer splits his addresses). Goblin Stepsister Obsession still abstains: its narrator is
split between "Rakos" and "Onii-chan", both with votes.

- Tests: 946 app tests pass in the container.

### 49.4 Deployed and checked live; the rest of the review deferred

`0cfb3db` (with §49.1-49.3) was deployed after Master of Bodies finished, the queue paused and
empty; the container matched the commit. Live runs on the deployed code, each with a manifest
(commit `0cfb3db`, model digest recorded):

| Run | Before | Live now |
|---|---|---|
| You Like It Darker | 100 wrong, chapter narrators 0 of 4 wrong | identical |
| Apex Prey 3 | 7 wrong + 1 unresolved, Polly throughout | identical |
| Six Wakes | 2 wrong | identical |
| Monster Girls chapters 2-9, 29-31, 53 | the saved cast: 5 of 12 chapter narrators wrong, "Man" and "Markus" two voices | 12 of 12 match the book's PoV headings; no "Man" in the cast |

In the Monster Girls run the model voted Nora for Yuki's chapter; the narration check dropped it.

Measured and not done: re-asking whole untagged exchanges whose answers break turn-taking. Of You
Like It Darker's 100 wrong lines, such exchanges hold 52 lines, 26 of them wrong; the other test books
have none. The best case is about 26 lines on one book, at the risk of the 26 the model has right, so
it was left as an idea (scratch patch only, nothing committed). The listening comparison of voice
choices (review point 5) is for the owner. Not done either: merging a narrator split under two names
that both get "I said" votes (Goblin Stepsister Obsession's "Rakos" and "Onii-chan").

Casts analysed before these changes keep their narrators: re-analyse a book (Monster Girls, Goblin)
before generating it.

## 50. Faster Breeze chapters: length-sorted batches, cached references, quicker checks (2026-10-02)

The owner asked Codex how to speed up Breeze generation, and asked Claude to make the changes it
agreed with. Codex proposed five. Three were done, one measured change was added, and two were left.

| Codex's proposal | Outcome |
|---|---|
| 1. Group a chapter's units by delivery and length before batching | Done (§50.1) |
| 2. Cache encoded voice references in the server | Done (§50.2) |
| 3. Skip Whisper's word times for Breeze | Done (§50.3): 7% faster checks |
| 4. Try shorter references | Not done (§50.5) |
| 5. Lean model loader to cut memory | Not done (§50.5) |
| (added) Hear 3 takes at once | Done (§50.3): the checks had become the bottleneck |

### 50.1 Length-sorted batches (`_breeze_batches`)

A batch generates until its longest take ends. The app used to send units in book order, 32 at a
time, and the server then split each request by template, so directed lines became small calls of
their own. In the Breeze log for Master of Bodies, 36 small chunks (under 24 units) held 13% of the
units and took 25% of the generating time (688 of 2,754 s). One chunk of 4 units took 32 s, about
what a full chunk takes.

Now each attempt's pending units are split into plain and directed, sorted longest text first, and
cut into batches of 32. A group's leftover partial batch holds its shortest units. Equal lengths
keep book order. The takes still go back in unit order.

Evidence:
- **Simulation over the 7 finished Breeze books' clip maps.** The cost is the sum of each call's
  longest take, first attempt only. Savings: Apex Prey 2 39%, Depths of Desire 41%, Forbidden
  Temptation 31%, Master of Bodies 36%, On Earth as it is Beneath 37%, Trad Wife 49% (310 → 299
  calls), Whores Versus Sex Robots 36% (174 → 162 calls). Codex's estimates were 28-47%. This
  version also puts each group's leftover batch last.
- **Replay on the live server.** Units kept their real voice, transcript, instruction and text
  length; their words were sentences of the same length from another book. Same seeds both ways,
  run in the order book, grouped, grouped, book.
  - Master of Bodies chapter (101 units): book order 138 and 137 s, grouped 84 and 87 s (62%).
  - Whores Versus Sex Robots chapter (183 units, 9 directed): book order 324 and 338 s, grouped
    190 and 185 s (57%).
- **End to end through `OpenAITTSProvider`.** Real Breeze and Whisper, one 108-unit chapter in the
  Chloe voice, seeds pinned. Run in a fresh model load in the order new, old, new: 197.7 s,
  325.2 s, 209.0 s. Generating time was 189 and 201 s against 307 s. No retries in any run.

### 50.2 Reference cache (`breeze/server.py` `ReferenceCache`)

The pinned runtime reads and encodes the voice clip for every item, and twice for a directed item
(its guidance prompt again). Each encode takes 35-50 ms on the 4070, so 1.1-1.6 s of every
32-item batch. `BreezeSynthesizer` now routes `breeze_infer.templates._encode_prompt_audio` through
a cache keyed by path and checked against size and modification time. Unloading clears it. If a
future pin drops that function, the server logs a warning and runs uncached.

Checked in the container: repeated fresh encodes of 4 clips are bit-identical, so a cached clip
gives exactly what a fresh one would. 32 lookups through the runtime's own path took 0.06-0.10 s,
mostly the `os.stat` on the Windows-mounted `/voices`. The saving is about 3% of a batch.

### 50.3 Speech check: no word times, three takes at once

`SpeechChecker.transcribe(audio, words=False)` skips faster-whisper's word alignment. Only
Chatterbox's lead-in cut uses the word times. On 64 real Breeze takes (3 alternating runs), the
median fell from 791 to 736 ms per take, with all 64 transcripts identical.

With length-sorted batches, the later batches of shorter units generated in 14-31 s, while
hearing 32 takes took 27-39 s. Generation waited on the checks: 8.4 and 7.0 s in the first end-to-end run, plus
about 20 s of checks after the last batch. Whisper is now loaded with `num_workers=3` and 4 threads
each, and `_check_breeze_batch` hears a batch's non-silent takes 3 at a time. Takes per 32 on the
24-thread host, all with the same 64 transcripts:

| Threads x workers | Per 32 takes |
|---|---|
| 8 x 1 (before) | 23.0 s |
| 6 x 2 | 19.7 s |
| 8 x 2 | 20.8 s |
| 4 x 3 | 17.3 s |
| 6 x 3 | 17.6 s |

Memory: 736 MB peak RSS with one worker, 758 MB with three (the weights are shared). In the end-to-end
runs, checks went from 39.4/30.2/26.6 s to 26.6/20.0/18.5 s, and generation no longer waited.

Not measured: one take at a time on 4 threads instead of 8, which is how the Chatterbox path, voice
transcripts and voice design use it. Chatterbox is stopped.

### 50.4 New log lines (app)

- `Breeze attempt n/3, batch i of k: N units (directed), seed=…`
- `Breeze attempt n/3: checked N takes in Xs`
- `Breeze attempt n/3 waited Xs for the checks of batch i` (only when it waited at least 1 s)

Together with the server's `chunk of N` lines, these split a chapter's time into generating,
checking and waiting.

### 50.5 Not done, and what was seen

- **Shorter references (4).** The live voices are 6.4-15.3 s, except "good morning.wav" (25.4 s).
  The 26 s clip suspected in §36.1 is no longer in the folder. Trimming changes the voice, so the
  owner's ear would decide; there's little left to gain.
- **Lean loader (5).** It needs an audit of the pinned runtime and an isolated memory test. Today's
  runs add evidence for that direction. In one end-to-end run, after the old code's runs, the WSL
  process held 11.2 GB of dedicated GPU memory and 0.67 GB of shared memory. The identical first
  batch (same seeds, same audio lengths to 0.1 s) then took 176.7 s, against 109.3 s before. After
  an unload and load it took 104.2 and 115.0 s. This is the first direct sign that a reload restores
  speed, which the §48 guard relies on.
- **Codex's web findings** (upstream `--fast-all`, audio.cpp, BreezeRT) weren't pursued, for the
  reasons Codex gave: memory, hardware and latency-not-throughput.

Risks and things to watch:
- **The §48 guard counts a chunk of at least 24 units as full.** Books with many soft lines now make
  full directed chunks. These run slower per item (cfg 4.0, two prompts). One per chapter can't pull
  the median of 4 below 1.5x, but a book that is mostly whispers might. If a `slowed down` line
  follows directed chunks, that's why.
- **A unit retried on its own is still slow.** One long take ran at 0.37x (77.6 s) in the live check.
  This is not new.
- **No whole book has been run yet.**

### 50.6 Deployed and checked

Both containers were rebuilt with the queue empty and paused; `/opt/breeze-infer/server.py` and the
two changed app files match the working tree. Live check on the deployed `/app_src`: a 34-unit
chapter went out as batches of 32 and 2. One near-silent take was re-sent alone with a new seed and
passed. Breeze was unloaded afterwards, as it was found.

- Tests: 956 app tests pass (1 skipped); 28 Breeze server tests (2 new). The provider and speech-check
  tests were written by a Sonnet subagent and reviewed here.

## 51. A real book on §50, and GPU memory under WSL (2026-10-02)

### 51.1 Glass Children: the first whole book on length-sorted batches

The owner queued a small cast book, Glass Children: 8 chapters, 2,056 units, 166 min of audio.
- **Speed:** it finished in 46.5 min including the checks, retries and the M4B, which is 3.6x real
  time. Whole books ran at 2.5-2.8x before §50. Full chunks had a median of 3.85x (it was 3.1x).
- **Checks:** generation waited on them for 16 s in all.
- **Retries:** 27 units were retried (1.3%, the usual 1.2-1.8%). Three were kept although no attempt
  passed: 0:36:16 (Zoe, a 3-character line that came out as 5 s, match 0.0), 0:53:41 (Zoe, 0.67)
  and 1:39:50 (Nina Peterson, a 432-character unit, 30.9 s, 0.33).
- **Guard:** it never fired; the slowest full chunk ran at 1.45x. The book had one directed batch.
- **Listening:** the owner's verdict is pending.

Speed now falls inside each chapter, since its longest units go first; compare chapters by their
first batch. Chapter 8's first batch ran at about half the usual seconds generated per second of
audio. The logs can't say whether one long take or memory caused it.

### 51.2 The spill, measured

With the book done and Breeze still loaded, Windows showed the WSL VM holding 11.18 GB of the
card's dedicated memory plus 2.26 GB of shared (system) memory. Unloading took the card from 11.68
to 1.16 GB and the shared memory from 2.27 to 0.24 GB. That spill is larger than the 0.67 GB seen
in §50.5 or the 1.6 GB in §48.

### 51.3 Experiment: three allocator settings on one fixed workload

**Logging first.** Every `chunk of N` line now ends with `GPU peak N MiB, reserved M MiB`: the
chunk's own peak (`max_memory_allocated` since a reset) and what PyTorch keeps reserved afterwards.

**The workload.** Three chapters' batch shapes, all grouped as the app does, with fixed seeds:
171 long Crichton sentences (Chloe voice), the Whores Versus Sex Robots chapter with 9 directed
units, and a Master of Bodies chapter. Each setting got a fresh container. A PowerShell sampler read
the Windows GPU adapter counters every 2 s.

| Setting | Workload | Chunk peaks | Reserved | Windows dedicated max | Shared max |
|---|---|---|---|---|---|
| Default allocator | 406.1 s | 7.5-9.2 GB | 9.68 GB, flat | 10.37 GB | 0.15 GB |
| `expandable_segments:True` | 407.2 s | 7.5-9.1 GB | 9.23 GB, flat | 10.15 GB | 0.12 GB |
| `empty_cache()` after each chunk | 402.6 s | 7.5-9.2 GB | 7.6-10.2 GB | 10.22 GB | 0.12 GB |

None of the settings spilled. 7 minutes on a fresh load doesn't reach the state a 46-minute book
left behind. None of them changed the speed either: each chunk took within a few percent of the
same time.

**Chosen: `expandable_segments:True`.** Compose sets it by default as
`PYTORCH_CUDA_ALLOC_CONF=${BREEZE_CUDA_ALLOC_CONF-expandable_segments:True}`, and an empty
`BREEZE_CUDA_ALLOC_CONF` turns it off. Its reserve sat about 0.1 GB above the peak, against 0.5 GB
with the default. Its segments grow and shrink in place, so it resists the fragmentation suspected in
Glass Children. After that book the card held 11.68 GB, against 10.37 GB at this workload's highest
point, and 2.27 GB had spilled; a bigger peak in the book can't be ruled out.

**Rejected: emptying the cache after each chunk.** It was tried behind an env switch and then
removed. Between chunks the reserve fell, but while it grew back it fragmented, and its highest
point (10.2 GB) was above the default's. The highest point is what decides a spill.

### 51.4 Deployed; not yet known

The Breeze container was rebuilt with the queue idle; it reports `PYTORCH_CUDA_ALLOC_CONF=
expandable_segments:True` and `/opt/breeze-infer/server.py` matches the working tree. Live: one
sentence generated, with the new memory figures on its log line. Breeze was unloaded afterwards.

Still unknown: whether the reserve stays flat through a whole book now. The next book answers that.
- In `docker logs breeze`, the `reserved` figure across the book should stay near the chunk peaks
  (about 9.2 GB).
- After the book, the Windows counter `\GPU Process Memory(*)\Shared Usage` should be well under the
  2.26 GB seen here.
- If the reserve still creeps up, the next step is a model reload between chapters.

- Tests: 28 Breeze server tests pass.

### 51.5 The same book again, with expandable segments

The owner re-ran Glass Children into a second folder ("Glass Children-1"). Same cast and text; the
seeds were random. Both runs, side by side:

| | First run (default allocator) | Re-run (`expandable_segments:True`) |
|---|---|---|
| Book time | 46.5 min (3.6x) | 42.1 min |
| Generating | 42.3 min | 38.2 min |
| Full chunks: median / slowest | 3.85x / 1.45x | 4.15x / 2.00x |
| Units retried | 36 | 20 |
| Kept after failing every attempt | 3 | 2 |
| Highest chunk peak | not logged | 9.86 GB |
| Reserved | not logged | 7.6 GB at load, 10.24 GB after 5 min, then flat (10.27 GB at the end) |
| After the book: card / shared | 11.68 GB / 2.27 GB | 10.77 GB / under 0.2 GB |

The spill is gone, and the reserve stayed flat over the 42 minutes, 0.4 GB above the book's biggest
peak. The speed gain is partly luck (the re-run needed fewer retries); one run each can't separate
the two. A long book is the next check that memory stays flat over hours.

## 52. Breeze under WSL: allocator faults instead of out-of-memory (2026-10-05)

### 52.1 What happened

Widow's Point (2025): 19 chapters, 573,981 characters, cast mode with adaptive delivery.
- **First run (10:50-13:35):** chapter 5 ("Video/audio footage #1A", 1,354 units in 43 batches)
  failed. Its 32 longest units got no audio in all three attempts (10:55, 11:24, 11:29), and a unit
  the server never voices fails its chapter (§50). So no M4B. Chapter 14's third-attempt batch of 32
  hit the same error at 13:26, but that chapter still converted on takes kept from earlier attempts.
- **Retry (13:36):** chapter 5's batch 1 failed again at 13:38, and so did batch 2 at 13:41 (it had
  passed in the first run). The owner stopped the book at 13:49 so this could be fixed first.

The Breeze log:
- The first failure, at 10:55, five minutes after a fresh model load, was `RuntimeError: CUDA driver
  error: device not ready`, raised by an allocation in the attention code.
- Every failure after that was `!handles_.at(i) INTERNAL ASSERT FAILED at
  "/pytorch/c10/cuda/CUDACachingAllocator.cpp":430`.
- Failing chunks of 32 ran 155-172 s before failing, against about 55 s for healthy full chunks in
  the same chapter.
- The reserve went 7.79 GB at load, 9.38 GB at 10:52, then 11.44 GB from 10:56 on, flat. The
  biggest chunk that succeeded that day peaked at 10.29 GB. `nvidia-smi` showed 11,924 of 12,282 MiB
  in use.

### 52.2 Cause

Breeze runs torch 2.9.1+cu128 with `expandable_segments:True` (§51). Line 430 is
`TORCH_INTERNAL_ASSERT(!handles_.at(i))` in `ExpandableSegment::map`. The open issue pytorch#166234
reports the same assert on the same line (torch 2.9.0, WSL, an RTX 4090). pytorch#188008 explains
it: `map()` records a handle for each page and then maps the pages one by one. If a driver call fails
partway, the recorded but unmapped pages stay recorded, and every later growth over them trips the
assert, until the process ends.

So on a full card under WSL, PyTorch never raises out-of-memory. The first failed growth is a driver
error, and every growth after it is the assert. Both are plain RuntimeErrors. The server split a
chunk only on `torch.cuda.OutOfMemoryError`, so the whole chunk failed, and the same 32 units failed
on every attempt.

Why these batches needed more memory is less certain. Chapter 13's longest sentences are as long as
chapter 5's (its 32 longest sentences total 13,536 characters against 12,816) and it went through.
The failing chunks ran about 3x as long as healthy ones, which fits a take running on toward the
token cap (3x the expected length plus 3 s).

### 52.3 Decisions

**The server treats both errors as running out of memory and splits the chunk**
(`is_allocator_fault`: a RuntimeError naming `CUDA driver error` or `CUDACachingAllocator`).
- A damaged process still works within the memory it has already mapped. After each fault it ran
  full chunks peaking at up to 10.29 GB inside its 11.44 GB reserve; only growth past the reserve
  fails. Half a chunk needs less.
- Kernel errors (`CUDA error: ...`) are not split. They usually break the CUDA context, and a split
  wouldn't help.

**The next unload restarts the server.**
- Unloading the model doesn't clear the stranded pages, so only a new process does.
- The first fault sets a flag and logs `GPU allocator fault` once, with the GPU memory figures.
- `POST /api/unload` then replies and, as a background task, sends SIGTERM to its own process.
  Uvicorn is PID 1 and shuts down cleanly; compose's `restart: unless-stopped` starts a fresh
  process.
- The app unloads Breeze only before a cast LLM run (`engine_gpu.unload_breeze_if_loaded`), so no
  book is waiting, and the next load is minutes away.

**Rejected:**
- *Restarting right after the faulting request.* That would land mid-book. A reload costs 20-35 s,
  the card is just as full afterwards, so the next long batch would fault again, and the split
  already copes.
- *Turning expandable segments off.* With the default allocator, WSL spills an overfull card into
  Windows RAM instead of failing: the 2.26 GB spill of §51, and the 2026-10-01 VM crash.
- *Reporting the damage in `/health`.* Docker doesn't restart an unhealthy container, and the app
  doesn't read it.
- *Upgrading PyTorch.* pytorch#187955 (merged 2026-06-23) makes the failed mapping roll back. Which
  release ships it wasn't checked, and a new base image would mean re-checking the pinned
  `breeze-tts` runtime.

**Not done:** a chunk that faults still uses its 2.5 min before it splits. Capping the first batch
of very long units would avoid that; first see how often it happens.

### 52.4 Checked; not yet known

- **Tests:** 33 Breeze server tests pass (28 before). The 5 new ones check that:
  - a fault is split like out-of-memory;
  - the server restarts only after a fault, and only at an unload;
  - other CUDA errors and non-RuntimeErrors are neither split nor followed by a restart.
- **Restart in a throwaway container** (new image, no GPU): SIGTERM to uvicorn as PID 1 gave
  `Finished server process`, exit 0, restart count 1, and the server came back up.
- **End to end with a fake synthesizer raising the assert:**
  - a 4-item batch came back with every item's audio;
  - the unload replied 200;
  - the server logged `restarting the server to clear the GPU allocator fault` and came back up.
- **Deployed with the queue idle:** the `breeze` container was recreated, and
  `/opt/breeze-infer/server.py` matches the working tree. The new process starts with a clean
  allocator.
- **Not yet seen live:** chapter 5 re-run. `docker logs breeze` should show `GPU allocator fault at a
  chunk of 32`, then `out of memory at batch 32; retrying as 16 + 16`, then the chunks of 16
  finishing. After the next cast analysis it should show `restarting the server to clear the GPU
  allocator fault`.

### 52.5 Live: Widow's Point chapter 5 on the new server

The owner restarted the book at 14:00 with finished chapters kept, so only chapter 5 ran. The model
loaded in 18.6 s in the freshly deployed process.

- **14:03:21:** batch 1, the 32 longest units, failed again with `CUDA driver error: device not
  ready`, this time in a fresh process. So the batch really doesn't fit the card; earlier damage
  wasn't the reason. After emptying the cache, the server logged `GPU allocator fault` (GPU free
  3,626 of 12,281 MiB, reserve 7,378 MiB) and split the batch 16 + 16. The halves made 300.1 s of
  audio in 184.2 s (peak 9,357 MiB) and 326.2 s in 82.1 s (peak 8,606 MiB). With the 155 s spent
  before the fault, that batch took about 7 min instead of about 1.
- **14:39:23:** the third attempt's batch of 30 split 15 + 15. The fault line is logged only once,
  by design.
- **Every other chunk:** 42 full chunks of 32 ran without a fault. The highest peak was 9,947 MiB.
- **14:42:33:** chapter 5 converted: 1,354 units, 133.6 min. 1,311 units passed on the first take,
  13 on the second, and 30 went to a third.
- **14:43:01:** the M4B was built from all 19 chapters, 10.80 h.

Of the 32 units that never got audio before the fix, 29 passed on their first take and 2 on their
second. The last one, a 270-character line, was kept after failing all three (match 0.14).

A separate pattern, not linked to the fault: chapter 5 has 28 takes kept after failing the check,
all in Cora.wav. Nearly all are short lines (6-50 characters), all flagged as speech mismatch. Across
the book Cora's flag rate is 2.3% (28 of 1,226 units), against 0.5% for the narrator (Chloe, 24 of
4,379). Gabriel (5 of 214) and Everett (9 of 462) are also around 2%. Chapter 14 was made before the
fix and has 41 flagged takes; the 32 units of its third-attempt batch lost that attempt to the fault.

Still to see: the restart at the next unload. The Breeze process has carried the fault flag since
14:03 (restart count 0), so the next cast analysis should log `restarting the server to clear the
GPU allocator fault`.

### 52.6 Live: the restart at the next unload

The owner's next cast analysis unloaded Breeze at 14:49:45, and the server logged `restarting the
server to clear the GPU allocator fault` after its 200 reply. Uvicorn shut down cleanly, and Docker
had it back up 4 s later (restart count 1). The app saw nothing wrong: `Breeze model unloaded (GPU
memory freed)`, and the cast ran.

The next book (Showering With Jennifer) loaded the model at 15:18 in 17.5 s. Its full chunks ran at
about 4-6x real time. The reserve stayed at 9.0-10.0 GB, with peaks up to 9.8 GB, and there were no
allocator faults through 16:12.

## 53. Faster Breeze: the depth decoder as a CUDA graph, quicker speech checks (2026-10-05)

### 53.1 Where a chapter's time went

Two books after §52 (Showering With Jennifer and three chapters of Stranded): 18 chapters, 136 min.
- **Time:** generating 91%, gaps between batches 4%, assembling and saving chapters 5%. Retry rounds
  were about 6%, counted within those.
- **Whole chapters:** 3.0-4.3x real time, with no downward trend over the evening. Within each chapter
  the speed falls from about 5x to 2.3x and then to 0.4-1.5x for the retry batches, because units go
  longest first (§50). That fall is what looked like a slowdown.
- **Checks:** 0.51-0.62 s per take in every chapter. Generation waited on them 0-21 s per chapter.
- **While generating:** the GPU was 17-40% busy at 45-65 W of 200 W, and the server sat at 100% of
  one CPU core. Generation was limited by launch and host overhead, not by GPU compute.

### 53.2 The depth decoder was three quarters of it

A Sonnet subagent read the pinned upstream loop, and the key lines were checked by hand. Every frame
of plain (no-CFG) generation runs a whole Hugging Face `generate()` for the depth decoder
(`generation_breeze.py` ~977): a 2-token prefill (the backbone's hidden state and codebook 0), then
14 one-token steps through 12 layers. Each call builds a fresh DynamicCache and syncs with the host
once per step. Timed around that call with the stock code, it was 17.0 of a 32-medium chunk's
22.7 s and 8.8 of a 32-short chunk's 12.1 s: 75%, about 195 ms per frame.

### 53.3 `breeze/fast_depth.py`

- **Same math, fixed shapes:** the same modules run on fixed buffers: a 16-slot KV cache, and RoPE
  and causal masks precomputed. The sampling steps are the same: reserved codec ids suppressed,
  temperature 0.9, top-k 50.
- **Graphs:** the 15 steps are captured once per row count (1, 2, 4, 8, 12, 16, 24, 32, up to
  `BREEZE_MAX_BATCH`), largest first, sharing one memory pool, when the model loads (4.2 s for ten).
  A frame copies its rows in, replays one graph and copies the codes out. Inside the graph, sampling
  is the exponential race (argmax of p / Exp(1), an exact categorical draw), so nothing needs the
  host.
- **Exact mode:** samples with `multinomial` as `generate()` does. Run beside the stock decoder on
  every frame of two batches with the CUDA RNG rewound (`experiments/breeze-speed/check_depth.py`),
  97.8% of 2,026 frames and 99.6% of 34,442 codes came out identical. The rest are bf16 rounding from
  the fixed-size attention flipping one draw, and the rest of that frame after it. An offset or
  position error would break nearly every code.
- **Switch:** `BREEZE_FAST_DEPTH=0` keeps the stock decoder, and a failed capture logs why and does
  the same. Directed (CFG) lines still use upstream's own loop.
- **A bug found on the way:** the first version created the static cache inside the capture function
  and didn't keep it. After capture its memory went to new tensors, the backbone's `input_ids` among
  them, and every replay wrote keys and values over them. That showed up as garbage codes, device
  asserts and segfaults that seemed to depend on bucket size. Keeping the cache with its graph fixed
  all three, and `torch.cuda.empty_cache()` between replays is then safe (30 of 30 replays clean). The
  server calls it on its out-of-memory path.

### 53.4 Results

Fixed workload of real Stranded sentences, four voices, fixed seed, in a throwaway container
(`experiments/breeze-speed/`, workload kept out of git):

| Chunk | Stock | Fast | Peak memory (stock → fast) |
|---|---|---|---|
| 32 short | 11.8 s (4.67x) | 5.0 s (10.8x) | 8.64 → 8.77 GB |
| 32 medium | 23.7 s (5.67x) | 10.4 s (12.7x) | 8.72 → 8.85 GB |
| 32 long | 112.8 s (7.32x) | 50.5 s (16.0x) | 10.27 → 10.40 GB |
| 64 medium | 28.9 s (8.83x) | 12.6 s (20.2x) | 10.08 → 10.21 GB |

Per frame at 32 rows: 255-275 ms down to 100-115 ms.

Whisper on every saved take, judged as the app judges (pass at 0.70):

| | Stock | Fast |
|---|---|---|
| 32 short | 32/32 | 32/32 |
| 64 short | 63/64 | 63/64 |
| 32 medium | 32/32 | 32/32 |
| 64 medium | 64/64 | 64/64 |
| 32 long | 30/32 | 30/32 |

Mean match was equal or higher with the fast decoder (0.992 → 0.999 on 32 short). The deployed module
timed the same as the prototype, run back to back: 13.0 against 13.0 s and 51.8 against 51.3 s.

### 53.5 Measured and not taken

- **SDPA attention instead of eager:** with the graph, frames were no faster (110 against 100 ms on 32
  medium), with about 100 MB less memory. Kept eager.
- **96 short lines in one chunk:** the stock decoder hit `device not ready` (§52). 64 fits at
  10.0-10.2 GB.
- **Upstream's own CUDA-graph fast path:** it handles one request at a time (it asserts a batch of 1)
  and pairs rows for CFG, so it doesn't fit batched books.
- **Whisper competing for the CPU:** the same takes generated while the speech check ran (12 threads)
  took 25% longer for 32 short, 17% for 32 medium and 10% for 64 medium. The decode loop is bound to
  one core.

### 53.6 Speech check: 6 workers x 3 threads, greedy for batch checks

The same 96 fast takes, time per 32:

| Setting | Short | Medium | Long | Verdicts |
|---|---|---|---|---|
| 3 x 4, beam 5 (before) | 13.6 s | 16.5 s | 38.8 s | |
| 3 x 4, beam 1 | 13.5 s | 15.6 s | 29.4 s | 96/96 the same |
| 6 x 3, beam 1 (now) | 11.5 s | 13.4 s | 25.1 s | 96/96 the same |

`transcribe()` takes a `beam_size`. Breeze's batch checks pass `speech_check.BATCH_BEAM` (1). A voice
clip's words, which Breeze clones from, and the voice-design check keep beam 5.

With generation 2.2-2.4x faster, the checks now set the pace of short and medium batches: 32 checks
take 11.5-13.4 s against 5-10 s of generating. Long batches stay bound by generation (about 50 s
against 25 s).

### 53.7 Next

- **Faster checks first:** bigger batches for short and medium lines (64 medium runs at 20x) only pay
  once the checks keep up. Two candidates: Whisper on the GPU inside the Breeze process, or a quick
  first pass with a smaller model, with Whisper small only for the takes it doubts.
- **The backbone:** it is now the main GPU cost, about 65-100 ms per frame at 32 rows. A static cache
  and a graph there are the next generation lever.
- **Directed (CFG) lines:** they still run upstream's loop, two depth forwards per step and no cache.

### 53.8 Tests and deploy

- **Tests:** 38 Breeze server tests pass (5 new: the switch, the batch size, a failed capture). The
  app suite passes, 959 tests with 1 skipped (one new test for the beam setting; the fake checkers now
  take `beam_size`).
- **Deploy:** `breeze` and `epub-to-audiobook` were rebuilt with the queue paused. The containers'
  `server.py`, `fast_depth.py`, `speech_check.py` and `openai_tts_provider.py` match the working tree.
- **Live (A Tale of Two Nannies, cast mode, from 19:16):**

| Chapter | Audio | Time | Speed |
|---|---|---|---|
| 1 | 27.4 min | 4.8 min | 5.68x |
| 2 | 22.4 min | 5.4 min | 4.14x (one allocator fault) |
| 3 | 23.1 min | 4.2 min | 5.45x |
| All three | 72.9 min | 14.5 min | 5.04x |

  Before this change, whole chapters ran at 3.0-4.3x (§53.1). The graphs loaded in 3.7 s.
  - **Checks:** 0.38-0.43 s per take (it was 0.51-0.62 s). Generation now waits on them 47-62 s per
    chapter, about a fifth of the time, as §53.6 predicted.
  - **The fault:** chapter 2's first batch, the 32 longest lines, overflowed the card. It was split
    16 + 16 after 57 s, and the halves peaked at 8.2 and 8.8 GB.

## 54. Breeze reads every line plain: no directed lines (2026-10-05)

The owner doesn't care for directed lines and asked to drop them if that speeds things up.

On Breeze, adaptive delivery only ever directed soft speech: whispers, murmurs, lines said under the
breath (§47's listening ruled out loud directions). Each directed line ran in a small batch of its
own through the guided (CFG) path. That path makes two backbone passes per frame, and its depth loop
has no cache and no graph (§53.7). In the two books timed in §53.1, directed lines took 4% of batch
time (Showering With Jennifer, 44 lines) and 8% (Stranded, 41 lines). With plain batches now about
twice as fast, that share roughly doubles. The single directed line in A Tale of Two Nannies chapter 1
took 12 s for 2.2 s of audio.

- **`build_config`:** a Breeze book always runs with adaptive delivery off, books queued earlier
  included.
- **Queue:** settings store it off for Breeze.
- **UI:** the Adaptive delivery checkbox shows only for Chatterbox.
- **Unchanged:** units and voices. On Breeze, adaptive and plain units are split the same way; adaptive
  only added the mood. Chatterbox keeps adaptive delivery. Breeze's instruction support stays for voice
  design.
- **Tests:** three Breeze UI tests now expect plain lines, including a book queued with delivery on.
  The app suite passes, 959 tests with 1 skipped.
- **Deploy:** waits for an idle queue, since restarting the app mid-book loses the chapter in progress.

## 55. A quick first hearing: Whisper tiny.en, with small only for the takes it doubts (2026-10-05)

### 55.1 Why

After §53 the speech check set the pace of short and medium batches.
- **Waiting:** generation waited on the checks for 47-62 s a chapter.
- **CPU:** the checks' 18 Whisper threads slowed Breeze's single-core decode loop by 10-25%.

Whisper encodes a padded 30-second window per take, so a short take costs small almost as much as a
long one. A smaller model hears faster.

### 55.2 The test (`experiments/breeze-speed/two_stage.py`, `two_stage_thresholds.py`)

- **Real takes:** the 472 saved from §53's bench, from both decoders.
- **Bad takes, made three ways:**
  - each take scored against the next line's text in its batch (wrong words, 472);
  - every third take cut to its first 60% (missing words, 158);
  - every third take with another take's audio appended (a runaway tail, 158).
- **Models:** Whisper small (the check), base.en and tiny.en, each heard as Breeze's batch checks
  hear: 6 workers x 3 threads, beam 1, no word times.

| Model | 32 short | 32 medium | 32 long |
|---|---|---|---|
| small | 11.2 s | 13.3 s | 25.3 s |
| base.en | 4.2 s | 5.5 s | 10.1 s |
| tiny.en | 2.1 s | 3.1 s | 5.3 s |

In the two-stage check, a take passes if the quick model scores it at least the quick pass mark;
otherwise small decides. The cost is the real takes sent on to small; the risk is the bad takes
passed that small alone would reject.

| tiny.en mark | Real takes sent on | Wrong words passed | Cut passed | Tail passed |
|---|---|---|---|---|
| 0.70 | 12/472 | 0/472 | 5/19 | 1/142 |
| 0.80 | 17/472 | 0/472 | 1/19 | 0/142 |
| **0.85** | **24/472** | **0/472** | **0/19** | **0/142** |
| 0.95 | 41/472 | 0/472 | 0/19 | 0/142 |

(The second number in each risk column is how many of those takes small itself rejects.)

- **base.en:** passed one runaway tail at every mark up to 1.0, and is twice as slow as tiny.
- **A weak model is lenient on missing words:** at the app's own 0.70 mark, tiny and base passed 5-6
  cut takes that small rejects. They fill in the missing words more readily. Hence the stricter
  quick mark.

### 55.3 Found on the way: the check barely notices a missing ending

Whisper small at 0.70 passed 139 of the 158 takes cut to 60%. A take that loses its last 40% scores
about 0.75, so it passes. That is unchanged here: catching it would take a length test (a
too-short take for its text) or a stricter mark, and either means more retries. Worth measuring on
real cut-off takes before changing anything.

### 55.4 What changed

- **Quick model:** `speech_check.get_quick()` loads Whisper tiny.en (75 MB, downloaded on first use)
  from `SPEECH_CHECK_QUICK_MODEL`, by default the folder `faster-whisper-tiny.en` beside the main
  model. `off` hears everything with small, and it is also off when the speech check is.
- **Pass rule:** `quick_pass` settles a take at `QUICK_PASS_SCORE` (0.85), or when `match()` can't
  judge the text (digits, no letters). That would happen whichever model heard it.
- **Breeze's batch checks:** tiny hears every take, then small hears the ones tiny didn't settle. A
  failed quick hearing counts as doubted. Every rejection is small's. The log line reads
  `checked 32 takes in 3.1s (2 heard again by Whisper small)`.
- **The clip map's `match`:** the quick model's score for takes it settled (0.85 or more), small's
  for the rest.
- **Unchanged:** voice clip transcripts and voice design still use small with beam search.

### 55.5 Tests and deploy

- **Tests:** the app suite passes, 965 tests with 1 skipped. The 6 new ones cover:
  - where the quick model is loaded from, and that it downloads itself;
  - the off switch;
  - the pass rule;
  - that settled takes never reach small;
  - that only doubted takes are heard again, and small decides them;
  - that a take both hearings reject is sent again.
- **Deploy:** the app was rebuilt with the queue paused, carrying §54 too. The container's files match
  the working tree. The live container loads tiny.en from `/app/models/faster-whisper-tiny.en` in
  0.4 s.
- **Live (Stranded's last three chapters, 20:07-20:15):**

| Chapter | Audio | Time | Speed | Heard again by small |
|---|---|---|---|---|
| TEN | 22.5 min | 2.5 min | 8.94x | 37/411 |
| ELEVEN | 26.4 min | 3.0 min | 8.89x | 48/503 |
| TWELVE | 19.9 min | 2.4 min | 8.31x | 34/395 |

  The same book's first chapters ran at 3.3-3.7x (§53.1), and A Tale of Two Nannies at 5.0x on the
  fast decoder with the old check (§53.8).
  - **Checks:** 0.10 s per take (0.38-0.43 s with small alone, 0.51-0.62 s before §53). Generation
    waited on them 0 s per chapter, against 47-62 s, and the GPU generated 82-86% of the time.
  - **Heard again by small:** about 9%. A cast book has more short dialogue fragments than the test
    set had.
  - **Quality:** each chapter's retries (1.2%, 2.9%, 1.8% of units) and takes kept after failing (0, 3,
    0) sit inside the range of the same book's eight earlier chapters (1.6-4.5%, 0-5). The recorded
    mean match is 0.980-0.984 against 0.983-0.992, since a settled take records tiny's score.
- **Next:** with generation the limit again, bigger batches for short and medium lines are the next
  lever (64 medium lines run at 20x against 12.7x for 32, §53.4), then a per-take length cap for
  runaway takes.

## 56. Bigger batches for short units (2026-10-05)

### 56.1 Measured

After §55 generation was the limit again. The test sent the same 64 units as one request and as two
of 32, on the fixed workload (`harness.py`, with offsets) in two new length groups beside §53's, at
seeds 4242 and 777:

| Units (characters) | 2 x 32 | 1 x 64 | Faster | Peak memory at 64 |
|---|---|---|---|---|
| short (6-28) | 8.9 / 9.6 s | 7.7 / 7.2 s | 13% / 25% | 10.0 GB |
| medium (45-75) | 18.4 / 15.6 s | 12.8 / 12.4 s | 30% / 21% | 10.2 GB |
| mid (90-150) | 28.4 / 28.2 s | 28.7 / 22.2 s | -1% / 21% | 10.6-10.8 GB |
| midlong (150-248) | 50.9 / 43.1 s | out of memory, both seeds | | |

A frame costs 60% more at 64 rows than at 32, and a request runs until its longest take ends. So 64
pays where takes are short and similar. From 90 characters it is a toss-up near the memory limit,
and from 150 it doesn't fit. The desktop held 1.4 GB of the card during these runs (0.6 GB earlier
in the day).

### 56.2 What changed

- **App:** `_breeze_batches` makes a request of 64 units (`BREEZE_SHORT_BATCH_SIZE`) when its longest
  text is at most 80 characters (`BREEZE_SHORT_BATCH_CHARS`), else 32 as before. Units go longest
  first, so a request's first unit sets its size.
- **Server:** `DEFAULT_MAX_BATCH` and compose's `BREEZE_MAX_BATCH` are now 64, so graphs are captured
  up to 64 rows.
- **Slowdown guard:** 3/4 of the batch size, but at most 24 rows, counts as a full chunk, so chunks of
  32 long units still count.
- **Queue estimate:** `BREEZE_GENERATION_SPEED` was a guessed 4.0. In the estimate's own units
  (characters / 20.2 per second of wall time), Stranded's last three chapters ran at 7.3-8.2x, so it
  is now 7.5. Estimates were about twice too long.
- **Expected gain:** units of up to 80 characters were about 40% of generation time (§53.1), so
  chapters should be about 8-10% faster. That is a smaller step than §53 and §55.

### 56.3 Tests and deploy

- **Tests:** the app suite passes, 966 tests with 1 skipped. Batch tests now expect 64 for short
  units, and a new one keeps units over 80 characters at 32. Tests about order, retries and the
  check thread fix the short size at 32, since they test something else. The 38 Breeze server tests
  pass.
- **Deploy:** `breeze` and `epub-to-audiobook` were rebuilt with the queue paused. `/health` reports
  `max_batch` 64, and the files match the working tree.
- **Live (A Tale of Two Nannies chapters 5-7):**

| Chapter | Characters | Time | Estimate units | Audio | 64-unit requests |
|---|---|---|---|---|---|
| 5 | 30,429 | 185 s | 8.14x | 8.99x | 4, at 13.5x |
| 6 | 24,921 | 235 s | 5.25x (one fault) | 6.39x | 5, at 12.1x |
| 7 | 17,815 | 112 s | 7.87x | 9.10x | 5, at 13.8x |

  - **The 64-unit requests:** they peaked at 10.2 GB and never faulted. Retries stayed normal (7, 7
    and 4 units sent again).
  - **The gain:** chapters 5 and 7 ran at 7.9-8.1x in estimate units, against Stranded's 7.3-8.2x
    without them (§55). That is within book-to-book variation: a few percent at most, less than the
    8-10% expected.
- **The fault, now the bigger loss:** chapter 6's first request, 32 of its longest units, overflowed
  the card after 52 s and was split 16 + 16. With chapter 2 (§53.8) that is 2 of this book's 7 chapters
  so far, each losing about a minute, roughly 20% of the chapter.

  A straight-line fit to the bench peaks gives about 7.5 GB with no units, plus about 34 MB per unit,
  plus 0.13 MB per unit per frame. By that:
  - 32 units of about 400 characters fit while their takes end normally (measured 10.4 GB).
  - If one take runs on to the current cap (3x the expected length plus 3 s, about 1,040 frames), the
    whole request runs that long and needs about 12.8 GB, which doesn't fit.
  - At 16 units it would need about 10.1 GB.
  - At 32 units with a 2x cap, about 11.4 GB, still at the edge.

  Next: smaller requests for the longest units, and a per-take length cap.
