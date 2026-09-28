# Chatterbox edition: work log and findings

Covers 2026-09-25 to 2026-09-28. Written for the owner and for any agent reviewing or continuing
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
| Kokoro-FastAPI, container `kokoro` (stopped) | Alternative engine | ~50× real time vs Chatterbox's ~1.8×, but no voice cloning and flatter delivery. Not wired into this app yet. |

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
| Tests | 241 passing, plus 19 for `chatterbox/` (135 before the review fixes; upstream baseline: 28 tests, 2 erroring on a mock that closed stdout) |

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
