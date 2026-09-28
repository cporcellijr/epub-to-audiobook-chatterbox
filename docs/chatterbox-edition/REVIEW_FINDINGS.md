# Code review findings: Chatterbox edition

Review date: 2026-09-27. Repository state reviewed: commit `1eb4437` (`main`). No application code was changed.
Method: the brief in `REVIEW_PROMPT.md`. Seven work packages (WP1 text pipeline, WP2 narration engine,
WP3 generator and output, WP4 UI/queue/library, WP5 hygiene, WP6 throughput, WP7 Chatterbox server and
deployment) were investigated by junior agents; every finding below was then re-checked by the reviewing
PM against the cited code and, wherever cheap, reproduced with a script in a scratch folder (nothing was
written into the repository except this file). Test baseline before and after: 135 passing.

Severity scale (from the brief): Critical = wrong or lost book, or a crash; High = audible quality
problem, major time cost, or a failure that recurs on every restart; Medium; Low. Confidence: verified
(reproduced here), likely (read from the code, mechanism certain, effect not exercised), speculative.
Known items K1 to K17 from `WORKLOG.md` are referenced where a finding confirms or sharpens them.

**Chatterbox scope.** `chatterbox/` was reviewed as the brief asked: the local-patch commit `71b7e1f`
hunk by hunk, the `/v1/audio/speech` endpoint, `chunk_text_by_sentences`, the stitching and
normalisation code, `engine.synthesize` with its voice cache and autocast handling, `lifespan`,
`config.yaml`, `chatterbox/Dockerfile`, `patches/apply_speed_patches.py`, and the compose file. Not
reviewed: the upload and reference-audio endpoints, the `/tts` UI path and its streaming generator,
the config manager, the static UI, the alternative Dockerfiles, and the pip-installed `chatterbox-v2`
model package (not in this repository; the speed patch edits it at build time). Server-side speed ideas
that need a GPU to test are in section 3, Priority 3b.
A second pass over the rest of `chatterbox/` (remaining endpoints, config manager, browser UI) and the
pip-installed model package is in `REVIEW_FINDINGS_CHATTERBOX.md` (F-49 onward); its section 2 lists
corrections to F-03, F-10, F-22, F-24 and F-45 below.

**Status, 2026-09-28.** The owner approved every finding except the GPU items. What happened to each:

| Outcome | Findings |
|---|---|
| Fixed | F-01 to F-42, F-44, F-52 to F-57, F-61 to F-63, F-66, F-67, F-69; F-64 through F-03's new config file, which has no `ui_state` |
| Partly fixed | F-39: `chatterbox/.dockerignore` added; build tools stay in the image and compose still requests the GPU two ways. F-43: patterns are validated (F-15); no timeout, since the owner writes the patterns |
| Not done: needs a GPU to build and measure | F-46 to F-48, F-59, F-60 |
| Built on branch `feature/f45-compiled-t3`, awaiting GPU validation | F-45: whole requests 2.35x faster with the same model output (`WORKLOG.md` section 12); `chatterbox/fast_t3.py` behind `TTS_COMPILE` (default off), validated by `experiments/f45/validate_build.py` |
| Not done: upstream utilities this deployment doesn't use | F-65, F-68 |
| Do not apply to the installed model package | F-49, F-50, F-51, F-58: the addendum read the package repository's default branch (`df73ac8`), but the image installs `@master` (`cc03573`, now pinned by F-22), where the alignment analyzer is created only for the multilingual model and Turbo is present |

One regression was caught while integrating the fixes and fixed before deployment: F-06's stream
copy placed AAC chapter joins by ffprobe's bitrate estimate (`WORKLOG.md` section 10).

---

## 1. Five findings with the best value for their effort

| ID | What | Why it pays | Effort |
|---|---|---|---|
| F-02 | The app starts before Chatterbox has loaded its model; the OpenAI SDK gives up after ~7 s, so a book resumed after a restart fails its first chapters every time | Removes a deterministic failure on every host reboot or `docker compose up`; one healthcheck plus one `depends_on` condition | S |
| F-03 | The checked-in `chatterbox/config.yaml` that the README tells a new deployment to copy selects the **Turbo** model, seed 0 and exaggeration 1.3, not the Original model and seed 888 everything was tuned on | A rebuild on a new machine silently runs a different model with a different chunk size; fix is editing one data file | S |
| F-01 | "Stop current book" only kills the job process; its pool worker (the process actually talking to Chatterbox) is orphaned and keeps generating the chapter | Stop actually stops; no wasted GPU time competing with the next book | S |
| F-04 | The M4B build fails for GIF, WebP, TIFF and SVG covers, the queue reports "some chapters failed", and Retry fails again forever | Books with those covers can be finished; one ffmpeg flag change | S |
| F-05 | Send paragraph-sized units and stretch the model's own inter-sentence gaps client-side, instead of one request per sentence | About 30 minutes less generation per 10-hour book (the whole +14% pacing overhead), no server change | M |

Runner-up: F-06 (write chapters as AAC and remux without re-encoding) removes the double lossy pass (K1)
and about 3 minutes of encode time per 10-hour book, verified end to end here.

---

## 2. Where the time goes (WP6, checked against WORKLOG figures)

One ~30-minute chapter (36,462 characters of invented prose, 73 paragraphs; `paced_units` produced
332 units, median 107 characters):

| Stage | Seconds | Share | Basis |
|---|---|---|---|
| Server generation (token sampling + decode) | ~984 | 88.8% | derived: 1,771 s of audio at 20.2 chars/s, ÷ 1.8× |
| Per-request fixed cost (K7) | 332 × 0.35 = ~116 | 10.5% | derived from WORKLOG's 0.35 s |
| Client per-unit work (HTTP, SDK, WAV parse) | ~0.8 | 0.07% | measured here, 2.5 ms per unit against a loopback stub |
| Chapter export to MP3 64k | ~7 | 0.6% | measured here |
| M4B build for a whole 10-hour book (once) | ~290 (~4.8 min) | ~1.3% of the book | measured here on 1 h of audio, scaled |

The model predicts 1.61× real time overall, matching the WORKLOG's measured ~1.6×. With the WORKLOG's own
424 sentences for a 30-minute chapter the fixed cost is ~15%, matching the measured +14% pacing overhead.
Conclusion: only two client-side levers matter for time, the request count (F-05) and, for speed ≠ 1.0,
where the time-stretch runs (F-28). Everything else client-side is under 1% combined. Two assumptions in
the brief were wrong and are recorded in section 5: pydub does not spawn ffmpeg to read WAV, and the
server's resample step is a no-op at 24 kHz.

---

## 3. Verified findings, grouped by priority

### Priority 1

| Field | Content |
|---|---|
| ID | F-01 |
| Title | "Stop current book" orphans the pool worker, which keeps generating |
| Area | WP3 / WP4 |
| Type | robustness |
| Severity | High |
| Evidence | `audiobook_generator/ui/job_queue.py:171-173` `self._process.terminate(); self._process.join()`; `audiobook_generator/core/audiobook_generator.py:233-239` `multiprocessing.Pool(processes=self.config.worker_count, ...)` |
| Problem or opportunity | The job process is SIGTERMed and dies at once, without Python cleanup. Its `Pool(1)` worker, the process that holds the HTTP request to Chatterbox and writes the `.part` file, is reparented to PID 1 and runs on until the chapter finishes (up to ~17 min of GPU time), then renames the finished chapter into `.chapters/`. Meanwhile the UI says "stopped", and if the user resumes the queue the next book's requests compete with the orphan on a strictly serial server. |
| Proposed change | With `worker_count == 1` (always, in the UI) run chapters in-process instead of through a Pool, so `terminate()` kills the only worker. General case: start the job in its own process group (`start_new_session=True` / `os.setpgid`) and have `stop_current` signal the group, or install a SIGTERM handler in `run_job` that calls `pool.terminate()`. |
| Benefit | Stop is immediate; no wasted server time; no surprise chapter appearing after a stop. |
| Effort | S |
| Risk | In-process chapters change nothing else for `worker_count == 1`; process-group signalling needs a test on the real container. |
| Confidence | verified |
| Checked by PM | Reproduced: a parent with `Pool(1)` SIGTERMed while the worker slept; the worker stayed alive with ppid 1 (exit code -15 on the parent). WP3 reproduced it independently. |

| Field | Content |
|---|---|
| ID | F-02 |
| Title | App starts before Chatterbox is ready; a resumed book fails on every restart |
| Area | WP7 |
| Type | robustness / deployment |
| Severity | High |
| Evidence | `docker-compose.chatterbox.yml:48-49` `depends_on: - chatterbox` (no condition, no healthcheck); `chatterbox/server.py:138-156` `engine.load_model()` runs inside `lifespan` before `yield`, so no connection is accepted until the model is loaded; `audiobook_generator/ui/job_queue.py:45-49` re-queues a RUNNING job at start-up, `tick()` starts it within 2 s; `audiobook_generator/tts_providers/openai_tts_provider.py:91` `OpenAI(max_retries=4)` |
| Problem or opportunity | After a reboot or `docker compose up`, the queue resumes the interrupted book immediately. The SDK retries connection errors 4 times with 0.5/1/2/4 s backoff and gives up after ~7 s (measured here against a refused port). Model load takes longer than that, so chapter after chapter fails (each after ~7 s) until the server is up; the book ends FAILED with the note "some chapters failed" and needs a manual Retry. The same applies whenever Chatterbox alone is restarted mid-book (K6). |
| Proposed change | Add a `HEALTHCHECK` to `chatterbox/Dockerfile` (e.g. `curl -f http://localhost:8004/api/model-info`, which reports `MODEL_LOADED`) and `depends_on: chatterbox: condition: service_healthy` in the compose file. Independently, make the provider (or `run_job`) wait for the server before the first request, e.g. poll `/api/model-info` for up to 10 minutes, since restarts of Chatterbox during a book are a normal event here. Note the healthcheck only works while F-10 is fixed or the check has a generous timeout, because a busy server blocks every endpoint. |
| Benefit | Removes a deterministic failure on every restart; the queue's crash-recovery feature works as intended. |
| Effort | S |
| Risk | Slightly longer start-up before the UI appears. `curl` must be in the Chatterbox image (one more apt package). |
| Confidence | verified (mechanism and 7 s window); model-load time on the owner's GPU is unknown (open question Q1) |
| Checked by PM | Timed `OpenAI(max_retries=4)` against a refused port: `APIConnectionError` after 7.0 s. Read the SDK's retry rules (408/409/429/5xx and connection errors; backoff 0.5 s doubling to 8 s). Confirmed the requeue-on-start code path. |

| Field | Content |
|---|---|
| ID | F-03 |
| Title | The checked-in `chatterbox/config.yaml` is the upstream template: Turbo model, seed 0, exaggeration 1.3 |
| Area | WP7 |
| Type | deployment / docs |
| Severity | High |
| Evidence | `chatterbox/config.yaml:11-12` `model: repo_id: chatterbox-turbo`; `:21-27` `exaggeration: 1.3`, `seed: 0`; `README.md:53-54` and `.env.example:6` tell a first run to copy this file into `CHATTERBOX_DATA`; `chatterbox/server.py:1430` `DEFAULT_CHUNK_SIZE = 500 if engine.loaded_model_type == "original" else 400`; WORKLOG sections 1 and 2.1 (Original model, seed 888) |
| Problem or opportunity | The file has never been edited since the subtree import (`git log -- chatterbox/config.yaml`). A fresh deployment that follows the README loads the Turbo model, gets 400-character chunks, non-deterministic generation and a different default delivery, while every measurement, the speed patches and the app's estimate constants assume the Original model. The app's own fallback (`chatterbox_ui.py:48`, exaggeration 0.5) also disagrees with the template. Nothing errors; the output is just different from what was tuned. |
| Proposed change | Commit the live deployment's values (model, seed, generation defaults, `TTS_BF16` note) into `chatterbox/config.yaml`, or ship a separate `chatterbox/config.audiobook.yaml` that the README points to, leaving the upstream template untouched for easier subtree pulls. Re-diff after each `git subtree pull`. |
| Benefit | Rebuilds and new machines reproduce the validated setup. |
| Effort | S |
| Risk | None. |
| Confidence | verified |
| Checked by PM | Read the file and the engine default (`engine.py:350` also defaults to `chatterbox-turbo`); confirmed the README/.env instructions. |

| Field | Content |
|---|---|
| ID | F-04 |
| Title | M4B build fails for GIF, WebP, TIFF and SVG covers; the queue misreports it and Retry loops |
| Area | WP3 |
| Type | bug |
| Severity | High (Critical for the affected books: they cannot be finished from the UI) |
| Evidence | `audiobook_generator/core/m4b.py:69-71` `-i cover -map 2:v -c:v copy -disposition:v:0 attached_pic`; `audiobook_generator/core/audiobook_generator.py:20-28` `_MIME_TO_EXT` accepts gif/webp/svg/tiff/bmp; `:262-263` the `except Exception` returns False; `audiobook_generator/ui/job_queue.py:194` note "some chapters failed; see log" |
| Problem or opportunity | ffmpeg's MP4 muxer only accepts JPEG, PNG and BMP as attached pictures with `-c:v copy`. For any other cover the mux fails after every chapter has been generated, `run()` returns False, the job is marked FAILED with a note blaming chapters, and because `retry()` forces `skip_existing`, a retry goes straight to the same failing build. There is no way out from the UI. |
| Proposed change | Re-encode the cover to JPEG or PNG in `build_m4b` (`-c:v mjpeg` or `-c:v png` instead of `copy`; cost is one image), or convert the cover when it is saved in `run()`. Also let `run()` distinguish "M4B build failed" from "chapters failed" so the queue note is right. |
| Benefit | Every supported cover type produces a book; the retry loop disappears. |
| Effort | S |
| Risk | None significant; JPEG/PNG covers could keep `copy`. |
| Confidence | verified |
| Checked by PM | Called `build_m4b` with 64×64 covers made by ffmpeg: jpg, png, bmp succeed; gif, webp, tiff fail with "Nothing was written into output file". WP3 additionally confirmed svg fails and traced the retry loop. |

| Field | Content |
|---|---|
| ID | F-05 |
| Title | Cut the request count with paragraph-sized units and client-side pause stretching |
| Area | WP6 / WP2 |
| Type | performance |
| Severity | High (major time cost) |
| Evidence | `audiobook_generator/tts_providers/openai_tts_provider.py:22-23` `MIN_UNIT_CHARS = 40`, `MAX_UNIT_CHARS = 400`; `:27-47` `paced_units` flushes as soon as a unit reaches 40 characters, so nearly every unit is one sentence; `:163-185` one request per unit; `chatterbox/server.py:1430` the server already generates up to 500 characters in one pass |
| Problem or opportunity | About 330 to 420 requests per 30-minute chapter, each paying the ~0.35 s fixed decoder cost: 10 to 15% of generation time (the measured +14% pacing overhead). |
| Proposed change | Send whole paragraphs (split at sentence boundaries when over ~450 characters) as units. After the WAV comes back, find the model's own inter-sentence gaps (RMS envelope, 10 ms frames) and lengthen the N−1 longest gaps of an N-sentence paragraph to the configured sentence pause; paragraph pauses stay as today. WP6 prototyped the detector on synthetic audio (all 4 sentence gaps found, 0 of 16 comma gaps picked). Simpler fallback if the detector misbehaves on real audio: raise `MIN_UNIT_CHARS` to ~150-200 (2-3 sentences per request) and accept the model's ~0.2 s gaps inside a unit; that alone saves ~24 min per 10-hour book but partly undoes what paced narration was built for. |
| Benefit | 332 → 73 requests in the sample chapter: −91 s per chapter, about −31 min per 10-hour book (derived); the +18% audio length from pauses is unchanged. |
| Effort | M |
| Risk | Pause placement quality depends on the gap detector: dialogue, semicolons and sentence-count mismatches between `sentencex` and the model's pauses. Paragraph units get closer to the 1000-token cap, so the 450-character split is essential (see F-07). Validate with the WORKLOG's `silencedetect` method on one real chapter before adopting. |
| Confidence | likely (unit counts measured; detector validated on synthetic audio only) |
| Checked by PM | Re-derived the arithmetic from the WORKLOG figures; confirmed `paced_units` behaviour on sample text (units of 49 to 65 characters for ordinary sentences). Server-side alternatives were checked and rejected (section 5): the decoder cost is per generation pass, not per HTTP request, so a server "pause" parameter saves only ~0.2 min per book. |

| Field | Content |
|---|---|
| ID | F-06 |
| Title | K1 sharpened: write chapters as AAC and remux the M4B without re-encoding |
| Area | WP3 |
| Type | audio quality / performance |
| Severity | High |
| Evidence | `audiobook_generator/tts_providers/openai_tts_provider.py:189-191` chapter export at 64 kbps (`mp3` from the UI, `chatterbox_ui.py:232` `config.output_format = "mp3"`); `audiobook_generator/core/m4b.py:74` `-c:a aac -b:a 64k` re-encodes the whole book |
| Problem or opportunity | Two lossy generations at 64 kbps mono. Measured here: the second pass costs ~32 s per hour of audio (~5.4 min per 10-hour book, serial, at the very end of the job). |
| Proposed change | In M4B mode export chapters as AAC (`_PYDUB_EXPORT` already maps `aac` to ADTS) and build the M4B with `-c:a copy -bsf:a aac_adtstoasc`. Keep MP3 for loose-file output. Alternatively FLAC intermediates plus one AAC pass (quality fix only, same encode time). |
| Benefit | One lossy pass instead of two; M4B build drops from minutes to seconds (50 ms for the 9 s test); about −3 min per 10-hour book net of the slightly slower AAC chapter encode. |
| Effort | M (provider output format in M4B mode, `build_m4b` flags, `chapters_done` file matching, tests) |
| Risk | ADTS chapter durations are estimated by ffprobe (55 ms priming per chapter observed, same order as today's MP3 padding); markers stay internally consistent. Per-chapter ID3 tagging does not apply to AAC (only relevant to loose files). |
| Confidence | verified |
| Checked by PM | Encoded 3 ADTS AAC chapters, concatenated with `-c:a copy -bsf:a aac_adtstoasc`, chapter metadata and a cover: markers at 0/3/6 s, cover attached, 0.05 s wall time. WP3 measured MP3 (19 s/h) vs AAC (31 s/h) chapter encode and the 32 s/h re-encode. |

| Field | Content |
|---|---|
| ID | F-07 |
| Title | Single sentences over ~800 characters reach the 1000-token cap in the paced path (K5 sharpened) |
| Area | WP2 / WP7 |
| Type | audio quality |
| Severity | High (silent truncation when it happens; rare) |
| Evidence | `audiobook_generator/tts_providers/openai_tts_provider.py:37-45` a unit is never split, only joined; `chatterbox/utils.py:1132-1138` `chunk_text_by_sentences` emits a segment longer than `chunk_size` as one whole chunk; WORKLOG 2.1: 1000 tokens ≈ 40 s ≈ 800 characters |
| Problem or opportunity | `MAX_UNIT_CHARS` only bounds the trailing-fragment join; a sentence that `sentencex` does not split (legal text, lists, comma-only run-ons, tables read as one sentence) goes to the server whole, the server keeps it whole, and the model stops at 1000 tokens with no error. WP2 found units of 527 to 715 characters in `tests/long_text.txt`; a 980-character sentence went through as one unit here. |
| Proposed change | In `paced_units`, pass any unit over ~450 characters through the existing `split_long_sentence` (commas, semicolons, then spaces) and insert no extra pause at those cuts. |
| Benefit | No silent mid-sentence truncation; also required for F-05. |
| Effort | S |
| Risk | A mid-clause cut may be audible; the server's 200 ms stitch pause would apply if the pieces were sent in one request, so keep them as separate units with a 0 ms client gap instead. |
| Confidence | verified (path); the truncation itself is documented in the WORKLOG (4 of 629 requests) |
| Checked by PM | Ran `paced_units` on a 980-character comma-only sentence: one unit of 980 characters. Ran the extracted server chunker on a 1000-character sentence: one chunk of 1000. |

| Field | Content |
|---|---|
| ID | F-08 |
| Title | Non-paced path: WAV and FLAC chapters are silently truncated to their first chunk |
| Area | WP2 |
| Type | bug |
| Severity | Critical for the CLI with `--output_format wav|flac` (the UI is unaffected: paced path, MP3) |
| Evidence | `audiobook_generator/utils/utils.py:245-262` `direct_merge_audio_segments` concatenates raw response bytes; chosen by `merge_audio_segments:265-289` when `use_pydub_merge` is falsy (the default); called from `openai_tts_provider.py:143` |
| Problem or opportunity | A WAV or FLAC file with several headers plays only its first ~1800 characters of narration; ffprobe and players see a valid short file. MP3 and ADTS AAC are self-framing and survive. Upstream behaviour, but this fork advertises those formats. |
| Proposed change | Force the pydub merge for non-self-framing formats, or always use the paced joiner (PCM concatenation) which handles this correctly. |
| Benefit | No silently short chapters. |
| Effort | S |
| Risk | None. |
| Confidence | verified |
| Checked by PM | Three 1 s chunks concatenated: ffprobe duration 1.0 s for wav and flac, 3.3 s for mp3. |

| Field | Content |
|---|---|
| ID | F-09 |
| Title | Non-paced path: `--use_pydub_merge` always raises after a successful merge and destroys the chapter |
| Area | WP2 |
| Type | bug |
| Severity | Critical for the CLI with `--use_pydub_merge` (never used by the UI) |
| Evidence | `audiobook_generator/utils/utils.py:234-242`: the `finally` loop removes every temp file, then line 241 `os.remove(tmp_file)` removes the last one again |
| Problem or opportunity | `FileNotFoundError` on the success path propagates into `process_chapter`, whose `finally` deletes the just-completed `.part` file, and the chapter is logged as failed. Every chapter fails with this flag; it also masks any real exception from the merge. |
| Proposed change | Delete lines 241-242. |
| Benefit | The flag works. |
| Effort | S |
| Risk | None. |
| Confidence | verified |
| Checked by PM | Called `merge_audio_segments(..., use_pydub_merge=True)` with two MP3 chunks: `FileNotFoundError` after the output was written. |

| Field | Content |
|---|---|
| ID | F-10 |
| Title | `/v1/audio/speech` blocks Chatterbox's event loop for the whole synthesis |
| Area | WP7 |
| Type | robustness / UX |
| Severity | Medium-High |
| Evidence | `chatterbox/server.py:1395` `async def openai_speech_endpoint`, `:1447` blocking `engine.synthesize(...)`; same in the non-streaming branch of `custom_tts_endpoint` (`:1074`); the streaming branch already uses `await loop.run_in_executor(...)` (`:1018-1020`); WORKLOG 2.1 (`/v1/audio/voices` timed out at 5 s) |
| Problem or opportunity | While a chapter generates, every other endpoint hangs: the voice list, `/save_settings` (the Voice lab's Save waits up to 300 s, `chatterbox_ui.py:109`), the server's own UI, and any Docker healthcheck (F-02). This is why the app reads voices from the folder instead of the server. |
| Proposed change | Make the endpoint a plain `def` (Starlette runs it in a threadpool) or copy the `run_in_executor` pattern from two call sites above, plus a module-level `threading.Lock` in `engine.synthesize` around `generate()` so GPU work stays strictly serial. Small, contained change in `chatterbox/`. |
| Benefit | Health and settings endpoints respond during generation; the healthcheck in F-02 becomes reliable; the Voice lab's waits shrink to the current chunk. |
| Effort | S-M |
| Risk | Without the lock, two concurrent requests would share `chatterbox_model.conds` (`engine.py:483, 519-520`) and cross-contaminate voices. Smoke test on the GPU. |
| Confidence | verified (code); concurrency risk speculative |
| Checked by PM | Read the three call sites; consistent with the WORKLOG's own timeout measurement. |

### Priority 2

| Field | Content |
|---|---|
| ID | F-11 |
| Title | `skip_existing` with a changed chapter selection reuses the wrong chapter when titles repeat |
| Area | WP3 |
| Type | bug |
| Severity | High when it occurs (needs two conditions: repeated titles and a changed selection) |
| Evidence | `audiobook_generator/core/audiobook_generator.py:204-213` renumbering to 1..n; `:79-82` file name = new index + sanitized title; `:111` skip when the file exists; `job_queue.py:137` and `:47-48` force `skip_existing` on retry/resume |
| Problem or opportunity | Nothing records which original chapter produced `0003_Title.mp3`. If a book is re-added into the same folder with different ticks, any chapter whose new index and title match an existing file is skipped and the old audio is kept under the new number. Identical titles are common when every section's `<h1>` is the same running head, in `tag_text` mode (`<blank>`), or when titles are page numbers. Reproduced by WP3 with a fake provider. Queue retries keep the same selection, so the risk is a manual re-add. |
| Proposed change | Write a small manifest in `.chapters/` (output name → original chapter index + text hash) and regenerate on mismatch; or include the original chapter number in the file name. |
| Benefit | No silently mislabeled chapters. |
| Effort | M |
| Risk | Manifest must be written atomically; keep resume working for jobs already in flight. |
| Confidence | verified |
| Checked by PM | Read the numbering and skip code; the collision requires identical sanitized titles, which the parser produces for repeated headings (the `<title>` element is not the cause, see section 5). |

| Field | Content |
|---|---|
| ID | F-12 |
| Title | Same-title books get the same auto-suggested output folder; the second overwrites the first M4B |
| Area | WP4 |
| Type | bug |
| Severity | High when it occurs |
| Evidence | `audiobook_generator/ui/chatterbox_ui.py:469-474` `library_output_dir`; `audiobook_generator/ui/web_ui.py:43-55` `suggest_output_dir`; `audiobook_generator/core/audiobook_generator.py:272` M4B name from the EPUB title; `m4b.py:67` ffmpeg `-y`, `:82` `os.replace` |
| Problem or opportunity | The folder and the `.m4b` name both derive from the EPUB title, and nothing checks the queue or the disk. Two different books with the same title (re-releases, generic titles), or the same book queued twice with another voice, end in the same folder and the second run replaces the first `.m4b` without warning; if both are queued the second also inherits the first's `.chapters/` (see F-11). |
| Proposed change | In `queue_settings`, refuse or disambiguate an `output_dir` already used by a queued/running job or holding a `.m4b`/`.chapters` (append author or a counter). |
| Benefit | No overwritten books. |
| Effort | S |
| Risk | None. |
| Confidence | verified (code); consequence derived |
| Checked by PM | Read the three functions; confirmed `-y` and the final `os.replace` overwrite an existing M4B. |

| Field | Content |
|---|---|
| ID | F-13 |
| Title | M4B chapter markers and tags use the file-name-sanitized title: punctuation lost, hyphenated words fused |
| Area | WP1 / WP3 |
| Type | UX |
| Severity | Medium |
| Evidence | `audiobook_generator/book_parsers/epub_book_parser.py:199` `re.sub(r"[^\w\s]", "", title)`; `audiobook_generator/core/audiobook_generator.py:270` `title.replace("_", " ")` for markers; `:117-118` the same string in `AudioTags` |
| Problem or opportunity | "Chapter One: Don't Look Back" becomes the marker "Chapter One Dont Look Back"; "Twenty-One Nights" becomes "TwentyOne Nights"; "Part I—The Fall" becomes "Part IThe Fall". Every player's chapter list shows this. |
| Proposed change | Keep the raw heading for markers and tags and sanitize only for file names; at minimum replace punctuation with a space instead of deleting it. |
| Benefit | Readable chapter lists. |
| Effort | S |
| Risk | `chapter_selection` scores on the sanitized title today; keep feeding it the same normalized string. |
| Confidence | verified |
| Checked by PM | Parsed a hand-built EPUB: `<h1>Chapter One: Don’t Look Back</h1>` → `Chapter_One_Dont_Look_Back`. WP1 reproduced the hyphen and dash cases. |

| Field | Content |
|---|---|
| ID | F-14 |
| Title | Adjacent table cells without whitespace are read as one word |
| Area | WP1 |
| Type | audio quality |
| Severity | Medium |
| Evidence | `audiobook_generator/book_parsers/epub_book_parser.py:99-100` `_BLOCK_TAGS` has `tr` but not `td`/`th`; `:121` `soup.get_text(strip=False)` with no separator |
| Problem or opportunity | `<tr><td>Name</td><td>Age</td></tr>` narrates as "NameAge". Only tightly packed markup is affected; pretty-printed tables are fine. |
| Proposed change | Add `td`, `th` (and `li` is already there) to the tags that get a trailing separator, or append a single space to cell-like tags. |
| Benefit | Tables are intelligible. |
| Effort | S |
| Risk | Do not use `separator=" "` globally: inline `<em>` inside a word would split. |
| Confidence | verified |
| Checked by PM | `'NameAge Ann30'` for packed cells, `'Name Age'` with newlines between cells. |

| Field | Content |
|---|---|
| ID | F-15 |
| Title | Search-and-replace file parsing: last line loses a character, `==` in the replacement empties the rule, an invalid regex aborts the run |
| Area | WP1 |
| Type | bug |
| Severity | Medium |
| Evidence | `audiobook_generator/book_parsers/epub_book_parser.py:190-191` `split('==')[1][:-1]`; `:150` `re.sub(search, replace, text)` with no error handling; `audiobook_generator/core/audiobook_generator.py:262-263` the only handler ends `run()` |
| Problem or opportunity | (a) `[:-1]` assumes a trailing newline: the last rule of a file without one silently loses its final character (`baz==qux` → replace with `qu`). (b) `a==b==c` yields replacement `""`. (c) An unbalanced `(` raises `re.error` inside `get_chapters`; in the UI `chapter_overview` shows the message, but a CLI run produces zero chapters with only a stack trace. |
| Proposed change | `splitlines()` plus `split("==", 1)`; `re.compile` each pattern on load and report "line N: bad pattern". |
| Benefit | The pronunciation-fix feature behaves predictably. |
| Effort | S |
| Risk | None. |
| Confidence | verified |
| Checked by PM | Reasoned from the code; WP1 reproduced all three with scratch files. |

| Field | Content |
|---|---|
| ID | F-16 |
| Title | `remove_endnotes` deletes digits attached to words and quoted years |
| Area | WP1 |
| Type | bug |
| Severity | Medium (opt-in option) |
| Evidence | `audiobook_generator/book_parsers/epub_book_parser.py:140` `(?<=[a-zA-Z.,!?;”")])\d+` |
| Problem or opportunity | `COVID19` → `COVID`, `B2B` → `BB`, `He said "1990s"` → `He said "s"`. Upstream regex. |
| Proposed change | Limit to 1-3 digits followed by whitespace, punctuation or end of string, and not followed by a letter. |
| Benefit | Endnote removal without corrupting ordinary text. |
| Effort | S |
| Risk | Heuristic; add fixtures for real endnote markers. |
| Confidence | verified |
| Checked by PM | Ran the regex on the three examples. |

| Field | Content |
|---|---|
| ID | F-17 |
| Title | EPUB2 table-of-contents pages referenced only from `<guide>` are narrated |
| Area | WP1 |
| Type | bug |
| Severity | Medium |
| Evidence | `audiobook_generator/book_parsers/epub_book_parser.py:71-74` `_is_nav_document` checks `EpubNav` or the EPUB3 `nav` property only |
| Problem or opportunity | An EPUB2 `toc.xhtml` in the spine passes as a chapter. Usually caught by the chapter auto-selection ("contents" title, short text), but a longer or generically titled TOC page is read aloud. WP1 built such an EPUB and saw the TOC returned as chapter 1. |
| Proposed change | Also treat the item referenced by `<guide><reference type="toc">` as navigation; ebooklib exposes `book.guide`. |
| Benefit | Closes the gap the spine-order fix was meant to close. |
| Effort | M |
| Risk | Some publishers point `type="toc"` at a real content page; keep it a secondary signal combined with the existing heuristics. |
| Confidence | verified |
| Checked by PM | Read the function; WP1's constructed EPUB2 reproduction is consistent with it. |

| Field | Content |
|---|---|
| ID | F-18 |
| Title | Scene-break and punctuation-only paragraphs are sent to the model as speech requests |
| Area | WP2 |
| Type | audio quality / performance |
| Severity | Medium |
| Evidence | `audiobook_generator/tts_providers/openai_tts_provider.py:41-45` a short leftover is emitted as its own unit when the paragraph has no other unit; `chatterbox/utils.py:1084-1155` the server chunker passes `* * *`, `***`, `…`, `—` through as chunks |
| Problem or opportunity | Every `* * *` scene break (common in fiction) costs a request (~0.35 s + generation) and the model is asked to voice symbols; what it produces is unknown (a WORKLOG lesson notes very short inputs are safe, symbol-only ones were not measured). A one-word paragraph like "Yes." is also sent alone, which the comment on `MIN_UNIT_CHARS` says makes the model stumble. |
| Proposed change | Drop units with no letters or digits and replace them with a paragraph pause; consider carrying a very short paragraph's text into the next unit only when it is not dialogue. |
| Benefit | Fewer requests, no symbol narration. |
| Effort | S |
| Risk | Keep genuine one-word lines ("No.") spoken. |
| Confidence | verified (units are sent); audible effect speculative (open question Q6) |
| Checked by PM | `paced_units` produced units `* * *`, `…`, `Yes.`; the extracted server chunker returned each unchanged. |

| Field | Content |
|---|---|
| ID | F-19 |
| Title | K8 sharpened: job processes are forked from a multithreaded server; use the spawn context |
| Area | WP4 |
| Type | robustness |
| Severity | Medium |
| Evidence | `audiobook_generator/ui/job_queue.py:203-204` `self._process_factory(...)` defaults to `multiprocessing.Process` (fork on Linux) called from the worker thread while Gradio's request threads run; `main.py:232` `setup_logging` in the child |
| Problem or opportunity | A fork copies any lock another thread holds at that instant (the logging module lock, `library_index._lock`, the queue's `RLock`) in its locked state; the child then hangs on its first log line, forever RUNNING with 0 progress and nothing in the log. WP4 reproduced the deadlock deterministically with a logging thread. How often the window is hit here is unknown. |
| Proposed change | `multiprocessing.get_context("spawn").Process` as the `process_factory` in `host_ui`. WP4 confirmed `GeneralConfig` and `run_job` pickle, `run_job` imports lazily, and `main_ui.py` has the `__main__` guard. |
| Benefit | Removes a silent-hang class; also makes F-01's in-process option cleaner. |
| Effort | S |
| Risk | Child start-up is slower (re-imports); one job at a time, so negligible. Test on the container. |
| Confidence | likely |
| Checked by PM | Read the start path; the mechanism is standard CPython behaviour. |

| Field | Content |
|---|---|
| ID | F-20 |
| Title | Library picker labels mis-read multi-title, CDATA and editor-first OPF metadata |
| Area | WP4 |
| Type | bug / UX |
| Severity | Medium |
| Evidence | `audiobook_generator/ui/library_index.py:44-46` regex `first(tag)` takes the first `<dc:title>`/`<dc:creator>` in the raw OPF |
| Problem or opportunity | A sort title listed before the display title, a CDATA-wrapped title (shown literally with the CDATA wrapper), or an editor (`opf:role="edt"`) before the author produce wrong labels in the Book picker, which is how books are chosen. WP4 reproduced all three with constructed OPFs. |
| Proposed change | Parse the OPF with `xml.etree.ElementTree` (namespace-aware); prefer the title not marked as a sort/alternate refinement; prefer `role="aut"` creators; fall back to first. |
| Benefit | Correct labels. |
| Effort | S-M |
| Risk | Keep the existing `library_index_test.py` cases passing; speed stays similar (one small XML parse per new file). |
| Confidence | verified |
| Checked by PM | Read the regexes; the failure modes follow directly. |

| Field | Content |
|---|---|
| ID | F-21 |
| Title | K12 sharpened: `output_dir` and the Book box accept any container path; `/voices` is writable |
| Area | WP4 |
| Type | security |
| Severity | Medium (personal LAN deployment) |
| Evidence | `audiobook_generator/ui/chatterbox_ui.py:277-288` validation checks only that the folder is non-empty and the book path is a file; `docker-compose.chatterbox.yml:66` `/voices` mounted writable in the app container |
| Problem or opportunity | An output folder of `/voices` (typed or pasted by mistake) drops chapter MP3s into Chatterbox's voice folder, where they appear as voices; `/app` would put files beside `queue.json`. The Book box reads any file the container can see. |
| Proposed change | `os.path.realpath` and require `output_dir` under `OUTPUT_ROOT` and library picks under `EBOOK_LIBRARY_DIR` (uploads under `queue_uploads/`). |
| Benefit | Removes an easy way to corrupt the voice folder, even by accident. |
| Effort | S |
| Risk | None. |
| Confidence | verified (code); no exposure beyond the LAN is claimed |
| Checked by PM | Read `queue_settings` and the compose mounts; `safe_folder_name` (`web_ui.py:36-40`) does strip `/` and `\`, so voice names cannot traverse. |

| Field | Content |
|---|---|
| ID | F-22 |
| Title | The Chatterbox image installs `chatterbox-v2@master` unpinned |
| Area | WP7 |
| Type | maintainability / deployment |
| Severity | Medium |
| Evidence | `chatterbox/Dockerfile:45` `pip3 install ... git+https://github.com/devnen/chatterbox-v2.git@master`; `chatterbox/patches/apply_speed_patches.py:27-33` exits the build when the patch target changed |
| Problem or opportunity | Any rebuild pulls whatever `master` is that day. The speed patch's fail-closed check is right, but it means an unrelated rebuild (base image, requirements) can fail at an unplanned moment, and two builds are never byte-identical. |
| Proposed change | Pin to a commit SHA; bump it deliberately together with `git subtree pull` and a patch re-check. |
| Benefit | Reproducible builds; breakage only on scheduled bumps. |
| Effort | S |
| Risk | None. |
| Confidence | verified |
| Checked by PM | Read the Dockerfile and patch script. |

| Field | Content |
|---|---|
| ID | F-23 |
| Title | `cover.<ext>` is written visibly before any chapter exists and stays after the M4B is built |
| Area | WP3 |
| Type | robustness |
| Severity | Medium (owner decision, see Q2) |
| Evidence | `audiobook_generator/core/audiobook_generator.py:163-170` cover written directly into `output_folder` before chapters start; `:268-275` `_merge_into_m4b` removes only `.chapters/` |
| Problem or opportunity | For the hours a book generates, the library folder contains only `cover.jpg`, so a folder-watching scanner sees a cover-only book (the same symptom WORKLOG item 9 fixed for previews); after success the loose cover stays beside the `.m4b`. If the book is stopped or fails permanently (F-04) the cover-only folder is permanent. |
| Proposed change | Write the cover into `.chapters/` and move or copy it out only when the book completes (or when loose-file output is chosen). Whether to keep a folder cover next to the M4B depends on the library software. |
| Benefit | No half-made books visible to scanners. |
| Effort | S |
| Risk | Some library apps use a folder `cover.jpg`; confirm with the owner. |
| Confidence | verified |
| Checked by PM | Read the code; WP3 ran a full fake generation and saw `['My Book.m4b', 'cover.jpg']`. |

| Field | Content |
|---|---|
| ID | F-24 |
| Title | The single-file bind mount of `config.yaml` becomes a directory if the copy step was skipped |
| Area | WP7 |
| Type | deployment |
| Severity | Medium (first run only) |
| Evidence | `docker-compose.chatterbox.yml:24` `${CHATTERBOX_DATA}/config.yaml:/app/config.yaml`; README step 1 |
| Problem or opportunity | Docker creates an empty directory for a missing single-file bind source; the server then fails to read its config with an unhelpful error. |
| Proposed change | Have the Chatterbox entrypoint seed the file from the image when the mount is a directory or missing, or use Compose `configs:`; or add a preflight check to the README/`.env.example`. |
| Benefit | Clear first-run behaviour. |
| Effort | S |
| Risk | None. |
| Confidence | likely (standard Docker behaviour; not run here) |
| Checked by PM | Read the compose file and README. |

| Field | Content |
|---|---|
| ID | F-25 |
| Title | The autocast-cache leak fix is skipped when generation raises |
| Area | WP7 |
| Type | performance / robustness |
| Severity | Low-Medium |
| Evidence | `chatterbox/engine.py:490-515` `torch.clear_autocast_cache()` (line 515) runs after the `with torch.autocast(...)` block, inside the `try`; `:526-528` the `except` returns `(None, None)` without clearing |
| Problem or opportunity | The leak the local patch fixes (+248 MB over 240 requests) still accumulates on failed generations (bad chunk, CUDA error). |
| Proposed change | Move the clear into a `finally` around the autocast block. |
| Benefit | Leak fix holds on the failure path. |
| Effort | S |
| Risk | None; the call is safe when nothing was cached. |
| Confidence | verified (control flow) |
| Checked by PM | Read the function. |

| Field | Content |
|---|---|
| ID | F-26 |
| Title | Loose-file output in wav/flac/aac/opus carries no tags |
| Area | WP2 |
| Type | UX |
| Severity | Medium (CLI / non-M4B only) |
| Evidence | `audiobook_generator/tts_providers/openai_tts_provider.py:145` and `:192-193` `set_audio_tags` only when `output_format == "mp3"` |
| Problem or opportunity | Formats offered by `--output_format` produce untagged chapter files; in M4B mode `build_m4b` supplies metadata so it does not matter there. |
| Proposed change | Use mutagen's FLAC/MP4/OggOpus classes per format, or log that tags are skipped. |
| Benefit | Tagged loose files. |
| Effort | S (warning) / M (tagging) |
| Risk | None. |
| Confidence | verified |
| Checked by PM | Read both call sites. |

| Field | Content |
|---|---|
| ID | F-27 |
| Title | K17 sharpened: at speed ≠ 1.0 apply `atempo` once per chapter on the client, not per unit on the server |
| Area | WP6 / WP7 |
| Type | performance / audio quality |
| Severity | Medium (only when speed ≠ 1.0; the UI recommends 1.0) |
| Evidence | `chatterbox/server.py:1466-1467` `apply_speed_factor` per chunk; `chatterbox/utils.py:497-527` one ffmpeg subprocess per call; `audiobook_generator/tts_providers/openai_tts_provider.py:153-155` pauses divided by speed client-side |
| Problem or opportunity | Measured here: 57 ms per ffmpeg spawn (WP6: 57.5 ms, WP7: 57.1 ms) × ~330 units ≈ 19 s per chapter, versus 5.4 s for one `atempo` pass over the whole 30-minute chapter. Stretching speech and pauses together also removes the per-unit stretch boundaries. |
| Proposed change | Always request speed 1.0; stitch with unscaled pauses; run one `atempo` pass over the finished chapter PCM. The server-side speed patch could then be retired, shrinking the subtree diff. |
| Benefit | About −4.6 min per 10-hour book at speed ≠ 1.0; simpler server. |
| Effort | S |
| Risk | None to quality (WSOLA once vs. many times is at least as clean). |
| Confidence | likely |
| Checked by PM | Re-derived from both juniors' independent timings. |

### Priority 3 (compact format)

| ID | Title | Area | Evidence | Problem → proposed change | Effort | Confidence | Checked by PM |
|---|---|---|---|---|---|---|---|
| F-28 | `stop_current` can mark a book that just finished as STOPPED | WP4 | `job_queue.py:166-180` | If the process exited (code 0) in the ≤2 s before `tick()` runs, Stop skips `terminate()` but still sets STOPPED; a Retry would then regenerate the whole book because `.chapters/` was already removed → check `exitcode` and reuse `tick()`'s DONE/FAILED logic | S | verified | read; WP4 reproduced with a fake process |
| F-29 | `chapters_done` counts stale numbered files from a previous run | WP4 | `job_queue.py:89-103` | Progress can show "10 of 10 done" for a job that has produced nothing; `min()` hides it → count only files newer than `started`, or per-job marker | S | verified | read; WP4 reproduced |
| F-30 | Voice preview MP3s are never deleted | WP4 | `chatterbox_ui.py:145-148` `tempfile.mkstemp` | Container writable layer grows ~100 KB per Play for the container's lifetime → unlink the previous preview or sweep on start | S | verified | read; the UI test has to delete it manually |
| F-31 | Uploaded copies orphaned in `queue_uploads/` when validation fails after copying | WP4 | `chatterbox_ui.py:291-299` | The EPUB copy happens before the replace-file copy; a failure leaves both files unreferenced forever → copy after validation, sweep unreferenced files on start | S | verified | read; WP4 reproduced |
| F-32 | Numeric headings padded with whitespace skip the "use text preview" fallback | WP1 | `epub_book_parser.py:161` `re.match(r'^\d{1,3}$', title)` on unstripped text | `<h1>\n 12\n</h1>` keeps "12" as the title → match on `title.strip()` | S | verified | reproduced |
| F-33 | An empty or corrupt chapter file, or missing ffprobe, fails the M4B with an opaque error and no self-healing retry | WP3 | `m4b.py:19-24` `check=True`, no handling | Book stuck like F-04 → catch, name the chapter, delete it so `skip_existing` regenerates it; check ffmpeg/ffprobe once at start | S | verified | read; WP3 reproduced |
| F-34 | Three sanitizers disagree; `safe_book_file_name` truncates after stripping (can end in a space) and none filters Windows device names (`CON`, `NUL`, ...) | WP3 / WP5 | `m4b.py:12-16`, `web_ui.py:36-40`, `filename_sanitizer.py:18-42` | Cross-platform name edge cases; one shared module would fix all → unify, truncate then strip, reserved-name check | S | verified | read; WP3 reproduced |
| F-35 | K14: `.gitattributes` gives Windows checkouts CRLF shell scripts | WP5 | `.gitattributes:1` `* text=auto` | `entrypoint.sh`, `chatterbox/start.sh` fail in the container → add `*.sh text eol=lf` | S | verified | read |
| F-36 | Runtime files not git-ignored; two imports rely on transitive installs | WP5 | `.gitignore`; `requirements.txt`; `chatterbox_ui.py:25` `import yaml`; `epub_book_parser.py:119` `"lxml-xml"` | `queue.json`, `library_index.json`, `queue_uploads/` can be committed from a local run; `pyyaml` comes via Gradio and `lxml` via EbookLib → add to `.gitignore` and `requirements.txt` | S | verified | checked `pip show` |
| F-37 | Dead code and eager debug strings | WP5 | `openai_tts_provider.py:4-5` unused `tempfile`, `os`; `utils.py:2,4,9,11` duplicate imports; `epub_book_parser.py:122,134` `f"...{raw[:]}"` formats whole chapters even with DEBUG off | Noise; the f-strings copy the chapter text twice per chapter (sub-millisecond, but pointless) → remove; use lazy `%s` logging or `isEnabledFor` | S | verified | grep |
| F-38 | Stale upstream compose files and unlabelled docs drift | WP5 | `docker-compose.example.yml`, `docker-compose.webui.yml` (upstream image, other providers); WORKLOG K11 says dependencies are unpinned but `requirements.txt` pins everything | Confusing for a new reader → label the upstream files as upstream examples; update K11 | S | verified | read |
| F-39 | Chatterbox image carries build tools and repo cruft; compose requests the GPU two ways | WP7 | `chatterbox/Dockerfile:13-23` `build-essential python3-dev git` in the runtime image; no `chatterbox/.dockerignore`; compose `:15,19-20` `runtime: nvidia` + env and `:34-41` `deploy.resources` | Bigger image, redundant config → `.dockerignore`, multi-stage or purge, keep one GPU mechanism (test on the host first) | S | verified (redundancy likely) | read |
| F-40 | `_ffmpeg_atempo` swallows ffmpeg's stderr | WP7 | `chatterbox/utils.py:525-527` logs `e` but `CalledProcessError.__str__` omits `.stderr` | A real atempo failure falls back to librosa (the "tin can" path) with no diagnosable reason → log `e.stderr` | S | verified | read |
| F-41 | Chatterbox's deterministic 400 becomes a 500 and earns 4 retries | WP7 | `chatterbox/server.py:1432-1434` raised inside the `try`; `:1547-1549` `except Exception` re-wraps | Only reachable with empty input, so rare → `except HTTPException: raise` before the generic handler | S | verified | read |
| F-42 | Small UI items | WP4 | `job_queue.py:129-140` `retry` keeps the full-book `estimate_seconds`; `:654-677` `input_file.change` runs `chapter_overview` with the old library pick first (double parse, brief wrong table); `:430-432` typed non-file text hides the table with an empty message; `:109` Save waits up to 300 s with no hint | UX polish → scale the estimate on retry/resume; sequence the upload handler with `.then`; return a message; shorten the save timeout or say so | S each | verified / likely (event order not run in a browser) | read |
| F-43 | User regex from the search-and-replace file runs unguarded on the request thread on every option change | WP4 | `epub_book_parser.py:150` via `chatterbox_ui.py:672-677` | A catastrophic-backtracking pattern freezes the request thread; personal UI, so Low → validate patterns (F-15) and run `book_chapters` with a timeout | M | verified (path); not triggered | read |
| F-44 | Tests: the areas above have no coverage | all | `tests/` (grep) | No tests for: merge helpers (`merge_audio_segments` is mocked in the one test that touches it), terminate/orphan, unusual covers, `skip_existing` with a changed selection, `paced_units` over 500 chars or symbol-only, search-and-replace parsing, endnote regex, EPUB2 guide TOC, `stop_current` racing a finished job, `chapters_done` with stale files, multi-title OPF, event wiring → add focused tests alongside each fix | M | verified | grep across `tests/` |


### Priority 3b: server-side speed ideas that need the GPU (all speculative)

From WP6 and WP7. Nothing here could be run in this environment (no GPU, no torch), so each row names
the cheap experiment on the owner's machine that would settle it. These are the only ideas with a
ceiling above F-05's ~30 minutes per 10-hour book, because token sampling is 80 to 90% of generation
time (WORKLOG 2.2) and the GPU sits at 50 to 60%.

| ID | Title | Evidence | Opportunity → proposed change | Estimated gain per 10-hour book | Effort | Risk | How to check on the GPU |
|---|---|---|---|---|---|---|---|
| F-45 | CUDA graphs or `torch.compile` for the per-token step, after fixing the per-call recompile | `chatterbox/engine.py:509-513` (the leak-fix comment: `T3.inference` builds a fresh `T3HuggingfaceBackend` wrapper on every call and resets `self.compiled`); WORKLOG 2.2 "kernel-launch bound", "not yet tried" | Because the wrapper is rebuilt per call, a naive `torch.compile` would recompile on every request and make things slower. First make the wrapper (or its compiled graph) persist across calls, then compile the per-token forward with a static KV cache (`mode="reduce-overhead"`) | 15 to 30% of chapter time, roughly 55 to 110 min (speculative, wide) | L | Static shapes vs. variable prompt and text lengths; silent quality regressions; the change lives in the pip-installed model package, outside `chatterbox/` and outside the subtree | Time 50 identical `engine.synthesize` calls and log whether `compiled` resets; patch persistence; compile the step; compare tokens/s and `nvidia-smi dmon` utilisation |
| F-46 | Overlap the S3Gen decode of chunk *i* with T3 sampling of chunk *i+1* | `chatterbox/engine.py:427-529` one blocking `generate()` per chunk; `chatterbox/server.py:1439-1470` strictly sequential loop | The ~0.35 s fixed decode per pass is not overlapped with anything; a two-stage pipeline inside `generate()` (or across the server's chunk loop) could hide part of it | Upper bound 0.35 s × passes, about 40 min if fully hidden; realistically a fraction, and F-05 already removes most of these passes | L | Model-internals surgery in the pip package; CUDA kernels on one GPU serialise anyway | `torch.profiler` or `nvidia-smi dmon -s u` at 100 ms over one `synthesize()`: if there is no idle valley between the T3 and S3Gen phases, there is no headroom |
| F-47 | BF16 (or at least profiling) for the S3Gen decoder | `chatterbox/engine.py:490` autocast wraps the whole `generate()` call, so S3Gen is nominally inside it; WORKLOG only reports the T3 gain | Measure the decode phase with `TTS_BF16` on and off; if the decoder is float32-bound, cast it | At most ~11 min (a 30% faster 0.35 s decode over ~2,000 passes); small | M | Vocoder numerical quality, audible artefacts | Instrument decode-only time in the pip package with `TTS_BF16=on` vs `off` |
| F-48 | `set_seed` on every chunk, including `torch.cuda.manual_seed_all` | `chatterbox/engine.py:128-141` `set_seed`; `:463` called for every chunk because the default seed is non-zero (888 + chunk index) | CUDA reseeding per pass costs an unmeasured amount (the CPU parts are noise); trade-off is reproducibility | Unknown, probably small | S | Dropping the seed loses reproducible regeneration of a chapter | Time 50 calls with `seed=0` (skips `set_seed`) against `seed=888` |

---

## 4. Suggested order of work

1. Deployment correctness, an afternoon: F-03, F-02, F-24, F-22 (config, healthcheck, first-run mount, pin).
2. Queue robustness, a day: F-01 (+F-19), F-04, F-12, F-28, F-33.
3. Output quality, a day or two: F-06, F-13, F-07, F-18, F-14, F-15, F-16.
4. Speed, two to three days including listening tests: F-05, then F-27 if speed ≠ 1.0 is used.
   If more speed is wanted after that, run the F-45 experiment first (a day on the GPU) before committing to any of Priority 3b.
5. Everything else as convenient; F-44 alongside each fix.

---

## 5. Considered and rejected

One line each, with the reason.

- `<head><title>` wins over the chapter `<h1>` (WP1): false. ebooklib rebuilds each document's head on read and drops the publisher's `<title>`; a hand-built EPUB with `<title>Book</title>` and `<h1>Chapter One</h1>` yields the h1. The `title` entry in `title_levels` is dead code, not a bug.
- Long "About the Author" (3,500 chars) and "Excerpt from the next book" (8,000 chars) are unticked (WP1): by design; the WORKLOG lists these as non-story and the user can tick them.
- README says Python 3.10+ but the image is 3.11 (WP5): no file uses syntax newer than 3.10 (checked with `ast.parse(feature_version=(3,10))`); the base image does not set a minimum.
- `config.speed or 1.0` masks a speed of 0 / string "0" divides by zero (WP2): unreachable; the UI slider starts at 0.5 and the CLI parses a float then validates.
- `instructions: null` is sent in every request (WP2): Chatterbox ignores unknown fields; harmless.
- Two copies of a chapter's PCM in memory (~170 MB) during the join (WP2): transient, one worker, fine.
- `pacing_enabled()` treats pauses of 0 as "pacing on" (WP2): intended; the UI lets users choose zero pauses and still get sentence-level requests.
- Server-side pause parameter so the client can send paragraphs (WP2/WP6): saves only HTTP overhead (~0.2 min per 10-hour book) because the decoder cost is per generation pass; adds a subtree patch.
- One-request-ahead client pipeline (WP6): the server is strictly serial; the client gap is ~2.5 ms per request, under 1 min per book.
- Streaming endpoint (WP6): same GPU work, only earlier first byte; no batch gain.
- pydub spawns ffmpeg per WAV unit (assumption in the brief): false; pydub parses WAV in Python (`_from_safe_wav`), measured 0.02 ms per call.
- Server resample, peak normalisation, int16 clip, WAV header, ID3 tagging, `set_seed` CPU parts (WP6): each under 0.1 ms or a no-op at 24 kHz; noise.
- Per-request peak normalisation causes loudness dips (WP7): only fires when a unit peaks above 0.99 and then scales by at most a few dB; not verified audible. Listen for it if level dips are noticed (Q6).
- `os.replace` in `_save()` may fail on a Windows bind mount (WP4): speculative, no reproduction, no report of it happening.
- MP3 encoder padding adds 30-45 ms per chapter join (WP3): markers stay consistent with the file; below audibility; F-06 changes the codec anyway.
- `.txt` chapter text files stay in the output folder in M4B mode (WP3): `output_text` is an explicit request to keep the text.
- Cover embedded in every chapter MP3 that is later deleted (WP3/WP6): 7 ms per chapter.
- `_ffmpeg_atempo` stage chaining, `apply_speed_factor` fallbacks, `encode_audio` sample-rate variables, `loaded_model_type` initialisation order, `apply_speed_patches.py` fail-closed design, Compose `:?` guard placement, `requirements-nvidia.txt` cu121 wheels on the cu128 base (WP7): all checked and correct.
- Dataframe bool round-trip, `safe_folder_name` traversal, `remove` vs `tick` race, `retry` while a process writes, `_duration` rounding (WP4): all checked and correct.
- Spine order, `linear="no"`, dangling idrefs, nested blocks, `<hr>`, "Chapter 1" of 300 chars, short "Epilogue", parser speed on a 1 MB chapter (WP1): all checked and correct.

---

## 6. Open questions for the owner

1. How long does Chatterbox take from container start to model loaded on the RTX 4070? This sizes F-02 (the failure window is anything over ~7 s).
2. Should a folder `cover.jpg` stay next to the finished `.m4b` (Audiobookshelf-style folder art) or be removed (F-23)? What does BookOrbit expect?
3. Is the live `config.yaml` (Original model, seed 888, delivery defaults) something you are happy to commit as the repo default, or should it live in a separate audiobook config file (F-03)?
4. Is speed ≠ 1.0 ever used for books? If not, F-27 and the server-side speed patch are low priority.
5. For F-05: is trading exact sentence-pause placement (detector-based) for about 15% less generation time acceptable, and would you first want the simpler "2-3 sentences per request" variant A/B-tested with `silencedetect`?
6. Have you heard how the model voices a `* * *` scene break (F-18) or a loud sentence being pulled down (peak normalisation, section 5)? A quick listen to one of each settles both.
7. Is the app ever reachable beyond the LAN (VPN, reverse proxy)? That sets the severity of F-21 and K12.
8. `output_text` in M4B mode leaves `.txt` files beside the book; intended?
9. Is a day of GPU experiments on the token-sampling loop (F-45) worth it to you? It is the only idea with a ceiling above F-05, and it lands in the pip-installed model package rather than in this repository.
