# Work log: Chatterbox (retired 2026-10-01), delivery and listening checks

Part of the project work log. Sections keep the numbers they were written with; [WORKLOG.md](../WORKLOG.md) lists every section and which file holds it.

Sections here: §2, §3, §12, §14, §19, §20, §21, §23, §24, §25, §26, §34.

## Where things stand (2026-10-07)

- **Chatterbox is retired.** Breeze replaced it on 2026-10-01 (§35). Its container is stopped and
  kept in the compose profile `chatterbox` for rollback. These sections are its history, but three
  parts still apply:
  - the Whisper speech check of every take (§26), which on Breeze hears with tiny.en first (§55, in
    [breeze.md](breeze.md));
  - tone matching, each voice turned down to its own clip, which applies to Breeze's voices too (§34,
    §35.1);
  - reviewing a finished book by machine listening instead of by ear (§26).
- **Delivery presets are Chatterbox-only:** adaptive delivery (§14), softer excited lines (§21) and
  per-character delivery (§17, in [cast.md](cast.md)). Breeze reads every line plain (§54).

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
