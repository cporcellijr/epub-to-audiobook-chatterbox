# Work log: Breeze TTS: the engine, its speed and GPU memory under WSL

Part of the project work log. Sections keep the numbers they were written with; [WORKLOG.md](../WORKLOG.md) lists every section and which file holds it.

Sections here: §35, §36, §37, §38, §39, §48, §50, §51, §52, §53, §54, §55, §56, §57, §58.

## Where things stand (2026-10-07)

- **Breeze TTS 2 (3B) has been the only engine in use since 2026-10-01** (§35). Chatterbox moved to
  the compose profile `chatterbox` and was stopped, and Kokoro was commented out of `.env` (§35). The
  server is `breeze/server.py` in the container `breeze`; it loads the model on demand.
- **Speed:**
  - chapters are batched longest-first, with voice references cached (§50);
  - the depth decoder replays as CUDA graphs (`breeze/fast_depth.py`, §53);
  - Whisper tiny.en hears every take first, and Whisper small only the takes it doubts (§55);
  - live chapters then ran at 8.3-8.9x real time (§55);
  - requests stay at 32 units: 64 was tried (§56) and reverted after a fault (§57).
- **GPU memory under WSL:** a process past roughly 11 GB of the card is spilled into Windows RAM
  instead of failing, and Breeze then runs several times slower (§51, §57). The defences:
  - the reserve is capped by `BREEZE_GPU_MEMORY_GB`, 10.5 (§57);
  - PyTorch's expandable segments are on (§51);
  - an allocator fault splits the request and restarts the server at the next unload (§52);
  - a slowdown guard reloads the model when full chunks slow far below normal (§48);
  - the app unloads the LLM before Breeze loads (§38).
- **Every line is read plain:** Breeze gets no directed (whispered) lines (§54).
- **Voices:** voices designed from the cast profiles (§36), and plain-English voice creation in the
  Voice lab (§39).

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

## 57. A full book on the new code: a fault, then a spill into system RAM; cap the GPU memory (2026-10-05)

### 57.1 What happened

The owner reran A Tale of Two Nannies from scratch: cast analysis 20:58-21:09, then the book from
21:10. Breeze had restarted itself at the cast's unload, clearing an earlier allocator fault (§52), as
designed.
- **Chapters 1-7:** 2.2-2.5 min each, as fast as §55-56.
- **21:25, chapter 8:** a request of 64 short units overflowed the card (`device not ready`) and was
  split 32 + 32. So 64-unit requests do fault; §56's live run of three chapters was too short to see it.
- **From then on:** full chunks ran at 1-3x real time instead of 8-16x. Chapters 9-12 took 6-15 min
  each instead of about 2.5. The rolling median of 4 full chunks sat at 1.5-2.1x; in 135 healthy
  chunks it never went under 6.8x. PyTorch's reserve stayed at 11.18 GB; before the fault it was
  10.5-10.8 GB.
- **Windows' counters (`\GPU Process Memory(*)`) during the slump:** the WSL VM held 11.31 GB of the
  card plus **0.53 GB of shared (system) memory**, with the card at 11.55 of 12 GB. Part of Breeze had
  been moved into system RAM, and every access to it crossed PCIe.
- **The desktop is not the cause:** the counters showed `dwm` at 2.25 GB and spacedesk (a tablet used
  as an extra display) at 0.65 GB. But with Breeze restarted and unloaded, `nvidia-smi` showed
  245 MiB in use for the whole card; those per-process figures count shared surfaces twice. What
  matters is Breeze's own size. Under WSL a process doesn't get refused past roughly 11 GB of the
  card: it gets spilled, which is far worse than a fault.
- **The slowdown guard never fired:** its threshold was still 1.5x, set in §48 when 3x was normal.

### 57.2 Changes

- **No more 64-unit requests:** back to 32 (`BREEZE_BATCH_SIZE`; server `DEFAULT_MAX_BATCH` and
  compose 32). §56's gain was within noise, and the 64s added about 0.5 GB to the peak and caused
  this fault. Graphs are captured up to 32 rows again. The queue estimate (7.5x) stays.
- **GPU memory cap:** `BREEZE_GPU_MEMORY_GB` (10.5, 0 for none, passed by compose) calls PyTorch's
  `set_per_process_memory_fraction` before the model loads. 10.5 GB is under the 10.8 GB reserve that
  ran at full speed today and well under the 11.18 GB that spilled. A chunk that needs more now gets
  PyTorch's own out-of-memory error and is split by the existing path, with no driver fault and no
  spill. The batch of a chapter's 32 longest units peaks near 10.4 GB, so it may split now and then.
- **Slowdown guard at 4x (`SLOW_REAL_TIME`):** between the healthy minimum (6.8x) and the slump
  (1.5-2.1x). A spill like this one now reloads the model before the next request, which frees and
  re-places its memory.

### 57.3 Tests and deploy

- **Tests:** the app suite passes, 965 tests with 1 skipped; the batching tests are back to §55's.
  43 Breeze server tests pass, 5 of them new for the cap (default, value, over the card, off, no CUDA).
- **Breeze restarted first:** at the owner's go-ahead it was restarted mid-book through
  `/api/unload`, the §52 restart, since it carried the fault. That also showed the card at 245 MiB.
- **Deploy:** both containers were then rebuilt mid-book, as the owner asked. The book resumed on its
  own ("resumes after restart"), skipped chapters 1-12 and redid chapter 13. Breeze logged
  `GPU memory cap: 10.5 of 12.0 GB` and graphs for 1-32 rows.
- **Live:** chapter 13 took 3.0 min, model load included, against 6-15 min for chapters 9-12. Its 16
  full chunks averaged 10.3x real time, and the reserve peaked at 10.09 GB, under the cap. No chunk
  hit the cap and nothing faulted.

## 58. Whisper's per-take log lines at DEBUG (2026-10-06)

faster-whisper logs `Processing audio with duration ...` at INFO for every take it hears: hundreds of
lines a chapter since §55, which buried the app's own lines. It is the library's logger
(`faster_whisper`), so `speech_check` adds a filter to it rather than changing the call.
- **The filter:** it turns the logger's INFO records into DEBUG ones and drops them unless the root
  logger, set from the job's log level by `setup_logging`, logs DEBUG. Warnings and errors pass
  unchanged.
- **Tests:** a new test checks both log levels; the app suite passes, 966 tests with 1 skipped.
- **The app's own per-unit line too:** `Clip chapter-N_..._chunk_K_of_M: chapter a-b s, seed=...` is
  now DEBUG. That is one line per unit, and the clip map (`.clips.json`) keeps the same facts.
- **Deploy:** the owner stopped Bewitched! for it.
