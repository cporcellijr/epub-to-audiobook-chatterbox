# Work log: Speaker attribution, narrators and character voices

Part of the project work log. Sections keep the numbers they were written with; [WORKLOG.md](../WORKLOG.md) lists every section and which file holds it.

Sections here: §13, §15, §16, §17, §18, §22, §27, §28, §29, §30, §31, §33, §40, §41, §42, §43, §44, §45, §46, §47, §49, §59, §60, §61.

## Where things stand (2026-10-07)

- **The pipeline** (`audiobook_generator/core/`):
  - quotations are split per chapter (`dialogue.py`, §13);
  - a line a speech tag names is an anchor and is never asked (`speech_tags.py`, §27, §30);
  - the other lines go to the local LLM, 20 at a time, with the speakers known so far shown
    (`cast_llm.py`, §13);
  - lines the text contradicts are asked once more (`cast_review.py`, §28);
  - identity rules merge one person's names and keep different people apart (§29-§31, §42-§43, §59);
  - a narrator is chosen per chapter and per story (§18, §22, §40, §44, §46-§47, §49);
  - two-person turn-taking is repaired (§45);
  - profiles and measured voice matching pick the voices (§15-§16).
- **The model** is qwen2.5:14b through Ollama, at temperature 0. On 2026-10-01 the owner decided to
  keep the cast LLM local: hosted models' content filters would likely refuse many of the books.
  Newer models that fit the card (gemma4:12b, qwen3.5:9b, qwen3:14b) did no better, and a vote across
  models gained little (§61).
- **Accuracy on the hand-labelled test chapters**, last measured on 2026-10-07 on §59's code, with
  §60's fix for the last book (§61). The answer keys are outside git; the scorer is
  `tests/audiobook_generator/cast_audit_eval.py` (§49.2).
  - Six Wakes: 1 wrong of 504 lines.
  - Apex Prey 3: 6 wrong and 4 unresolved of 269.
  - You Like It Darker: 90 wrong and 2 unresolved of 910, every chapter narrator right.
- **A wider comparison:** the masked-tag set, 600 lines from 27 library books whose tags were
  removed, is for comparing models and prompts (§61.2). qwen2.5:14b gets 376 right.
- **What is left is mostly the model's own reading:** long untagged exchanges, and an addressed name
  taken for the speaker ("..., Vic?") (§44.4, §45.2). Most of these errors are the same in every run
  (§61.1). Prompt changes move the rest by about 15 lines in 900 (§45.3), so judge a prompt change
  over three or more live runs per variant. Teaching the model itself is the next lever (§61.5).
- **Tried and not kept:**
  - evidence written before the answer (§14, §41.4);
  - re-asking unnamed-"I" chapters with only their own people listed (§44.3);
  - "I told her..." read as a tag (§45.1);
  - re-asking whole untagged exchanges (§49.4, measured only);
  - a line naming its own speaker as a review signal (§28.3).
- **Open:**
  - a narrator split under two names that both get "I said" votes (§49.4);
  - an editor merge doesn't survive a re-analysis (§30);
  - casts analysed before a narrator fix keep their old narrators until re-analysed (§49.4).

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

## 59. Reviewed identity guidance for future casts (2026-10-06)

The owner paused the queue and asked to wire the reviewed books' right/wrong examples into future
casting, with attention to first-person stories, long exchanges and collections. The review retained
54 source-verified examples from 23 books (33 right, 21 wrong), after screening 60 casts. These are
historical outcomes across revisions, not an accuracy score for current code. The compact examples,
chapter hashes and diagnostics stay in `data/diagnostics/cast-learning-audit-2026-10-06/`, outside Git.

### 59.1 What runs on future books

- **Shared guidance in `cast_llm.PROMPTS`:** first attribution, review and identity questions receive
  the same compact lessons: follow the speaker rather than an addressee, connect a description only
  to a proved name, keep different people with similar/shared names separate, establish the current
  first-person teller from the source, and respect whole turns and explicit speaker changes.
  This is curated prompt guidance; processing more books does not update Ollama's weights or
  automatically approve its own guesses. The diagnostic Markdown is supporting evidence, not a
  file the runtime silently reads.
- **Shared names:** the known roster and attributed context use a proved unique proper alias when
  possible (`Colin = Hee Haw`, `Alice = Hee Haw`), otherwise an explicit `@key`. Source pronouns or
  the reply's unambiguous gender declaration can resolve a shared bare name; otherwise it stays
  unresolved for review. An ambiguous named tag is asked instead of anchored to the first owner.
  Unknown/incompatible tokens and ambiguous metadata cannot manufacture another character.
- **Self-introductions:** literal introductions retain full names and supported nicknames even when
  the nickname is quoted. A name quoted as someone else's introduction cannot rename the outer
  speaker. A description promoted to a proved name survives the next chapter, and a literal shared
  self-name is retained without overwriting its first owner's alias lookup.
- **Broken quotes:** an explicit speech tag on a continuation overrides damaged quote punctuation
  in attribution. The existing default of `tagged_speakers` stays available to other callers.
- **First person and collections:** the tone prompt distinguishes a teller from `my contact` or
  `my sister`, keeping the narrator checks of §44-49. The live collection smoke test found a
  separate bug: story comparisons treated both chapters' `the girl` labels as shared people.
  `_chapter_cast` now excludes anonymous and chapter-local descriptions from that comparison,
  using the existing `_is_description` helper. Distinct stories keep their narrators and girls.
- **Version:** new cast files record `guidance_version: 2026-10-06.1`. Existing casts are untouched.
  Broad rechecking of every untagged exchange remains off: §45/49 measured regressions from it.

### 59.2 Validation

- **227 regression checks pass in the app container, no skips:** attribution, full cast analysis,
  profiles/narrators, speech tags, review and final voice routing. The collection regression fails
  against the old deployed code and passes with the change.
- **Focused live tests, three runs of each variant:** use the real configured `qwen2.5:14b` through
  production attribution, with recorded requests/replies and exact source chapter hashes. Six
  passages score **30/43 before and 43/43 after in every run**. Proven aliases may keep their
  canonical spelling; distinct seeded identities must retain their exact keys.

  | Passage | Before | After | What is checked |
  | --- | ---: | ---: | --- |
  | Hee Haw | 5/16 | 16/16 | Two established identities with the same display name |
  | Showering With Jennifer | 3/3 | 3/3 | First-person teller and another speaker |
  | Three Little Pigs | 3/3 | 3/3 | One identity under Shirley/Squirrelly; source name promotion |
  | Bewitched! | 15/15 | 15/15 | First-person conversation with Valerie, interrupted turns |
  | Glass Children | 3/4 | 4/4 | Separate Tony and Toni |
  | Depths of Desire | 1/2 | 2/2 | Explicit new speaker despite broken quotation marks |

  Hee Haw's two identities and the first-person tellers are seeded from independently reviewed
  source evidence. These scores test attribution with that evidence available, not whole-book
  narrator discovery or accuracy across the library. Bewitched!'s supplied narrator is Bob Masters,
  established by a different chapter, rather than its saved cast's mistaken Phil identity.
- **Full-pipeline live smoke tests:** two temporary invented EPUBs run through `analyse_book`, tone,
  narrator reconciliation, turn following and profiles. A shared-name introduction survives the
  next chapter; a two-story first-person collection keeps separate narrators and separate unnamed
  girls. Both finish with no review issues. Temporary EPUBs and cast files are removed; no audio is
  generated and no old book is recast.
- **Broader fixture, three live runs per variant:** the existing six invented passages (329 lines)
  score **268, 268, 266 before; 270, 270, 270 after**, keeping the original scorer: 802/987 versus
  810/987 across repeated runs. Lighthouse and Salt Road improve; Ferry, Winter Market and Night
  Shift lose 1-3 correct lines per run. The scorer also penalises descriptive identities for its
  `unknown` labels; its original scoring is retained for an honest comparison. Long untagged
  exchanges still have errors, and these focused changes do not establish a general accuracy gain.

### 59.3 Deployment and retained evidence

The app image was rebuilt with the changed attribution, narrator and speech-tag files; all four
SHA-256 hashes match the tested source. The model digest used for the comparisons is
`7cdf5a0187d5c58cc5d369b255592f7841d1c4696d45a8c8a9489440385b22f6`.
Live test drivers, requests/replies and scores remain in the diagnostic folder as reusable checks
for future guidance changes. All 60 original cast fingerprints still match the review's coverage
file. The queue remains paused.

Only `epub-to-audiobook` was recreated (`docker compose up -d --no-deps epub-to-audiobook`). The
running image is `sha256:854d095123678e315fe68a45eb894f86b0d2130df8c5ff7b382eddde98d90f5f`;
the running module reports the guidance version and matches all four source hashes. The UI returns
HTTP 200, the queue is still paused with no preparation in progress, and all 60 original cast
fingerprints still match after deployment. The test LLM was unloaded through the existing GPU
handover helper.

## 60. A review reply that names the "I" no longer fails the analysis (2026-10-07)

### 60.1 What failed

A private run of You Like It Darker's test chapters (12, 15-17) on the deployed code (`dbdf0d5`,
§59) with qwen2.5:14b failed in its last chapter with `KeyError: 'narrator 3'`. The same failure
would end a real cast job for that book.

1. The Dreamers' "I said" lines are anchors to the chapter's unnamed narrator, key `narrator 3`.
2. The review pass (§28) asks its flagged lines in groups. The first group's reply listed
   `{"name": "William Davis", "aliases": ["I"]}`: the model had named the teller.
3. That alias merges the unnamed narrator into William Davis (§42's rule that "I" is one person per
   chapter). `Roster.merge` retires the old key, but the chapter's decided lines still held it.
4. The next group looked `narrator 3` up to show the anchored lines as `[Name]`, and failed.

The lookup is older than §59 (it read `roster.characters[key]["name"]` before). The window loop
already moved decided lines to the merged key after each reply; the review pass and the line count
after it never did. Two more places held a retired key the same way: the line count
(`count_line`), and the review's "an 'I'-tagged line belongs to the narrator" check, which would
have rejected a correct answer naming the newly named teller.

### 60.2 Fix

`_follow_merges` (`core/cast_llm.py`) moves every decided line to the key its character now has:
- after each window's reply (the old inline loop, unchanged in effect);
- before the review flags lines, and at the start of each review group;
- before the chapter's lines are counted.

The review's "I"-tag check now looks the narrator key up where it uses it, so a teller named
earlier in the same reply counts as the narrator.

Not changed: the known-character list and the narrator line shown to the review are still built
once per chapter's review, so a later group still lists "The Narrator" after a reply named them.
Rebuilding them per group would change the questions every multi-group review asks, so it is left
for a measured change.

### 60.3 Checks

- New test `test_a_review_reply_naming_the_i_moves_the_narrators_lines_before_the_next_group`:
  an unnamed "I" chapter with two unknown lines far enough apart for two review groups, the first
  reply naming the "I". On the old code it fails with `KeyError: 'narrator'`; on the new code
  the narrator's three lines are William's, the second group shows `[William] "Me,"`, and William
  counts 3 lines.
- The failed run, replayed on the new code (`replay_cast.py`, saved replies only), passes the point
  of failure and stops at the next review request, which the failed run never got to send.
- A live run of the same chapters on the new code, with qwen2.5:14b and the same settings, finished
  (`runs/fix60_2026-10-07`): 90 wrong and 2 unresolved of 910, with 0 of 4 chapter narrators wrong
  (the last saved run, `final`, had 100 and 0).
  - Ollama's replies aren't exactly repeatable, so this run left the failed one's path at call 9.
  - It met its own merge in chapter 17's review: a reply listed "The Narrator" with alias "I".
  - Replaying its saved replies on the old code asks a different question at that point (call
    73), so the change took effect there.
- 980 app tests pass in the container (1 skipped: the private answer-key check).
- Not deployed yet.

## 61. A masked-tag benchmark and four local models (2026-10-07)

### 61.1 Why another model, and why another test

Most of the remaining errors are stable. Comparing saved runs line by line:
- Apex Prey 3's 8 wrong lines are the same in all 9 runs.
- You Like It Darker has 63 lines wrong in all four comparable runs (fix3, fix3_prod, phaseA,
  final), which score 88-108 wrong each.

So prompt changes, which move errors around (§45.3), can only reach the part that moves. The rest
needs a model that reads differently. Three hand-labelled books are too few to compare models on,
and labelling more takes days, so the library labels itself.

### 61.2 The masked-tag set

- **The idea.** A line whose paragraph is only the quotation and a bare named tag (`"...," Tom
  said.` / `Tom said, "..."`) loses its tag, and the tag's name is the answer. Each speaker's first
  tag in a chapter stays, so every name is still met once.
- **Books.** 40 books were sampled uniformly from the 1,171 eligible in the library scan (§44.1),
  at most 2 per author, fixed seed, with the four test books held out. 13 had too few clean
  third-person tags, mostly first-person books. That left 27 books, 53 chapters, 4,636 dialogue
  lines, 600 masked lines and 1,522 tags kept.
- **Checks when building.** Each edited chapter keeps the same line ids and the same text
  everywhere else. No masked line is still anchored, and none became a continuation.
- **Running.** The production `analyse_book` runs with the parser handing it the masked text.
  Profiles are skipped, since they don't change who speaks. A masked line is right when its
  character has the tag's name, or a fuller name whose first or last name it is.
- **It is harder than a real book by design:** authors tag exactly the lines that need it. Use it
  to compare models and prompts, not as an accuracy figure.
- **Where it lives.** Everything is in `data/diagnostics/masked_tags_2026-10-07/`, outside git,
  because `masked_set.json` holds book text. The tools there are `select_books.py`, `build.py`
  (no model), `run_models.py` (one container, models in turn, the same queue check as `go.sh`),
  `score.py` and `vote.py`.

### 61.3 Four models on the same code

Thinking was turned off (`reasoning_effort: "none"`) for the three models that think, and all ran
at Ollama's default 4,096-token context. The largest prompt seen so far is 3,909 tokens, and none
has been truncated. Mixture-of-experts models of 26-33B were left out: they would spill into the
16 GB Docker VM.

| Model | Masked lines right (of 600) | Minutes for the set | Apex Prey 3 wrong + unresolved | Six Wakes | You Like It Darker |
|---|---|---|---|---|---|
| qwen2.5:14b (current) | **376** | 24.7 | **6 + 4** | **1** | 90 + 2 |
| gemma4:12b | 368 | **12.3** | 42 + 17 | 2 | 100 + 96 |
| qwen3.5:9b | 340 | 14.5 | 53 + 11 | 24 | 413 + 5 |
| qwen3:14b | 339 | 16.5 | 49 + 6 | 18 | 167 + 63 |

- qwen2.5:14b's You Like It Darker run in the bake-off failed in its last chapter (§60). Its figure is
  the live run on the fixed code.
- **qwen2.5:14b stays.** No newer model that fits the card reads better.
- **gemma4:12b reads third-person lines as well, in half the time, but fails first-person
  identity.** On Apex Prey 3, 33 of Polly's and Gemma's lines went to "girl with needle phobia" and
  17 of Polly's had no speaker. On You Like It Darker it never named the tellers. The narrator code
  (§42-§49) was tuned on how qwen2.5 answers "I said" lines.
- **No refusals.** Every attribution and review reply from every model was a JSON speaker map, on
  explicit books too. "Unusable" replies covered the wrong line ids.
- **Apex Prey 3 on §59's code: 6 wrong and 4 unresolved, against 7 and 1 on `0cfb3db`.** Counting
  lines read in the wrong voice, that is 9 against 8: two of the unresolved are Stuart's and one is
  Polly's, which the narrator's voice reads correctly anyway. Two label-only names ("female doctor",
  "the news reporter") were added to the book's alias file; the old file is kept as
  `aliases.before-2026-10-07.json`.

### 61.4 Combining models doesn't pay

The models get different lines wrong: 492 of the 600 are right in at least one. A vote can't find
them, though:
- the best three-model vote (qwen2.5 + gemma4 + qwen3.5) gets 390, 14 more than qwen2.5 alone, for
  three times the analysis time;
- a two-model vote can't beat qwen2.5, since ties go to it;
- agreement is no confidence signal: lines all four models agree on are right only 81% of the time.

### 61.5 Next: teaching the model (researched, not started)

Notes with sources are in `finetune_research.md` in the same folder.
- **Data.** PDNC has 28 public-domain novels and 37,131 quotations, each with speaker and
  addressees. Its repository has no licence file: private training is low-risk, and the authors can
  be asked. The library's own masked tags are free training data in the owner's genres, as long as
  the benchmark books are held out.
- **Training.** A QLoRA of a 14B model fits the 12 GB card, tightly at 4,096 tokens. Qwen3.5 can't
  be trained this way.
- **Into Ollama.** 0.35.1 no longer loads LoRA adapters, so the result must be a merged GGUF
  (`FROM model.gguf`, with the base model's chat template).
- **Other option.** A fine-tuned encoder (ModernBERT) scored 94.5% on PDNC against 89.8% for
  Llama-3-8B zero-shot, about 1,000 times faster. It chooses among candidate mentions, so it would
  need more pipeline work.
