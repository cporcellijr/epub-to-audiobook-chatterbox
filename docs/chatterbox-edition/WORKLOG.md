# Chatterbox edition: work log and findings

Started 2026-09-25. Written for the owner and for any agent reviewing or continuing
this project. Later investigations name the affected books where needed to trace the evidence.

Since 2026-09-27 this repo holds the whole stack: the audiobook app at the root and the Chatterbox
server in `chatterbox/` (upstream commit 915ae28 as a git subtree, then one commit of local patches),
deployed together by `docker-compose.chatterbox.yml` as two containers.

## How this log is organised

- **Numbers are permanent.** Sections are numbered once, in the order they were written, and code
  comments, READMEs and notes cite them as "WORKLOG §27". Look the number up in the index below;
  the text is in the topic file it names. (Until 2026-10-07 every section was in this one file.)
- **Topic files** open with where things stand now, then their sections in number order, unchanged:
  - [worklog/cast.md](worklog/cast.md): Speaker attribution, narrators and character voices
  - [worklog/breeze.md](worklog/breeze.md): Breeze TTS: the engine, its speed and GPU memory under WSL
  - [worklog/chatterbox.md](worklog/chatterbox.md): Chatterbox (retired 2026-10-01), delivery and listening checks
  - [worklog/app.md](worklog/app.md): Setup, upstream changes, parser, review fixes, run and deploy
- **Adding a section:** take the number after the index's last row, write the section at the end
  of its topic file (`## N. Title (YYYY-MM-DD)`, subsections `### N.x`), add its row to the index,
  and update that file's "Where things stand" when the section changes it.

## Index

| § | Date | Section | File |
|---|---|---|---|
| 1 | 2026-09-25/27 | The setup | [app](worklog/app.md) |
| 2 | 2026-09-25/27 | Chatterbox server (`chatterbox/`) | [chatterbox](worklog/chatterbox.md) |
| 3 | 2026-09-25/27 | Voice reference clips (findings) | [chatterbox](worklog/chatterbox.md) |
| 4 | 2026-09-25/27 | BookOrbit live read-aloud (outside this repo) | [app](worklog/app.md) |
| 5 | 2026-09-25/27 | What was built in this repo, and why | [app](worklog/app.md) |
| 6 | 2026-09-25/27 | Measurements | [app](worklog/app.md) |
| 7 | 2026-09-25/27 | Known limitations and open questions | [app](worklog/app.md) |
| 8 | 2026-09-25/27 | Run, test, deploy | [app](worklog/app.md) |
| 9 | 2026-09-25/27 | Lessons | [app](worklog/app.md) |
| 10 | 2026-09-28 | Review fixes | [app](worklog/app.md) |
| 11 | 2026-09-28 | Kokoro engine and voice deletion | [app](worklog/app.md) |
| 12 | 2026-09-28 | F-45 measured: a compiled token loop | [chatterbox](worklog/chatterbox.md) |
| 13 | 2026-09-28 | Multi-voice narration | [cast](worklog/cast.md) |
| 14 | 2026-09-28 | Adaptive delivery | [chatterbox](worklog/chatterbox.md) |
| 15 | 2026-09-28 | Character profiles for picking voices | [cast](worklog/cast.md) |
| 16 | 2026-09-28 | Voices matched to character profiles | [cast](worklog/cast.md) |
| 17 | 2026-09-28 | Excited lines no longer clip; delivery per character and a narrator from the book's tone | [cast](worklog/cast.md) |
| 18 | 2026-09-28 | First-person books: the narrator reads the "I" character's lines | [cast](worklog/cast.md) |
| 19 | 2026-09-29 | Chatterbox artifact hunt: short dialogue and quoted narration | [chatterbox](worklog/chatterbox.md) |
| 20 | 2026-09-29 | Short-line delivery and clip diagnostics | [chatterbox](worklog/chatterbox.md) |
| 21 | 2026-09-29 | Softer excited delivery; deleted voices forget their gender | [chatterbox](worklog/chatterbox.md) |
| 22 | 2026-09-29 | Batch queuing, and first-person narrators per story | [cast](worklog/cast.md) |
| 23 | 2026-09-30 | Bookmark investigation: brief artifacts between voices | [chatterbox](worklog/chatterbox.md) |
| 24 | 2026-09-30 | First short-book regeneration after the tag cap | [chatterbox](worklog/chatterbox.md) |
| 25 | 2026-09-30 | Original first-chapter regeneration and remaining tiny-quote artifacts | [chatterbox](worklog/chatterbox.md) |
| 26 | 2026-09-30 | After the context fix: remaining marks are two-word quotes; Whisper as a garble check | [chatterbox](worklog/chatterbox.md) |
| 27 | 2026-09-30 | Speech tags point one way; present-tense tags | [cast](worklog/cast.md) |
| 28 | 2026-09-30 | A second look at the lines the text contradicts | [cast](worklog/cast.md) |
| 29 | 2026-09-30 | The narrator reads what the cast can't place; more careful merging | [cast](worklog/cast.md) |
| 30 | 2026-09-30 | A whole collection re-cast: family words, pairs and one person under two names | [cast](worklog/cast.md) |
| 31 | 2026-09-30 | Five cast review fixes, tested between phases | [cast](worklog/cast.md) |
| 32 | 2026-09-30 | Multiple EPUB chapters inside one text file | [app](worklog/app.md) |
| 33 | 2026-09-30 | §31–§32 checked against §26–§30 | [cast](worklog/cast.md) |
| 34 | 2026-10-01 | Tone matching: each voice turned down to its own clip | [chatterbox](worklog/chatterbox.md) |
| 35 | 2026-10-01 | Breeze TTS 2 replaces Chatterbox: a batched engine | [breeze](worklog/breeze.md) |
| 36 | 2026-10-01 | Designed voices, and the first real Breeze chapter | [breeze](worklog/breeze.md) |
| 37 | 2026-10-01 | Moods as spoken direction for Breeze; three clips archived | [breeze](worklog/breeze.md) |
| 38 | 2026-10-01 | The Docker crash, the LLM left on the GPU, and the Voice lab under Breeze | [breeze](worklog/breeze.md) |
| 39 | 2026-10-01 | Breeze implementation review and plain-English voice creation | [breeze](worklog/breeze.md) |
| 40 | 2026-10-01 | Apex Prey 2: narrator switched at the chapter boundary | [cast](worklog/cast.md) |
| 41 | 2026-10-01 | Apex Prey 3 cast audit, failed private rerun, and Astra handoff | [cast](worklog/cast.md) |
| 42 | 2026-10-01 | Astra audit implemented: identity fixes and an offline Apex Prey 3 replay | [cast](worklog/cast.md) |
| 43 | 2026-10-01 | A larger cast: Six Wakes test chapters | [cast](worklog/cast.md) |
| 44 | 2026-10-01 | Library scan and a story collection: one narrator per story | [cast](worklog/cast.md) |
| 45 | 2026-10-01 | The remaining errors: what code could fix, and a narrator risk | [cast](worklog/cast.md) |
| 46 | 2026-10-01 | Narrator choice: nobody the narration names, and an unnamed "I" stays unnamed | [cast](worklog/cast.md) |
| 47 | 2026-10-01 | Option 2: the person the others call by name tells the story | [cast](worklog/cast.md) |
| 48 | 2026-10-02 | Breeze slowdown guard | [breeze](worklog/breeze.md) |
| 49 | 2026-10-02 | Review follow-ups | [cast](worklog/cast.md) |
| 50 | 2026-10-02 | Faster Breeze chapters: length-sorted batches, cached references, quicker checks | [breeze](worklog/breeze.md) |
| 51 | 2026-10-02 | A real book on §50, and GPU memory under WSL | [breeze](worklog/breeze.md) |
| 52 | 2026-10-05 | Breeze under WSL: allocator faults instead of out-of-memory | [breeze](worklog/breeze.md) |
| 53 | 2026-10-05 | Faster Breeze: the depth decoder as a CUDA graph, quicker speech checks | [breeze](worklog/breeze.md) |
| 54 | 2026-10-05 | Breeze reads every line plain: no directed lines | [breeze](worklog/breeze.md) |
| 55 | 2026-10-05 | A quick first hearing: Whisper tiny.en, with small only for the takes it doubts | [breeze](worklog/breeze.md) |
| 56 | 2026-10-05 | Bigger batches for short units | [breeze](worklog/breeze.md) |
| 57 | 2026-10-05 | A full book on the new code: a fault, then a spill into system RAM; cap the GPU memory | [breeze](worklog/breeze.md) |
| 58 | 2026-10-06 | Whisper's per-take log lines at DEBUG | [breeze](worklog/breeze.md) |
| 59 | 2026-10-06 | Reviewed identity guidance for future casts | [cast](worklog/cast.md) |
| 60 | 2026-10-07 | A review reply that names the "I" no longer fails the analysis | [cast](worklog/cast.md) |
| 61 | 2026-10-07 | A masked-tag benchmark and four local models | [cast](worklog/cast.md) |
| 62 | 2026-10-07 | Teaching the model: a first fine-tuning trial | [cast](worklog/cast.md) |
| 63 | 2026-10-07 | Teaching round two: the 14B's own answers, corrected | [cast](worklog/cast.md) |
| 64 | 2026-10-07 | A description declared with a new name becomes that person | [cast](worklog/cast.md) |
