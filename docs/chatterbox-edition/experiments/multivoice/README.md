# Multi-voice validation

Behind `WORKLOG.md` section 13. `validate_multivoice.py` runs the production speaker attribution
(`audiobook_generator/core/cast_llm.py`) against the configured local LLM on the labelled passages in
`fixture/`, prints accuracy, confusion, reply-failure rates, speed and whether Chatterbox was unloaded
and reloaded around the pass, then narrates a two-minute multi-voice sample through the real pipeline.

The fixture is six **invented** passages (no real book text) of 22-77 dialogue lines each, 329 lines
in all, written in the usual novel styles: tagged and untagged exchanges, action beats, a speech that
runs over paragraphs, titles and first names for the same person, and never-named speakers whose
lines are labelled `unknown`. Passages 05 and 06 (added 2026-09-30, WORKLOG §27) hold the cases a real
book got wrong: a speech tag ending in a colon that introduces the *next* quotation ("Pell answered
without looking up: ..."), an untagged quotation before another speaker's tagged one in the same
paragraph, interrupted speech, and present-tense tags ("Dom murmurs"). Each quotation is preceded by
its true speaker in `«...»`; the script strips the tags and checks that `core.dialogue` finds exactly
as many lines as there are labels.

```
docker run --rm --network tts -e PYTHONPATH=/app_src \
  -e OPENAI_BASE_URL=http://chatterbox:8004/v1 -e OPENAI_API_KEY=not-needed \
  -e TTS_VOICES_DIR=/voices -v <CHATTERBOX_DATA>/voices:/voices:ro \
  -v <APP_DATA>:/app \
  -e LLM_BASE_URL=http://<llm host>:11434/v1 -e LLM_MODEL=<model name> \
  -v <this folder>:/mv \
  epub_to_audiobook:local python3 /mv/validate_multivoice.py --out /mv/validate_out
```

`<llm host>` must resolve from inside the container: the LLM container's name on the `tts` network, or
`host.docker.internal` with `--add-host=host.docker.internal:host-gateway` for a server on the host.
`-e LLM_UNLOAD_CHATTERBOX=off` keeps Chatterbox loaded during the pass; `--no-sample` skips the
narration. Output: `validate_out/attribution.json` (every prediction), `validate_out/sample_book/*.mp3`
(the sample), `validate_out/sample_cast.json` (the cast it used).

## Cast benchmark (private books)

Behind `WORKLOG.md` §41-48. Real books are scored against source-linked answer keys that hold the
books' text, so the keys, each book's alias file and every run stay outside the repository, in the
stack's `data/diagnostics/<book>/` folders. Three tools:

- `evaluate_cast.py`: one live run of the production cast analysis on chosen chapters. It saves the
  cast, every request and reply (`requests.jsonl`) and `cast.manifest.json`: the commit and a hash of
  the app's code, the model and its digest, the input's hash, the settings, the answer key's hash
  (`--reference`), the call count and how long it took.
- `replay_cast.py --saved-run <run> --out <new folder>`: the current code on a saved run, every
  request answered with the reply the model gave to that same request. Only the code differs, so one
  replay is enough for a change that asks the model nothing new (narrator choice, reconciliation).
  First check that the code the run was made with reproduces its scores; a change to prompts or
  attribution windows asks new questions and the replay stops at the first one.
- `tests/audiobook_generator/cast_audit_eval.py <cast.json> --reference <key> --aliases <book's
  aliases.json> --summary`: wrong speakers, unresolved lines, identity splits, false merges, wrong
  chapter narrators and wrong voice routes, each on its own.

Prompt changes move the local model's errors around: single live runs of the same chapters differ
by about 15 lines in 900. Judge them over at least three live runs per variant, on every book.
