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
