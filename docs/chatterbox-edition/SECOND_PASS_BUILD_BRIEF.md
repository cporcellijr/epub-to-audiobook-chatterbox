# Build brief: a trained 7B second pass for cast analysis

Status: **built 2026-10-19 (WORKLOG §81).** What changed from this plan while building:
- a guard: a change is kept only for someone already speaking in that chapter, or its narrator (Six Wakes
  chapter 28 lost 25 lines to a captain who wasn't in the scene);
- turn-taking and the gender check run again after the pass (another 14-20 lines a size on the novel);
- no Advanced checkbox and no `CAST_GUIDANCE_VERSION` bump: `CAST_SECOND_PASS_MODEL=` is the off switch, and
  the cast records `second_pass` itself; the fixed header lives in `cast_second_pass`, not `cast_llm.PROMPTS`.

Private evidence: `data/diagnostics/hybrid_2026-10-19/` (outside git: casts and logs hold book text).

## What it is

The 14B analyses the book as today. Then, with the finished cast held fixed, the trained 7B (round six, WORKLOG §69)
asks every first-pass window again under a closed list ("every speaker is one of these"). An answer that names
exactly one cast entry replaces the 14B's line; anything else keeps it. Tags, "I said" anchors and quoted terms
are never asked, exactly as in the first pass. Chapter narrators and the cast list are not changed.

Why it works (WORKLOG §68.1): the 14B is better at finding identities, the taught 7B at choosing among known
ones. The split gives each model only the job it is good at.

## Evidence so far

Such Sharp Teeth, the whole novel (3,304 lines), request sizes 16 / 20 / 24, deployed code:

| | Wrong + unresolved | Narrator lines lost | Off-voice |
|---|---|---|---|
| 14B today | 354 / 211 / 280 | 196 / 78 / 146 | 320 / 168 / 255 |
| + 14B second pass (control) | 245 / 215 / 221 | 78 / 70 / 84 | 208 / 169 / 184 |
| + 7B second pass | **151 / 156 / 167** | **40 / 23 / 36** | **111 / 108 / 127** |

- Holds on the 14 chapters the 7B never trained on (chapters 6 and 8 were in round four's data): 327/186/255 → 140/142/155.
- Serving check: the 7B scores 1,195 of 1,237 on §68.1's fixed-cast test (round six: 1,202).
- Cost: about 10 minutes for the 7B pass on top of the 14B's ~16 per novel.

## Gate before building (running)

The same test on Apex Prey 3, Six Wakes, You Like It Darker and Steamy Screams at all three sizes
(`run.py others`). Build only if no key gets worse on wrong + unresolved or off-voice beyond its own run-to-run
spread, and no chapter narrator changes. Anthologies are the risk: the roster spans every story.

## Design

1. **Model.** `qwen2.5-cast6:7b` in Ollama: the round-six adapter merged into Qwen2.5-7B-Instruct, Q8_0.
   Ollama 0.40.5 refuses adapters ("LoRA adapters are no longer supported"), so the merge is required.
   Steps, without Unsloth: `finetune6_2026-10-08/merge_lora.py` (torch + safetensors, shard by shard) in the
   `llama.cpp:full` image, `convert_hf_to_gguf.py` to F16, `llama-quantize` to Q8_0 (a 4-bit merge loses
   training), `ollama create` with qwen2.5:7b's template. Keep the GGUF, Modelfile and adapter under
   `C:/Server/stacks/local-ai/models/cast6/` so a lost Ollama volume can be rebuilt.
   **Parity gate:** through Ollama, the fixed-cast test ≥ 1,190 of 1,237, and the novel at size 20 within ±10 of
   the llama.cpp result.
2. **Code: `core/cast_second_pass.py`**, a port of `hybrid_2026-10-19/frozen.py`'s pure parts:
   - the roster: one entry per display name, most spoken first;
   - the resolver: display name (the entry speaking most in this chapter, else most spoken), else a unique
     alias or first/last name (narrowed to entries speaking in the chapter), "I"/"narrator" → the chapter's
     narrator;
   - `second_pass(cast, analysed, chat, log, save)` reuses the `ChapterText` list `analyse_book` already builds,
     so there's no reparsing. Per chapter it builds the same windows as `attribute_chapter`. A failed window
     keeps its lines, continued lines follow their first part, then `refresh_cast_counts`.
   - The fixed-list header becomes `PROMPTS["window_fixed_header"]` in `cast_llm` (no string replace).
3. **Wiring (`analyse_book`).** After `reconcile_lines` / `refresh_cast_counts` / `drop_empty_placeholders`,
   before `profile_cast`, so the profiles see the corrected lines. A second `ChatClient` on the same
   `LLM_BASE_URL` with the 7B. `OLLAMA_MAX_LOADED_MODELS=1` swaps 14B → 7B → 14B (two loads, ~20-30 s each)
   inside the job's existing `ollama` GPU reservation, so no coordinator change. Like profiles, a second-pass
   problem never fails the analysis: the 14B's lines stay and `cast["second_pass_error"]` says why.
4. **Settings (automatic by default, override under Advanced).** Compose env
   `CAST_SECOND_PASS_MODEL=qwen2.5-cast6:7b` (empty = off). At analysis start, `/api/tags` decides whether the
   model is there; if not, the analysis logs it once and skips. Advanced checkbox "Second pass with the trained
   model", on by default.
5. **UI.** Progress shows "⏳ Second pass: n of m chapters". The analysis time estimate grows by the pass's
   measured share. The cast records `second_pass: {model, asked, changed, unusable}` for audits.
6. **`CAST_GUIDANCE_VERSION` bump**, so casts made before can be told apart. No forced re-analysis.

## Tests

- Fake chat returning the base cast's own answers changes nothing (`test_frozen.py`'s check).
- The resolver: duplicate display names, an anthology's local preference, narrator words, ambiguous aliases.
- An unusable reply keeps the base lines; a missing model skips; an LLM error is noted, never raised.
- `analyse_book` with fakes: the pass runs between reconcile and profiles; profiles see corrected lines.
- UI progress text.

## Verification before deploy

- Production code path (not the diagnostic) on all five keys at 16/20/24; numbers match the diagnostic runs.
- The masked-tag set, for breadth (relative to today's 376-377).
- Monster Girl Islands keeps Nora (ch6) and Yuki (ch7); a few owner books for changed-line counts.

## Rollout and cleanup

- Deploy with the owner's OK. Existing casts stay until re-analysed. Rollback: `CAST_SECOND_PASS_MODEL=`.
- After the merge, delete what was only for building: the HF base (~15 GB), `qwen2.5:7b-instruct-q8_0` (8 GB)
  and the llama.cpp images (11 GB).
- After any Ollama update, rerun the parity gate (as for the 14B, memory "ollama-update-check").
- Implementation through the Omni bridge (production code and tests as separate calls). Full diff review,
  since this is cast code. WORKLOG section with the commit.
