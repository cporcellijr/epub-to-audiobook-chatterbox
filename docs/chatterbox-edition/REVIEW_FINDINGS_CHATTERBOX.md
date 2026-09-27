# Addendum: the rest of `chatterbox/` and the model package

Continues the numbering of `REVIEW_FINDINGS.md` (F-49 onward) so it can be merged
into that file. Same method: four junior packages (WP8 server endpoints, WP9 config/utilities/launchers/
docs, WP10 browser UI, WP11 model package), every finding re-checked by the PM against the code and
reproduced where cheap. The owner uses the Original English model only; Turbo and multilingual paths
were not reviewed except where a default trips a fresh deployment.

Sources read for this pass, beyond the subtree: a clone of the pip-installed model package
`devnen/chatterbox-v2` at its current `master` (`df73ac8`, 2025-05-29, and the last commit on that
branch), a clone of upstream `resemble-ai/chatterbox` (`5de7a54`, 2026-07-21) for comparison, and a
clone of `devnen/Chatterbox-TTS-Server` `main`, which is byte-identical to the subtree import
(`915ae28`), so a `git subtree pull` today brings nothing.

## 1. What matters most

| ID | What | Why | Effort |
|---|---|---|---|
| F-49 | The build-time speed patch is a no-op: its `is not None` test is always true, so attention weights are still requested for all 30 layers on every token | The WORKLOG's own A/B measured +13 to 23% tokens/s for this knob, which is 9 to 17% of total time (roughly 1.5 to 3 hours per 10-hour book) that the production image is very likely not getting | S |
| F-50 | Every request adds one more forward hook and one more `forward` wrapper to attention layer 9, never removed; each hook copies attention weights to the CPU on every token step, and nothing reads the result | Per-token cost that grows with the age of the server process; mechanism certain, magnitude unmeasured; the fix is what upstream Resemble has since done | M |
| F-51 | A fresh deployment loads no model at all: the checked-in config selects Turbo, the frozen package has no Turbo class, `load_model` swallows the ImportError, every request gets 503 | Sharpens F-03 from "different model" to "nothing works"; same one-file fix | S |
| F-53 | The Chatterbox web page's own Save posts all six delivery values from a form filled at page open, reverting the app's "Save for books" | Explains any "my settings came back" surprise if that tab is left open | S |
| F-57 | The S3Gen decoder runs 10 CFG-doubled Euler steps, not the 2 the engine comment claims; fewer steps is the one cheap knob on the 0.35 s fixed cost | About 7 to 21 minutes per 10-hour book if quality holds; needs a listening A/B | S |

## 2. Corrections to the committed report

- **F-03** becomes Critical for a fresh deployment (see F-51). Your live config is unaffected.
- **F-22** (unpinned `chatterbox-v2@master`): that branch has not moved since 2025-05-29, so pinning to
  `df73ac8` changes nothing today and is free insurance. Upstream Resemble is far ahead (Turbo,
  multilingual, `output_attentions=False`, analyzer removed for English) but pins `transformers==5.2.0`
  and `torch==2.6.0` against this server's 4.46.3 and 2.5.1, so switching packages is an L, not an S.
- **F-24** (single-file bind mount becomes a directory): also crashes the app container, because
  `build_ui` calls `read_saved_settings()` at start-up with no error handling, and `open()` on a
  directory raises. And on the server side `shutil.copy2` onto a directory silently writes
  `config.yaml.tmp` inside it and logs success (reproduced). Medium-High for a first run.
- **F-10** (`/v1/audio/speech` blocks the event loop): `/tts` does the same (`server.py:1074`), which is
  why a Voice lab preview and a running book stall each other; the streaming branch two call sites
  above already uses `run_in_executor`.
- **F-45** (`torch.compile`): the wrapper persistence it needs is blocked by the per-call analyzer in
  F-50; do F-50 first. The engine comment quoted in F-45 is confirmed correct.

## 3. Findings

### Priority 1

| Field | Content |
|---|---|
| ID | F-49 |
| Title | The build-time speed patch evaluates to `True` and changes nothing |
| Area | WP11 |
| Type | performance |
| Severity | High |
| Evidence | `chatterbox/patches/apply_speed_patches.py:19-21` replaces `output_attentions=True,` with `output_attentions=self.patched_model.alignment_stream_analyzer is not None,`; `chatterbox-v2 src/chatterbox/models/t3/t3.py:252-272` constructs an `AlignmentStreamAnalyzer` unconditionally on every call and passes it to the backend (`:269`), so the attribute is never `None`; `t3.py:316, 363` the two call sites |
| Problem or opportunity | The patch's premise ("only the multilingual model attaches an analyzer") does not hold for this package: the analyzer is built for every model on every call. The patched expression is therefore always `True`, behaviourally identical to the unpatched code, and the transformer runs eager attention on all layers for every token. The attention weights are never consumed (`t3_hf_backend.py:109` is commented out). The WORKLOG measured +13 to 23% tokens/s for this knob "in-process"; if that A/B used a literal `False`, the production image does not have the gain. |
| Proposed change | Change the replacement string in `apply_speed_patches.py` to the literal `output_attentions=False,` (one line in a file you own; the `expected=2` guard still applies). Layer 9 keeps eager attention through the analyzer's own hook until F-50 is done. |
| Benefit | 9 to 17% of total generation time, about 88 to 181 min per 10-hour book (derived from the WORKLOG's A/B; assumes it reproduces) |
| Effort | S |
| Risk | None: the weights being switched off are unread. |
| Confidence | verified (expression always true); benefit likely |
| Checked by PM | Read `t3.py:252-272` and `t3_hf_backend.py:33,109` in the clone; confirmed `output_attentions=True,` occurs exactly twice and the fork's only `t3.py` commit is the 2025-05-28 import. Upstream Resemble now has `output_attentions=False` at both sites. |

| Field | Content |
|---|---|
| ID | F-50 |
| Title | Forward hooks and `forward` wrappers accumulate on attention layer 9 for the life of the process |
| Area | WP11 |
| Type | performance / robustness |
| Severity | High (mechanism certain; magnitude unmeasured) |
| Evidence | `t3.py:252` `self.compiled = False` immediately before `:256 if not self.compiled:`; `alignment_stream_analyzer.py:57` calls `_add_attention_spy` in `__init__`; `:78` `register_forward_hook(...)` with the handle discarded; `:81-87` wraps `target_layer.forward` around whatever it already was (`# TODO: how to unpatch it?`); `:74` the hook does `output[1].cpu()` every step |
| Problem or opportunity | After N requests, layer 9 carries N hooks and N nested wrappers. Every token step then performs N device-to-host copies of the attention tensor (a synchronisation each) plus N Python frames, none of which is used. Each analyzer object also stays alive through its hook closure. A book is thousands of requests, so per-token overhead grows across the run and resets only on restart. This fits the WORKLOG's "GPU at 50 to 60%, kernel-launch bound" picture and its earlier host-RAM growth, but neither is proven here. |
| Proposed change | Build-time block patch to `t3.py` (about ten lines, same mechanism as today): do not construct the analyzer, pass `alignment_stream_analyzer=None`, and drop the `self.compiled = False` reset so the backend wrapper is built once. This is exactly the shape of upstream Resemble's current English path. Together with F-49 every layer runs SDPA. |
| Benefit | Unknown; possibly the largest item in either report if throughput does degrade over a long run |
| Effort | M |
| Risk | Removes the (already inert) hallucination guard; a multi-line exact-string patch is more fragile than a one-liner, so keep the `expected` guard. |
| Confidence | likely |
| Checked by PM | Read the constructor and hook; `nn.Module.register_forward_hook` appends, and the module (`T3.tfmr`) lives as long as the loaded model. How to check on the GPU: log `len(model.t3.tfmr.layers[9].self_attn._forward_hooks)` after N requests, and tokens/s at request 1, 300 and 3000 without a restart. |

| Field | Content |
|---|---|
| ID | F-51 |
| Title | A fresh deployment from the repo loads no model |
| Area | WP7 / WP11 |
| Type | deployment |
| Severity | Critical for a fresh deployment; no effect on the running one |
| Evidence | `chatterbox/config.yaml:12` `repo_id: chatterbox-turbo`; `chatterbox/Dockerfile:45` installs `devnen/chatterbox-v2@master`, whose `src/chatterbox/` has only `tts.py` and `vc.py` (no `tts_turbo.py`, no `mtl_tts.py`); `chatterbox/engine.py:19-31` sets `TURBO_AVAILABLE = False` on ImportError; `:200-207` `_get_model_class` raises; `load_model` catches it and returns False; `lifespan` logs CRITICAL and keeps serving |
| Problem or opportunity | Following the README's "copy `chatterbox/config.yaml`" step yields a server that answers every synthesis request with 503, while the app's queue starts the first book and marks chapter after chapter failed (F-02). The WORKLOG mentions Turbo behaviour, so the owner's image or config must differ from the repo; the repo as committed cannot run Turbo at all. |
| Proposed change | Same as F-03: commit the Original-model config as the template. Also state in the README that this Dockerfile's package provides the Original model only. |
| Benefit | The repo reproduces the working deployment. |
| Effort | S |
| Risk | None. |
| Confidence | verified (imports and control flow); not run |
| Checked by PM | Listed the clone's modules; read the import guards and `_get_model_class`. |

| Field | Content |
|---|---|
| ID | F-52 |
| Title | The config file is rewritten non-atomically while the app reads it, and the browser UI rewrites it 750 ms after every keystroke |
| Area | WP9 / WP10 |
| Type | robustness |
| Severity | Medium |
| Evidence | `chatterbox/config.py:407` `shutil.copy2(temp_file, CONFIG_FILE_PATH)` (truncate, then fill); `chatterbox/ui/script.js:35` `DEBOUNCE_DELAY_MS = 750`, `:257-293` `saveCurrentUiState` posts `ui_state` on every form change and `update_and_save` re-serialises the whole file; `audiobook_generator/ui/chatterbox_ui.py:88-89` `yaml.safe_load(f) or {}` with no retry |
| Problem or opportunity | A reader that opens the file during the truncate-and-fill window gets an empty document and the app silently shows its fallback sliders (0.5 / 0.5 / 0.8) instead of the saved values. WP9 measured about 20% torn reads under continuous writes at the real file size. In practice it needs someone typing in the Chatterbox page while the app loads its Voice lab, so it is rare; when it happens it is invisible. |
| Proposed change | The server copies instead of renaming because `config.yaml` is a single-file bind mount and a rename over a mount point fails with "device or resource busy". Fix it at the deployment layer: mount the data folder (or a `config/` subfolder) instead of the file, then a one-line `os.replace` in `config.py` is safe and F-24 disappears too. App side: treat an empty or unparsable read as "retry once, then warn", not as defaults. |
| Benefit | No silent fallback; removes the first-run directory trap as well. |
| Effort | S (compose) + S (config.py) + S (app) |
| Risk | The server reads `config.yaml` from its working directory; moving it needs `CONFIG_FILE_PATH` to follow (`config.py:20`). |
| Confidence | verified (write path); torn-read rate measured by WP9 on this machine |
| Checked by PM | Read `_save_config_yaml_internal`; confirmed no `os.replace`; confirmed the app's read has no error handling. The `os.replace` proposal in WP9's report was corrected for the mount-point case. |

| Field | Content |
|---|---|
| ID | F-53 |
| Title | The Chatterbox page's Save posts six stale values and reverts the app's "Save for books" |
| Area | WP10 |
| Type | UX / bug |
| Severity | Medium |
| Evidence | `chatterbox/ui/script.js:1315-1326` posts `generation_defaults` with `temperature, exaggeration, cfg_weight, speed_factor, seed, language` read from the form; the form is filled once from `/api/ui/initial-data` at page load (`:683-692`); `chatterbox/config.py:133-151` deep-merges, so the app's three-key save is otherwise safe |
| Problem or opportunity | If the Chatterbox page is open while you use the app's Voice lab, clicking its Save afterwards writes back the values it loaded earlier, including a possibly different seed. The app's save itself is fine (the merge keeps `seed`, `speed_factor`, `language`; reproduced). |
| Proposed change | Either avoid the Chatterbox page for delivery settings (documentation), or have its Save re-fetch current values first (a few lines in `script.js`, upstream-merge cost). |
| Benefit | No silent reversions. |
| Effort | S |
| Risk | None. |
| Confidence | verified |
| Checked by PM | Read the handler and the merge function; ran the merge with the app's payload. |

### Priority 2

| Field | Content |
|---|---|
| ID | F-54 |
| Title | `/tts` blocks the event loop like `/v1`; previews and books stall each other |
| Area | WP8 |
| Type | performance / UX |
| Severity | Medium |
| Evidence | `chatterbox/server.py:1074` blocking `engine.synthesize(...)` inside `async def custom_tts_endpoint`; `:1316` blocking `encode_audio` (an ffmpeg subprocess for mp3); the streaming branch at `:1018-1020` already uses `run_in_executor` |
| Problem or opportunity | A Voice lab preview (`/tts`, mp3) cannot start until the current book request returns, and a book request cannot start until the preview finishes; both sit on one event loop with `workers=1`. |
| Proposed change | Same change as F-10, applied to both endpoints, plus one `threading.Lock` around `generate()` in `engine.py` so GPU work stays serial. |
| Benefit | Health and settings endpoints respond during generation; the healthcheck in F-02 works; previews interleave at chunk granularity. |
| Effort | S-M |
| Risk | Same as F-10. |
| Confidence | verified (code) |
| Checked by PM | Read both call sites. |

| Field | Content |
|---|---|
| ID | F-55 |
| Title | Wide-open CORS with credentials on an unauthenticated server; `use_auth` is a dead setting |
| Area | WP8 |
| Type | security |
| Severity | Medium (LAN-only, per K12) |
| Evidence | `chatterbox/server.py:192-199` `allow_origins=["*", "null"], allow_credentials=True`; `/reset_settings` (`:550`), `/restart_server` (`:576`), `/api/unload` (`:611`), uploads (`:670, 754`) have no auth; `chatterbox/config.py:42` `use_auth` is referenced nowhere else; `/api/ui/initial-data` (`:450-491`) returns the whole config including `server.auth_password` |
| Problem or opportunity | Any web page rendered by a browser on the LAN can POST to these endpoints (Starlette echoes the request origin when the wildcard is combined with credentials). Setting `use_auth: true` changes nothing. The password field is only the shipped default today, but it is served to any caller. |
| Proposed change | Deployment layer first: bind Chatterbox to the Docker network only (drop the host port, or bind it to 127.0.0.1) since only the app and the reader app need it. In `chatterbox/`: narrow `allow_origins` to the app's origin, redact `server.*` from the initial-data response, log a start-up warning if `use_auth` is set. |
| Benefit | Removes a drive-by reset/unload path and a credential echo. |
| Effort | S |
| Risk | The Chatterbox page must stay reachable from wherever you use it. |
| Confidence | verified (code) |
| Checked by PM | Read the middleware block, the endpoints and the grep for `use_auth`. |

| Field | Content |
|---|---|
| ID | F-56 |
| Title | The 1000-token cap is hard-coded and hitting it leaves no trace |
| Area | WP11 |
| Type | robustness / audio quality |
| Severity | Medium |
| Evidence | `chatterbox-v2 src/chatterbox/tts.py:243` `max_new_tokens=1000,  # TODO: use the value in config`; `t3.py:324-372` returns the same way whether EOS was sampled (`:349-350`) or the loop ran out; `generate()`'s signature (`tts.py:207-214`) does not expose the cap |
| Problem or opportunity | This is the mechanism behind the WORKLOG's "4 of 629 requests lost the end of the text" and behind F-07. The server cannot raise the cap per request and cannot tell a clean stop from a cut-off. |
| Proposed change | One-line build-time patch after `tts.py:246`: log a warning (and, better, set an attribute the server can read) when `speech_tokens.shape[-1] >= 999`; the server can then return a 4xx or a header so the app retries with a split unit. F-07 (split long units client-side) remains the primary defence. |
| Benefit | Truncation becomes detectable instead of silent. |
| Effort | S (log) / M (signal through the server to the client) |
| Risk | None for the log. |
| Confidence | verified |
| Checked by PM | Read the loop and `generate()`. |

| Field | Content |
|---|---|
| ID | F-57 |
| Title | The S3Gen decoder runs 10 CFG-doubled Euler steps, not the 2 the engine comment states; fewer steps is the one cheap knob on the fixed per-request cost |
| Area | WP11 |
| Type | performance / maintainability |
| Severity | Medium |
| Evidence | `chatterbox/engine.py:372-373` "S3Gen ... runs only 2 CFM timesteps"; `chatterbox-v2 src/chatterbox/models/s3gen/flow.py:146, 238` `n_timesteps=10`; `flow_matching.py:105-111` each step runs the estimator on a batch of 2; `hifigan.py:463` the vocoder is a single pass |
| Problem or opportunity | The comment's reasoning for keeping S3Gen in fp32 is wrong by 5× (the dtype-mismatch reason it also gives may still hold). The 10-step CFM solve is very likely the bulk of the ~0.35 s fixed cost per request that F-05 and K7 are about. |
| Proposed change | Fix the comment. Then a one-line build-time patch trying `n_timesteps=6` or `8`, judged by a blind listening A/B on several voices. |
| Benefit | About 0.1 s per request if the CFM is 70 to 90% of the fixed cost: 7 to 21 min per 10-hour book (derived with stated assumptions); zero if quality suffers |
| Effort | S |
| Risk | Real quality risk: this is a generative sampling knob, not a plumbing one. |
| Confidence | verified (values); speculative (gain and quality) |
| Checked by PM | Read the engine comment and the flow code; confirmed the literal at both sites. |

| Field | Content |
|---|---|
| ID | F-58 |
| Title | The analyzer's early-EOS and repetition guards are dead code |
| Area | WP11 |
| Type | audio quality |
| Severity | Low-Medium (awareness) |
| Evidence | `alignment_stream_analyzer.py:89-154` `step()` forces or suppresses EOS from the attention alignment; its only call site `t3_hf_backend.py:109` is commented out |
| Problem or opportunity | The package pays for the analyzer (F-50) without using it. The only defences against runaway or premature stops are `repetition_penalty=2.0` and the 1000-token cap. Upstream Resemble removed the analyzer from the English path rather than re-enabling it. |
| Proposed change | None beyond F-50 (remove it). If early cut-offs are ever heard on short lines, re-enabling `step()` is a 5 to 10 line patch that changes audio and needs listening tests. |
| Benefit | Awareness only. |
| Effort | — |
| Risk | — |
| Confidence | verified |
| Checked by PM | Read both files. |

### Priority 3 (compact)

| ID | Title | Area | Evidence | Problem → proposed change | Effort | Confidence |
|---|---|---|---|---|---|---|
| F-59 | Per-call host syncs on the cached-voice path | WP11 | `tts.py:221` compares a Python float with a CUDA scalar every call; `cond_enc.py:28` `.item()` per tensor when exaggeration changes; `tts.py:258-259` full waveform to CPU then the Perth watermark on CPU | Sub-millisecond for the first two; the watermark is by design (README: survives compression) and should stay → cache the float; leave the watermark | S | verified (mechanism), speculative (size) |
| F-60 | Hand-rolled sampling loop does 8 to 10 small eager ops per token, plus `tqdm` | WP11 | `t3.py:324-368` | Plausible contributor to "kernel-launch bound"; a refactor is real effort in the hottest loop → profile with `torch.profiler` first; only act if CPU dispatch dominates | M | speculative (3 to 5% of total) |
| F-61 | Partial voice upload left in place and listed as a voice | WP8 | `server.py:707-709, 726-729` and `:793-795, 812-815` no `unlink` in the `except` | A mid-write failure leaves a broken `.wav` in the picker → unlink on failure | S | verified |
| F-62 | `/restart_server` and `/api/unload` race a streaming `/tts` | WP8 | `engine.py:546-547, 594-595` swap the global with no lock; streaming yields between chunks | Crash or truncated stream; the app never streams or unloads → lock in `engine.py` (same lock as F-54) | S | likely |
| F-63 | `documentation.md`, `models.py` and the browser UI don't know about `aac`/`flac` | WP9 / WP10 | `documentation.md:700` lists wav/opus; `models.py:64` Literal lacks aac/flac; `index.html:319-323` offers wav/mp3/opus; `index.html:149-151` vs `:454-457` give two chunk-size ranges and the slider default is 120 | Local patch not reflected in the subtree's own docs and UI → update three places | S | verified |
| F-64 | Shipped `ui_state` is a leftover demo session | WP9 / WP10 | `config.yaml:33-62` | Copied into every new deployment → reset to `DEFAULT_CONFIG["ui_state"]` | S | verified |
| F-65 | `download_model.py` cannot pre-download the configured model | WP9 | `download_model.py:30-36, 51, 71` passes the alias `chatterbox-turbo` to `hf_hub_download`; `engine.py:269-273` bypasses `paths.model_cache` | Misleading utility → delete or route through `_get_model_class` | S | verified (static) |
| F-66 | Dead code and unused launchers in the subtree | WP9 | `utils.py:392, 451` `save_audio_to_file`/`save_audio_tensor_to_file` unreferenced; `start.py`/`start.sh`/`start.bat` (3,162 lines) unused by the Docker `CMD`; `reference_audio/Gianna.wav` duplicates `voices/Gianna.wav` | Note in the root README that the launchers are unused; leave upstream files alone otherwise | S | verified |
| F-67 | Stock voices are 7 to 9.7 s; the app advises 10 to 15 s and warns under 6 s | WP9 | `ffprobe` over `chatterbox/voices/`; `chatterbox_ui.py:49, 642` | Guidance and shipped samples disagree → lengthen samples you use, or soften the label | S | verified |
| F-68 | `Dockerfile.cu128` lacks the speed-patch step; `pyproject` declares torch 2.6 while 2.5.1 is installed with `--no-deps` | WP9 / WP11 | `Dockerfile.cu128` has no `patches/` line; `pyproject.toml:13`; `requirements-nvidia.txt:8` | Upstream says cu128 is for Blackwell; the 4070 stays on cu121. Only port the patch line if ever switching | S | verified |
| F-69 | `chatterbox/` has no tests, and its modules import torch at top level | WP8 / WP9 | `find chatterbox -iname "*test*"` empty; `utils.py:17-19`, `engine.py` imports | Config merge, save, `sanitize_filename`, `safe_resolve_within`, chunker: all untested → a small `chatterbox/tests/` for the pure-Python parts, extracted the way this review did | M | verified |

## 4. Considered and rejected in this pass

- `/tts` normalises peaks but `/v1` hard-clips (WP8): false; both endpoints scale peaks above 0.99 to 0.95 (`server.py:1261-1263` and `:1494-1495`). The three silence-processing passes exist only on `/tts`, but their config keys do not exist anywhere, so they are off on both.
- mp3 encoding spawns ffmpeg per `/v1` chunk (WP9): the app's paced path requests WAV (`openai_tts_provider.py:173`); only the CLI's non-paced path asks for mp3.
- `os.replace` in `_save_config_yaml_internal` (WP9): fails on the current single-file bind mount; fixed at the compose layer first (F-52).
- Switching the package to upstream Resemble now: pins `transformers==5.2.0`, `torch==2.6.0`, `librosa==0.11`; the server pins 4.46.3 and 2.5.1, and installs with `--no-deps`. Worth a separate, tested migration, not a patch.
- Removing the Perth watermark: one line, but against the package's stated purpose; not proposed.
- CFG with batch 1: already tried and rejected by the owner (WORKLOG 2.2); the package has no batch-1 path.
- Perceiver resampler re-run per call: 150 tokens, negligible.
- Per-request numpy work, `set_seed` CPU parts, resample no-op: noise (as in the main report).
- The Chatterbox UI's missing client-side upload size check (WP10): the endpoints validate on the server; personal LAN.
- Alternative Dockerfiles for ROCm, RDNA4, Strix Halo, CPU, Colab: irrelevant to this host; checked and left alone.

## 5. Open questions for the owner

1. and 2. are measurements, not memory: the earlier work was done by an automated agent, so nobody
   knows whether the +13 to 23% A/B used a literal `False` or whether throughput falls over a long
   run. Both fixes (F-49, F-50) are safe regardless, so apply them and measure the gain afterwards with
   the script in section 6.
3. Where did the Turbo observations in the WORKLOG come from, given this Dockerfile's package cannot load Turbo (F-51)? A different image, or upstream docs?
4. Do you still use the Chatterbox web page alongside the app (F-53, F-64), or only the app?
5. Would you run a blind listening A/B for `n_timesteps` 10 vs 8 vs 6 (F-57)?
6. Does anything besides the app and the reader app need Chatterbox's port on the host (F-55)?

## 6. How to measure F-49 and F-50

Neither fix needs a decision first: switching the patch to a literal `output_attentions=False`
disables weights nothing reads, and skipping the analyzer removes code whose only consumer is
commented out. The measurements below tell you what they were costing.

**Script.** `experiments/measure_t3_hooks.py` (in this folder) loads the model once inside the
Chatterbox container, generates the same sentence N times with a cached voice, and prints the number of
forward hooks on attention layer 9 and the throughput (seconds of audio per second of wall time) every
25 calls. Run it while no book is generating, because it loads a second copy of the model:

```
docker cp docs/chatterbox-edition/experiments/measure_t3_hooks.py chatterbox:/app/
docker exec -it chatterbox python3 /app/measure_t3_hooks.py --calls 200
```

What to look for on the **current** image:

- The hook count climbs by one per call (F-50 confirmed as a mechanism; it is, from the code).
- If the throughput percentage falls as the count climbs, that is the cost of F-50 per call. A book is
  thousands of calls, so even a 10% drop at call 200 matters.

Then rebuild with `patches/apply_speed_patches.py` changed as F-49 and F-50 propose, run the script
again, and compare the first line's throughput (F-49's gain) and whether the percentage stays flat
(F-50's fix).

**Without the script.** The model's sampling loop prints a `tqdm` progress line per generation to the
server's stderr (`Sampling: 250it [00:04, 55.2it/s]`), so `docker logs chatterbox` already contains
tokens/s per request. Compare the `it/s` figures shortly after a restart with those a few hundred
requests later; a downward drift is F-50.
