# F-45 build brief: compiled T3 token loop for Chatterbox

You are implementing review finding F-45 in `cporcellijr/epub-to-audiobook-chatterbox`. The idea was
measured on the owner's GPU and is worth building; your job is to turn the prototype into production
code. You have no GPU, so you write the code, the unit tests and a GPU validation script. The owner's
local session runs that script on the real machine, and turns the feature on only if it passes.

## 1. Goal

Chatterbox spends 92% of each request generating speech tokens one at a time, and most of that time
is CPU-side launch overhead, not GPU work. Running the per-token transformer step from a static KV
cache through `torch.compile(mode="reduce-overhead")` (CUDA graphs) removes that overhead. Build it
for the **Original English model only**, behind a setting that is **off by default**, with an
automatic fallback to today's code path whenever anything about it doesn't apply or fails.

## 2. Evidence (measured 2026-09-28, RTX 4070 12 GB, Docker Desktop on WSL2)

Production engine (BF16, autocast, voice cache), voice `Elena.wav`, the book settings: seed 888,
temperature 0.61, exaggeration 0.73, cfg_weight 0.5, repetition_penalty 1.2, min_p 0.05, top_p 1.0.

| | Stock (today) | Prototype (compiled) |
|---|---|---|
| Time per speech token | 20-25 ms | 7.7-8.3 ms |
| Whole requests (`engine.synthesize`, 3 texts x 3 runs) | 44.1 s for 77.3 s of audio: 1.75x real time | 18.6 s for 76.6 s: **4.12x real time (2.35x faster)** |
| One-time compile + CUDA-graph recording | - | 16.9 s |
| Peak VRAM of the test process (model included) | - | 3.2 GB |

- Baseline breakdown of one 241-token request: token loop 5.09 s, S3Gen audio decode 0.44 s, total 5.55 s.
- **Same model.** The stock token sequence was forced through both paths and the next-token
  distributions compared at every step: the compiled path's top token matched stock at 94-97% of
  steps, was always in stock's top 5, and the mean KL was 0.0017. An uncompiled run on the same static
  cache differs from stock by the same amount, so the difference is bf16 rounding from the cache
  layout, not compilation. Sampled output drifts from stock after a few tokens (same distribution,
  different draws).
- **Blind listening by the owner** (3 A/B pairs of the same passages): could not tell which was which,
  and afterwards judged the compiled take better on 2 of 3.

The prototype and the measurement scripts are in `docs/chatterbox-edition/experiments/f45/`
(`f45lib.py` is the prototype; read it first). Background: `WORKLOG.md` sections 2.2 and 12,
`REVIEW_FINDINGS.md` F-45.

## 3. Where things are

- `chatterbox/` is devnen/Chatterbox-TTS-Server as a git subtree plus local patches (`git log -- chatterbox`).
  Keep changes to upstream files small; put new logic in a new module.
- `chatterbox/engine.py` loads the model and runs synthesis: `load_model()`, `unload_model()`,
  `reload_model()`, `synthesize()`, `BF16_ENABLED` (from `TTS_BF16`), `loaded_model_type`
  (`"original"`, `"turbo"`, `"multilingual"`), the voice-conditionals cache, `_synthesis_lock`
  (an RLock around generation and model load/unload/reload), and `torch.clear_autocast_cache()` after
  every generation (a memory-leak fix: keep it).
- `chatterbox/server.py` runs synthesis on FastAPI's threadpool since F-10: `/v1/audio/speech` is a plain
  `def` endpoint, and `/tts` uses `run_in_executor`. Requests therefore arrive on many different threads.
- The model code is the pip package `chatterbox-v2`, installed in `chatterbox/Dockerfile` from
  `github.com/devnen/chatterbox-v2` pinned to commit `cc0357396d9c73fc1e6c544ee40bb596020edd09`
  (branch `master`). **Read that commit, not the repository's default branch, which is a different
  line of history.** The token loop is `T3.inference` in `chatterbox/models/t3/t3.py`. `ChatterboxTTS.generate` in
  `chatterbox/tts.py` calls `self.t3.inference(t3_cond=..., text_tokens=..., max_new_tokens=1000,
  temperature=..., cfg_weight=..., repetition_penalty=..., min_p=..., top_p=...)` and takes `[0]` of the result.
  `chatterbox/patches/apply_speed_patches.py` edits the installed package at build time; don't add to
  it for this feature.
- Image versions: Python 3.10, torch 2.5.1+cu121 (with Triton), transformers 4.46.3, and build tools
  are present (inductor needs a C compiler). Backbone: `LlamaModel`, 30 layers, hidden size 1024,
  SDPA attention, weights in bfloat16 when `TTS_BF16` is on.

## 4. What to build

1. **A new module, e.g. `chatterbox/fast_t3.py`,** holding the compiled decoder for one loaded T3 model:
   - A static KV cache: transformers `StaticCache` for batch 2 (the CFG pair), 2,048 positions,
     the backbone's dtype.
   - Static input buffers for one step: embeddings `[2, 1, hidden]` and `cache_position` `[1]`.
   - The per-token step: backbone forward with `past_key_values=<static cache>` and
     `cache_position=<buffer>`, then `speech_head` on the last hidden state, returning `[2, vocab]` logits.
     Wrap it with `torch.compile(..., mode="reduce-overhead", fullgraph=True)`.
   - An `inference(...)` with **the same signature and return value as `T3.inference`**
     (`[1, N]` tokens, the EOS token included as the last element when reached, `max_new_tokens`
     honoured). Prefill the prompt eagerly into the static cache (`cache_position = arange(P)`), then
     loop: sampling exactly as the stock loop does (CFG combine, repetition penalty, temperature,
     min_p, top_p, softmax, multinomial, EOS check), next embedding = `speech_emb(token) +
     speech_pos_emb.get_fixed_embedding(i + 1)` duplicated for the CFG pair, copied into the static
     buffers, then the compiled step. `.clone()` the step's output before the next call (CUDA-graph
     outputs are overwritten on replay). `reset()` the cache at the start of every request. Use the
     parameters passed in, not constants (the prototype hard-codes them).
   - A fallback to the original `T3.inference` whenever the fast path doesn't apply: batch size is not 2
     (e.g. `cfg_weight == 0`), prompt length + `max_new_tokens` exceeds the cache, CUDA not in use, an
     alignment stream analyzer is needed (multilingual), or the fast path raised. Log the reason at DEBUG
     per request; log a WARNING with `exc_info=True` the first time it fails.
   - `close()` to drop the compiled function, graphs and cache (for unload and model swaps).
2. **One dedicated synthesis thread.** torch.compile's CUDA graphs are recorded and replayed per thread:
   `torch/_inductor/cudagraph_trees.py` keeps its tree managers in a `threading.local`. Without a
   dedicated thread, every FastAPI worker thread records its own graphs, costing extra memory and slow
   first calls. Run warm-up, every generation and teardown on one worker thread. A single-thread
   executor owned by the engine, with `synthesize()` submitting to it and waiting on the result, is
   one way. Keep `_synthesis_lock`'s guarantees: one generation at a time, and no load/unload/reload
   during a generation.
3. **Hooks in `engine.py`, kept small.**
   - After a successful `load_model()`, install the fast path only when the setting is on,
     `loaded_model_type == "original"`, `BF16_ENABLED`, and the device is CUDA.
   - Warm up on the worker thread with one short generation using the model's built-in default
     conditionals (`ChatterboxTTS.from_pretrained` loads them), inside the same contexts as real
     requests: `torch.inference_mode()` and the same `torch.autocast` settings. This way the first real
     request doesn't pay for compiling.
   - Log one INFO line with the outcome and the warm-up time.
   - Tear down in `unload_model()` and `reload_model()`, then reinstall after a reload if it still applies.
   - Any failure while installing or warming up: log a WARNING with `exc_info=True`, restore the stock
     loop, keep serving.
4. **The setting: `TTS_COMPILE=on|off`** (default `off`), read when the model loads.
   - Pass it through `docker-compose.chatterbox.yml` as `${TTS_COMPILE:-off}` and document it in
     `.env.example`, `README.md` and `chatterbox/documentation.md`.
   - Model it on how `TTS_BF16` is handled.

## 5. Traps found while prototyping

1. **Recompiles.** Every call to the stock `T3.inference` assigns `self.patched_model`, which registers a
   new submodule on `t3`. A compiled function that closes over `t3` has a dynamo guard on
   `len(t3._modules)`, so any stock call (the fallback) forces a recompile (seen as
   `len(G['t3']._modules) == 8` failing). Compile a small dedicated `nn.Module` or function that
   references only the backbone and `speech_head`, never `t3` itself.
2. **Contexts are part of the guards.** Compile and replay under the same inference mode and autocast
   state, or dynamo recompiles.
3. **Cache length.** The prompt is the conditioning (about 34 positions) plus the text tokens plus BOS.
   The app sends at most about 450 characters per request, and `/v1` chunks at 500. With 1,000 new
   tokens that fits in 2,048 with room to spare; check anyway and fall back if it doesn't fit.
4. **Memory.** The static cache is about 0.5 GB (30 layers x K/V x 2 x 16 heads x 2,048 x 64 x
   bf16), plus the CUDA-graph pool. The GPU also runs Kokoro (about 1.2 GB) and other services.
5. **Seeds.** `set_seed` still applies, but for the same seed the output differs from the stock loop
   (different rounding, same distribution). That's expected; don't try to make them identical.
6. **Leak fix.** `torch.clear_autocast_cache()` must still run after every generation.

## 6. Tests you can run without a GPU

The existing `chatterbox/tests` run inside the Chatterbox image. Your environment may not have torch
at all. Keep the decision logic in plain functions that import torch lazily, and unit-test those with
mocks: setting parsing, the install gate (model type, BF16, device), the fallback decision (batch
size, cache length, analyzer), fallback on a raised exception, restoring the stock loop, and the
worker thread (every call runs on the same thread, one at a time, and load/unload waits for a running
generation). Report which tests you could run and which need the image or a GPU.

## 7. GPU validation script (you write it, the owner's machine runs it)

Add `docs/chatterbox-edition/experiments/f45/validate_build.py`, runnable the same way as the other
scripts there (see that folder's README). It must exercise **your implementation through
`engine.load_model()` / `engine.synthesize()` with `TTS_COMPILE=on`**, not the prototype, and print
PASS or FAIL per check:

| Check | Pass when |
|---|---|
| Speed: the three passages from `f45lib.py`, 3 runs each, compile on vs off | on is at least 2.0x faster in real-time factor (prototype: 2.35x) |
| Same model: teacher-forced comparison as in `verify.py`, through your decoder | argmax agreement >= 0.90, compiled argmax always in stock top 5, mean KL <= 0.005 |
| No recompiles: 50 requests of varied lengths after warm-up, with `TORCH_LOGS=recompiles` | none logged |
| Threads: requests submitted from 8 different threads of a thread pool, one at a time | no latency spike after the first request, none of them re-records graphs |
| Memory: 240 requests; count live CUDA tensors (`gc.get_objects()` filtered to CUDA `torch.Tensor`s) and `torch.cuda.memory_allocated()` before and after, the method that found the autocast leak (`WORKLOG.md` section 2.3) | both flat |
| Fallback: force the compiled step to raise | request still succeeds on the stock loop; one WARNING logged |
| Start-up: load plus warm-up time | logged; well inside the compose healthcheck's 5-minute start period |
| Listening | writes stock and compiled WAVs of the three passages for the owner to compare |

Keep it within about 60,000 characters of synthesis in total (the 240-request memory check is most of it).

## 8. Out of scope

- The app's time estimates (`GENERATION_SPEED = 1.8` in `audiobook_generator/ui/chatterbox_ui.py`).
  They get updated after validation, from a real chapter.
- Compiling the sampling ops (F-60), the Turbo and multilingual paths, and S3Gen (F-46, F-47).
- Anything outside `chatterbox/`, the compose file, `.env.example` and the docs named above.

## 9. Delivery

- Work on branch `feature/f45-compiled-t3` from `main`, push the branch, and do not merge or open a PR.
  The owner's local session reviews it, runs the validation on the GPU, and merges.
- Commits: `git -c user.name="Black Cat Media Dev" -c user.email="cporcellijr@gmail.com" commit`, with
  subjects like `Chatterbox: ...`. No Co-Authored-By or other trailers.
- Match the surrounding style: type hints and a short docstring on new functions, comments only where
  the reason isn't obvious. Shell scripts stay LF (`.gitattributes` already enforces it).
- Update `WORKLOG.md` section 12 and the F-45 row of the status table in `REVIEW_FINDINGS.md` to
  "built on branch, awaiting GPU validation".
- Never put real book titles or authors anywhere; the owner's library is private.
- Report back with: files changed and why, design decisions, tests run and their results, what you
  could not verify, and the exact command to run `validate_build.py`.
