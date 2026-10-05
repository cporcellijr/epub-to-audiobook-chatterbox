# Breeze TTS 2 server

A small FastAPI server around BreezeBlue's Breeze TTS 2 (3B) for the audiobook stack. It starts
idle with no model in memory and loads on the first request (or `POST /api/load`), so it can share
one GPU with Chatterbox and the LLM; the app unloads it with `POST /api/unload`. Weights live in the
`/models` volume (`/models/breeze-tts-2`) and are downloaded from Hugging Face on first load if missing.

## Endpoints

- `GET /health` -> `{"status": "ok", "loaded": bool, "max_batch": int}` (200 even when unloaded)
- `POST /api/load`, `POST /api/unload` -> `{"loaded": bool}` (idempotent; unload frees GPU memory)
- `POST /v1/batch` -> body `{"items": [{"id", "text", "voice", "ref_text", "instruction", "cfg_scale"}], "seed"}`,
  returns `{"items": [{"id", "wav_b64", "seconds", "error"}], "elapsed_s"}` in request order.
  `voice` is a bare file name in the voices folder and needs `ref_text` (clone). No voice plus an
  `instruction` is voice design. Items are grouped by template and cfg scale (1.0 for cloning, 4.0 with
  an instruction), sorted by length and run in chunks of `BREEZE_MAX_BATCH`. A chunk that runs out of
  GPU memory is split and retried; an item that still fails comes back with `error` set.

## Allocator faults (WSL)

Under WSL, with expandable segments on, a full card doesn't raise PyTorch's out-of-memory error. The
first failed growth is `CUDA driver error: device not ready`, and it leaves pages half-mapped until
the process ends. Every later growth over them is `!handles_.at(i) INTERNAL ASSERT FAILED at
.../CUDACachingAllocator.cpp` (pytorch#166234, #188008). The server treats both as out of memory and
splits the chunk, and logs `GPU allocator fault` once. Unloading the model doesn't clear the damage,
so the next `POST /api/unload` (the app sends one before the cast LLM runs) replies and then restarts
the server; compose's `restart: unless-stopped` brings it back within seconds. On 2026-10-05 a
chapter's 32 longest units lost all their takes this way, three attempts running, before this
handling existed.
- `POST /v1/audio/speech` -> `{"input", "voice", "ref_text", "instruction", "response_format": "wav"}`,
  returns a WAV (a batch of one).

WAV output is mono, 24 kHz, 16-bit. One generation runs at a time; other requests wait.

Each voice clip is encoded once and reused (keyed by path, size and modification time, so a replaced
clip is encoded again; cleared on unload). The pinned runtime would otherwise encode it for every
item, twice for a directed one: 35-50 ms each, 1-3 s of a 32-item batch.

The app sorts a chapter's units before batching (directed apart from plain, longest first), since a
chunk runs until its longest item finishes.

## Slowdown guard

A chunk of at least 3/4 of `BREEZE_MAX_BATCH` counts as full. When the median speed of the last 4
full chunks falls under `BREEZE_SLOW_RTF` (1.5x real time), the server unloads and loads the model
before its next request, logging `slowed down` and the GPU memory before and after. If speed stays
low, it logs `still slow` and doesn't reload again for 30 minutes. One book on 2026-10-01 ran at a
median 1.2x for five hours; healthy books never hit the threshold (WORKLOG §48).

## Environment

| Variable | Default |
| --- | --- |
| `BREEZE_MODEL_DIR` | `/models/breeze-tts-2` |
| `BREEZE_VOICES_DIR` | `/voices` |
| `BREEZE_MAX_BATCH` | `32` |
| `BREEZE_REPO_ID` | `BreezeBlue/Breeze-TTS-2` |
| `BREEZE_SLOW_RTF` | `1.5` (0 turns the slowdown guard off) |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True`, set by compose from `BREEZE_CUDA_ALLOC_CONF` (empty turns it off) |

Every `chunk of N` log line ends with the chunk's peak GPU memory and what PyTorch keeps reserved
afterwards. A reserve creeping far above the peaks over a long book means fragmentation; under WSL
it spills into shared system memory (Windows counter `\GPU Process Memory(*)\Shared Usage`).

## Speed (RTX 4070, eager attention, voice cloning)

| Batch | Real time |
| --- | --- |
| 1 | 0.39x |
| 4 | 1.38x |
| 8 | 1.88x |
| 16 | 3.94x |
| 32 | 7.14x (8.2 GiB peak) |

These are bake-off sentences. In real books full chunks run at 3.1x (median of 488; 5% under 1.8x),
and a whole book at 2.5-2.8x including the speech check and retries. That was with chapters sent in
book order. Sorted by length (2026-10-02, WORKLOG §50), a 108-unit chapter's generating time fell
from 307 s to 189-201 s, and its total time with the checks from 325 s to 198-209 s.

## Licence

Inference code (`breeze-tts`, pinned in the Dockerfile): Apache-2.0. Model weights: BreezeBlue's
research / non-commercial licence; check it before any commercial use.

## Tests

No GPU needed: `docker build -t breeze-tts:local .` then
`docker run --rm --entrypoint python breeze-tts:local -m pytest -q /opt/breeze-infer/test_server.py`.
