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
- `POST /v1/audio/speech` -> `{"input", "voice", "ref_text", "instruction", "response_format": "wav"}`,
  returns a WAV (a batch of one).

WAV output is mono, 24 kHz, 16-bit. One generation runs at a time; other requests wait.

## Environment

| Variable | Default |
| --- | --- |
| `BREEZE_MODEL_DIR` | `/models/breeze-tts-2` |
| `BREEZE_VOICES_DIR` | `/voices` |
| `BREEZE_MAX_BATCH` | `32` |
| `BREEZE_REPO_ID` | `BreezeBlue/Breeze-TTS-2` |

## Speed (RTX 4070, eager attention, voice cloning)

| Batch | Real time |
| --- | --- |
| 1 | 0.39x |
| 4 | 1.38x |
| 8 | 1.88x |
| 16 | 3.94x |
| 32 | 7.14x (8.2 GiB peak) |

## Licence

Inference code (`breeze-tts`, pinned in the Dockerfile): Apache-2.0. Model weights: BreezeBlue's
research / non-commercial licence; check it before any commercial use.

## Tests

No GPU needed: `docker build -t breeze-tts:local .` then
`docker run --rm --entrypoint python breeze-tts:local -m pytest -q /opt/breeze-infer/test_server.py`.
