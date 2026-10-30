# Resemble Enhance service

A small FastAPI service that cleans voice reference clips with Resemble Enhance.
Used for the voices library, especially Breeze-designed voices. The model
needs about 4.5 GB of GPU memory at peak, so it cannot run while Breeze is
loaded; the app unloads Breeze first.

## Endpoints

- `GET /health` -> `{"status": "ok", "loaded": bool}`
- `POST /enhance` -> raw WAV bytes in, enhanced WAV bytes out (peak-normalised to -1 dBFS)
- `POST /api/unload` -> `{"loaded": false}`

Weights live in a volume mounted at
`/usr/local/lib/python3.10/site-packages/resemble_enhance/model_repo` (~1.6 GB)
and are downloaded on first load.

## Settings

Fixed inference settings chosen by ear on 2026-10-30 (variant "8D": default
tau, twice the steps, rk4):
- NFE = 128
- SOLVER = "rk4"
- LAMBD = 1.0
- TAU = 0.5

## GPU / speed

About 3.4 GB allocated / 4.4 GB reserved peak. ~2 s per 8 s clip once loaded,
~7 s for the first clip.

Idle unload after `RESEMBLE_IDLE_UNLOAD_SECONDS` (default 120 s) frees the GPU.

## Tests

```bash
docker run --rm --entrypoint python resemble-enhance:local -m pytest -q /app/test_server.py
```
