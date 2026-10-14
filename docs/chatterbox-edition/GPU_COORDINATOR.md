# Automatic GPU handoffs on this Windows host

The audiobook app shares one authenticated coordinator at
`C:/Server/stacks/local-ai/gpu_coordinator.py`. The Windows scheduled task
**Local AI GPU Coordinator** runs it now and starts it at user logon.

Set `GPU_COORDINATOR_URL` and `GPU_COORDINATOR_TOKEN` in the deployment `.env`.
The Docker app uses `http://host.docker.internal:8765`; native clients use
`http://127.0.0.1:8765`. The token comes from local `coordinator.json`; never commit
or print it. Empty URL preserves the original standalone handoffs.

The client holds a reservation throughout each book, cast, or preview.
Different engines wait their turn. Multiple requests for the current engine can
share it until an earlier request for another engine is waiting. Nested LLM calls
reuse their cast's reservation. The coordinator checks unloads before loading the
next engine, and unloads all managed models when the last reservation is released.
Ollama loads the existing `qwen2.5:14b` on its first reserved request.

Clients renew every 10 seconds; a missing client expires after 120 seconds.
Reservations persist across coordinator restarts. Failed switches remain blocked
until cleanup succeeds. Breeze's unload waits for its current synthesis to finish.

Speech and casting wait for each other instead of loading two models at once.
Cold loads can take one to three minutes. Raw model API calls and other GPU
applications bypass these reservations; use the app.
Chatterbox remains retired on this host.

See `C:/Server/stacks/local-ai/README.md` for service controls and logs. The native
Strata coding helper was removed from this host on 2026-10-14.
