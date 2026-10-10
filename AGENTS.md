# Local coding and GPU use

This repository has a `.codegraph/` index. Use CodeGraph to locate and understand
affected code before searching or reading it directly.

For contained coding drafts, utility changes, and test drafts, use the local
Strata helper when practical, then review its output and run meaningful checks.
Keep architecture and difficult debugging under careful review: the local model
has made incorrect assumptions about this code's GPU flow.

Write a JSON tasks file containing a list of `{name, prompt, files}`; `files` is
optional and contains absolute Python/Markdown paths under `C:/Server/stacks`.
Run it from PowerShell:

```powershell
& 'C:\Server\stacks\strata\.venv\Scripts\python.exe' 'C:\Server\stacks\local-ai\strata_task.py' '<tasks.json>'
```

The helper waits for the shared GPU reservation and saves drafts under
`C:/Server/stacks/local-ai/results`. Apply only reviewed changes. Do not pause
containers or directly start/load a GPU model to bypass the coordinator.

Casting uses **qwen2.5:14b**. The trained 7B was rejected; do not substitute it.
The shared reservation client covers books, casts, LLM requests, and previews.
Keep reservations alive throughout actual model work and release them on errors.

Keep output concise; reuse known source context and run checks appropriate to the change.
