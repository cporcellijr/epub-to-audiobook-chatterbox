# Local coding and GPU use

This repository has a `.codegraph/` index. Use CodeGraph to locate and understand
affected code before searching or reading it directly.

**The Omni bridge is the default for repository search and for code writing.**
Use the `omni_research` MCP tool first for multi-file search and reconnaissance.
Route code edits, small fixes included, through `omni_implement` with
`confirm=true` and explicit relative `allowed_paths` (relative to this `src/`
directory). Hand it a natural-language plan, not verbatim code. You plan and
review; Omni types. Review every Omni diff and run the tests yourself. Use
`omni_review_fix` only for confirmed, narrowly scoped findings. Never delegate
commits or pushes. Use CodeGraph for quick exact symbol reads and to verify Omni's
claims. Edit directly only when Omni is unavailable or spent, and say so.

- How it runs: the MCP server
  `C:/Server/stacks/abs-kosync-enhanced/scripts/omni_bridge/omni_mcp_server.py`
  launches OpenCode in this directory. OpenCode uses the git-excluded
  `opencode.json` here, which routes to FreeLLMAPI, a local free-tier router
  (Docker stack `C:/Server/stacks/freellmapi`, `http://localhost:3001/v1`, model
  `freellmapi/auto:smart`). Claude loads it from
  `C:/Server/stacks/epub-to-audiobook/.mcp.json` and Codex from
  `C:/Server/stacks/epub-to-audiobook/.codex/config.toml`. Both set
  `OMNI_BRIDGE_REPO_ROOT` to this `src/`.
- Phase large work into small per-file calls. One call carrying a whole multi-file
  feature exhausts OpenCode's 32-request run cap. Run `git diff` after any failed
  run, because a cut-off run can corrupt existing code.
- Omni sends repository code to free cloud models, so never point it at book text,
  `.env` files, or credentials.

Do not pause containers or directly start/load a GPU model to bypass the
coordinator.

Casting uses **qwen2.5:14b**. The trained 7B was rejected; do not substitute it.
The shared reservation client covers books, casts, LLM requests, and previews.
Keep reservations alive throughout actual model work and release them on errors.

Keep output concise; reuse known source context and run checks appropriate to the change.
