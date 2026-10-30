# Local coding and GPU use

This repository has a `.codegraph/` index. Use CodeGraph to locate and understand
affected code before searching or reading it directly.

**The Omni bridge is the default for repository search and for code writing.**
You orchestrate and review; Omni investigates, implements and tests.
- Before reading several unfamiliar files yourself, ask `omni_research` a precise
  question and ask for file:line plus short excerpts of only the code you will
  change. Use CodeGraph for quick exact symbol reads and to verify Omni's claims.
- Route code edits through `omni_implement` with `confirm=true` and explicit
  `allowed_paths` relative to this `src/`. Hand it a natural-language plan, not
  verbatim code. The plan must name the tests to add or update, the docs and
  comments to change, old names to search for stale references, and the exact
  verification commands. Keep it short: name an existing test to copy instead of
  spelling tests out, and leave out background and the rules the bridge prompt
  already adds (stay in scope, never weaken tests, report pass/fail counts); the plan
  should be much shorter than the code it produces. A trivial, low-risk edit whose code is already in your
  context may be made directly when delegating would cost more than it saves.
- Use `omni_review_fix` only for confirmed, narrowly scoped findings. Never
  delegate commits or pushes.
- Verify by risk, without redoing Omni's whole investigation: low risk (tests,
  scripts, docs) → the report's "Changed by this run" stat and your own test run;
  production code → the diff of the production files; cast attribution, output
  audio, deletions, the GPU coordinator or concurrency → the full diff plus
  independent checks. Never take a reported test pass on trust; Omni's summaries
  can claim edits it never made, so check them against "Changed by this run".
- Edit directly when Omni is unavailable or keeps failing a phase, and say so.

- How it runs: the MCP server
  `C:/Server/stacks/abs-kosync-enhanced/scripts/omni_bridge/omni_mcp_server.py`
  launches OpenCode in this directory. OpenCode uses the git-excluded
  `opencode.json` here, which routes to FreeLLMAPI, a local free-tier router
  (Docker stack `C:/Server/stacks/freellmapi`, `http://localhost:3001/v1`). Each agent
  has its own chain in `opencode.json`: research `auto:omni-research`, edits and
  review fixes `auto:omni-implement`, OpenCode's small_model `auto:omni-fast`. The
  fallback `OMNI_BRIDGE_MODEL` is `freellmapi/auto:vscode`, the user's own curated chain; do not go back to
  `auto:smart`, a global sort that sent ~90% of calls to Dots3 Note Preview). Claude loads it from
  `C:/Server/stacks/epub-to-audiobook/.mcp.json` and Codex from
  `C:/Server/stacks/epub-to-audiobook/.codex/config.toml`. Both set
  `OMNI_BRIDGE_REPO_ROOT` to this `src/`.
- Phase large work into small calls: the implementer stops at 40 steps, and the
  bridge allows 1800 s per run. A report saying "stopped early" hit one of those
  limits; split the rest of the plan.
- A failed or timed-out edit run still returns the full report (run state,
  partial output, "Changed by this run"). Inspect those changes before retrying;
  never resend the same plan over a partly edited tree.
- One edit run per checkout at a time: a second `omni_implement` or
  `omni_review_fix`, from Claude or Codex, gets an immediate busy error naming the
  holder. Wait for it rather than retrying in a loop. Research does not lock.
- Every Omni call appends one line (caller, tool, duration, outcome, OpenCode
  session, steps, and plan size `brief_chars` vs code added `added_chars`) to
  `~/.omni-bridge/runs.jsonl`. FreeLLMAPI's own database
  (`requests`, `request_attempts`) shows which provider served each step.
- Omni sends repository code to free cloud models, so never point it at book text,
  `.env` files, or credentials.
- `omni_research` can test code that is not checked out (a PR, another branch) in a
  throwaway worktree under `C:/Users/cporc/AppData/Local/Temp/omni-review/<name>`.
  This repository has no `.venv`; its test command, pointed at the worktree, is
  `python -m unittest discover -s C:/Users/cporc/AppData/Local/Temp/omni-review/<name>/tests -t C:/Users/cporc/AppData/Local/Temp/omni-review/<name> -p "*test*.py"`
  (ffmpeg required). For a review, ask it for the with-fix and fix-reverted runs. Its
  memo labels each claim confirmed (with file:line or command output) or hypothesis;
  verify confirmed claims that matter before acting on them.
- **When Omni falls short, look for a local fix before working around it.** It is
  local tooling the user wants improved:
  - a blocked command → the agent's `permission.bash` allowlist in this `opencode.json`
    (keep new rules confined to a path or an exact command; never a bare `*` on the
    read-only researcher);
  - running out of steps → `steps` on the agent in `opencode.json`, or smaller phases;
  - a recurring wrong behaviour (re-reading docs, wrong language, unverified claims,
    invented symbol names, weakening tests) → the prompt builders in
    `C:/Server/stacks/abs-kosync-enhanced/scripts/omni_bridge/omni_mcp_server.py`,
    which BookBridge shares, so a prompt fix there applies to both repositories;
  - weak answers → the `model` argument per call, or `OMNI_BRIDGE_MODEL`.
  Say what fell short and the fix. A small, reversible config or prompt change can be
  made and smoke-tested in the same session; anything bigger, ask first. Prompt edits
  load only after the MCP server restarts; `opencode.json` is read on every run.

Do not pause containers or directly start/load a GPU model to bypass the
coordinator.

Casting uses **qwen2.5:14b**. The trained 7B was rejected; do not substitute it.
The shared reservation client covers books, casts, LLM requests, and previews.
Keep reservations alive throughout actual model work and release them on errors.

Keep output concise; reuse known source context and run checks appropriate to the change.
