# Project instructions

Read and follow `AGENTS.md` in this directory. It covers CodeGraph, the Omni bridge
(the default for repository search and code writing), the shared GPU coordinator, and
the required 14B casting model.

For Omni in particular: it runs on the curated FreeLLMAPI chain `auto:vscode`; it can
test code in throwaway worktrees with this repository's unittest command; and when it
falls short, look for the local fix (its `opencode.json` allowlist or steps, the shared
bridge prompts, or the model) before working around it. `AGENTS.md` has the details.
