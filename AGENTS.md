# AGENTS.md

Instructions for coding agents (Pi, Codex, Claude Code, Cursor and others) working **in this repository**.

**Want to run the tuning loop on your own RAG?** This file isn't for that. In your project, run `rag-autotune agents-md >> AGENTS.md`, or install the skill (see README).

## What this repo is

`rag-autotune`, an autoresearch-style tuning loop for RAG retrieval. The CLI scores a config against a locked eval set, gates on a hidden holdout, and keeps or reverts the change.

## Layout

- `rag_autotune/core.py`: loading, splitting, scoring, guards, run log. Most logic lives here.
- `rag_autotune/cli.py`: argparse commands. Keep it thin.
- `rag_autotune/instructions.md`: the agent loop instructions. **Single source of truth**, see below.
- `skills/rag-autotune/`: the skill (SKILL.md plus references) for Claude Code, Pi and Codex.
- `examples/demo/`: a toy corpus, eval set, BM25 adapter and config.
- `tests/`: stdlib `unittest`.

## Rules

1. **Stdlib only.** No runtime dependencies. Install friction kills adoption.
2. **Never weaken a guard to make something pass.** The hash locks, the locked-k cut-off, the holdout gate, `min_delta` and the auto-restore are the product. A change that loosens one needs a test that shows why it's still safe.
3. **Run the tests before every commit:** `python -m unittest discover tests`. All must pass.
4. **Keep the instructions in sync.** The body of `skills/rag-autotune/SKILL.md` (after the frontmatter) must equal `rag_autotune/instructions.md` byte for byte. A test enforces it. Edit one, copy to the other.
5. **Python 3.9+.** Keep `from __future__ import annotations` in modules that use `X | None` hints.
6. **Never load user code through the import cache.** `load_module` compiles from source on purpose: `__pycache__` keys on (mtime seconds, size) and caused stale experiment results. There's a regression test for it.
7. After changing the demo, re-run the README demo commands and update the numbers in the README table.
8. Design the agent-facing CLI for weak models: every error should name the fix (see `set_values`, the NOOP guard and the baseline guard, all from real Gemma 12B failures). Rules a weak model must follow belong in CLI output, not only in the skill text.
9. Don't commit `.autotune/` run state (it's gitignored).
