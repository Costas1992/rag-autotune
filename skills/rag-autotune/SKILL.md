---
name: rag-autotune
description: Run an autoresearch-style (Karpathy) tuning loop on a RAG system's retrieval - propose one config change, score it against a locked eval set with a holdout gate, keep or revert, repeat. Use this whenever the user wants to tune, optimise, improve or benchmark RAG retrieval, chunking, top-k, BM25/hybrid weights, embeddings or rerankers, asks why their RAG misses documents, wants to "autoresearch" their RAG, or mentions rag-autotune - even if they don't say "tune". Also use it to help them build the eval set the loop needs.
---

# rag-autotune

You are running experiments on someone's RAG retrieval. The `rag-autotune` CLI is the lab bench: it scores a config, decides KEEP or DISCARD, and restores the last good config on its own. Your job is the scientist's job: read failures, form a hypothesis, change one thing, run, learn.

The whole method rests on one idea: **the score is only worth something if you cannot game it.** The CLI locks the eval set and the adapter by hash, cuts scoring at a fixed k, and hides holdout numbers behind a PASS/FAIL gate. Respect those guards. If you find a way around them, you've found a bug to report, not a shortcut to take.

## The files

| File | Who edits it | What it is |
|---|---|---|
| `config.py` (name may differ, see `.autotune/settings.json`) | **you** | `CONFIG` dict of knobs. The only file you change. |
| adapter (`adapter.py`) | human only, locked | `build_index(config)` and `retrieve(index, question, config) -> ranked ids` |
| eval set (`*.jsonl`) | human only, locked | one line per question: `{"id", "question", "relevant_ids"}` |
| `.autotune/` | the CLI | run log, best config, failures. **Do not read `holdout.jsonl`** - looking at holdout numbers turns the holdout into a second train set. |

Install if missing: `pip install git+https://github.com/Costas1992/rag-autotune` (or `pip install -e .` inside the repo).

## Phase 1: setup (do not skip)

1. **Check the eval set exists and is real.** If the user has none, stop tuning and help them build one first (see `references/eval-set.md`). A loop on a bad eval set optimises the wrong thing very efficiently.
2. **Check the adapter exposes the knobs.** Every tunable thing must be read from `config`. If a knob they care about is hard-coded, tell them; adapter changes are the human's job because the adapter is locked.
3. **Init** (ask the user for k if unclear; default 5, use what their app actually passes to the LLM):
   ```
   rag-autotune init --eval evalset.jsonl --adapter adapter.py --config config.py --k 5
   ```
   Pass on any WARNING lines to the user. Under ~30 questions the score is coarse.
4. **Baseline:** `rag-autotune run -m "baseline"`.

## Phase 2: the loop

Repeat until a stop condition (below):

1. **Look at failures:** `rag-autotune failures`. Group the misses by *why* they fail - vocabulary mismatch ("tracker" vs "tag"), answer split across chunks, a long doc drowning short ones, near-duplicate docs competing. A hypothesis that comes from a failure pattern beats a random sweep.
2. **Write the hypothesis first**, one line: what you change, and which failures it should fix.
3. **Change one thing** in the config. One change per run so the result is attributable. (Exception: two knobs that only make sense together, such as chunk_size and chunk_overlap.)
4. **Run:** `rag-autotune run -m "<hypothesis>"`
   - `KEEP` - the change is now the baseline. Build on it.
   - `DISCARD` - the CLI already restored the config. Read the reason. A holdout-gate FAIL with a positive train delta means you overfit: do not retry close variants of the same change.
   - `CRASH` - the config broke the adapter. Read the error, fix, rerun.
5. **Simplify when you can.** If a knob looks like it isn't pulling its weight, revert it toward the default and run with `--simplify`: it is kept if the score doesn't drop. Fewer moving parts is a win on its own.

Keep a short running log in the chat (one line per experiment) so the user can follow without opening files.

See `references/playbook.md` for the hypotheses that usually pay off, in the order worth trying.

## Stop conditions

Stop and report when any of these happens:
- the experiment budget the user gave is used up (default: 20 runs);
- 6 DISCARDs in a row (you are on a plateau; more runs mostly fit noise);
- train score is 1.0 (the eval set is too easy - say so, it needs harder questions);
- the remaining failures need adapter changes (new retriever, reranker, embedding model) that you cannot make from config.

## The final report

Run `rag-autotune report` (this one shows holdout numbers; it's for the human) and give the user:
- baseline vs best, train **and** holdout, as `metric@k`;
- the kept changes, one line each, with their deltas;
- what still fails and why, grouped by cause;
- **one** recommended next step outside the config's reach (for example "add a reranker" or "add 15 questions about pricing, it's under-tested");
- an honest caveat when the holdout is small: "11 holdout questions, so ±9 points per question; treat small gains as unproven."

Don't overclaim. A gain that only shows up on train is not a gain.

## Rules that protect the result

- Never edit the eval set, the adapter or anything in `.autotune/`. The CLI refuses to run if you do, and the refusal is correct.
- Never re-run `init --force` to escape a bad run log. That's a human decision.
- Don't tune k itself; k is a property of the user's app, not a free parameter.
- If the user asks you to "just make the number go up", explain the gate instead of working around it.
