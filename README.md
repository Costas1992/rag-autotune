# rag-autotune

An autoresearch-style tuning loop for RAG **retrieval**, after [karpathy/autoresearch](https://github.com/karpathy/autoresearch). An agent changes one config knob, the harness scores it against a locked eval set, and the change is kept or reverted. Then it repeats.

The loop is the easy part. The part that matters is making the score **hard to game**:

| Guard | What it stops |
|---|---|
| Eval set and adapter locked by SHA-256 at `init` | the agent "improving" the test instead of the system |
| Scoring cut at a locked `k` | inflating recall@k by raising `top_k` |
| Holdout split, shown to the agent only as PASS/FAIL | overfitting the questions it can see |
| `min_delta` defaults to one question's worth | keeping changes that are just noise |
| Auto-restore of the last kept config on DISCARD/CRASH | drifting from the best known state |
| Optional latency budget on p95 | buying recall with an unusable retriever |
| Config compiled from source on every run, no bytecode cache | stale results from fast same-size edits (a real bug found while building this) |

No dependencies. Python 3.9+.

## Install

```bash
pip install git+https://github.com/Costas1992/rag-autotune
```

Then give your agent the loop instructions. Pick one:

| Agent | How |
|---|---|
| **Pi** | `cp -r skills/rag-autotune ~/.pi/agent/skills/` (or `.pi/skills/` per project) |
| **Claude Code** | `cp -r skills/rag-autotune ~/.claude/skills/` |
| **Codex, Pi and others** using the shared folder | `cp -r skills/rag-autotune ~/.agents/skills/` |
| **claude.ai** | upload `rag-autotune.skill` in Settings → Skills |
| **Any agent that reads AGENTS.md** | in your project: `rag-autotune agents-md >> AGENTS.md` |

The skill and `agents-md` contain the same instructions, and a test keeps them in sync.

## Try the demo (1 minute)

```bash
cd examples/demo
rag-autotune init --eval evalset.jsonl --adapter adapter.py --config config.py --k 2
rag-autotune run -m "baseline"
rag-autotune failures          # what the current best config misses
# edit config.py, e.g. "expand_synonyms": True
rag-autotune run -m "synonym expansion: tracker->tag, money->refund"
rag-autotune status
rag-autotune report            # final summary, includes holdout numbers
```

Or tell your agent: *"autotune the retrieval in examples/demo, budget 15 experiments"*.

## Plug in your own RAG

You need three files.

**1. `evalset.jsonl`**, one question per line ([how to build a good one](skills/rag-autotune/references/eval-set.md)):
```json
{"id": "q01", "question": "How long does the battery last?", "relevant_ids": ["battery-doc"]}
```

**2. `adapter.py`**, the bridge to your retriever (locked after init):
```python
def build_index(config):            # build or load your index for this config
    ...
def retrieve(index, question, config) -> list[str]:   # ranked ids, best first
    ...
```
Read every knob you want tuned from `config`. If it's hard-coded, it can't be tuned.

**3. `config.py`**, the only file the agent edits:
```python
CONFIG = {"chunk_size": 512, "chunk_overlap": 64, "top_k": 5, "hybrid_alpha": 0.5}
```

Then `rag-autotune init ... --k 5` with the k your app actually uses.

## Commands

| Command | Does |
|---|---|
| `init --eval --adapter --config [--k 5] [--metric recall\|mrr\|hit] [--holdout 0.3] [--min-delta] [--latency-budget-ms]` | lock files, split train/holdout, start a run log |
| `run -m "hypothesis" [--simplify]` | score, KEEP/DISCARD/CRASH. `--simplify` keeps a change that doesn't lower the score |
| `failures` | train questions the best config misses |
| `status` | run log |
| `report` | baseline vs best, train **and** holdout |

## Demo run (9 experiments, recall@2)

| exp | change | train | gate | decision |
|---|---|---|---|---|
| 0 | baseline | 0.66 | - | BASELINE |
| 1 | remove stopwords | 0.64 | FAIL | DISCARD |
| 2 | synonym expansion | 0.76 | PASS | **KEEP** |
| 3 | stemming | 0.78 | PASS | DISCARD (below min_delta) |
| 5 | chunk 40 / overlap 10 | 0.80 | **FAIL** | DISCARD: train up, holdout down = overfitting |
| 4, 6-8 | title prefix, sum aggregation, BM25 b, stemming again | ≤0.78 | | DISCARD, then stop (plateau) |

Read the demo honestly:
- **The synonym table leaks.** It was written by hand while looking at the eval questions, which is exactly what you must *not* do with a real system. The demo uses it to show the mechanics. In real use, expansion terms must come from somewhere independent of the eval set, such as query logs or a thesaurus.
- **The holdout is small and easy here** (11 questions, baseline 0.91). Each question is about 9 points, so the gate catches big regressions only.

## Limits (v0.1)

- The agent can still *read* `.autotune/holdout.jsonl` if it ignores the skill's instruction. The PASS/FAIL hiding is a soft guard. Run the harness outside the agent's filesystem for a hard one.
- Retrieval only. Answer quality (faithfulness, correctness) needs an LLM judge, which is noisier and gameable. Planned for v2, behind the same gate.
- Config-only edits. Adapter or code edits (new rerankers, new chunkers) are a human step for now.

## Tests

```bash
python -m unittest discover tests
```

MIT licensed.
