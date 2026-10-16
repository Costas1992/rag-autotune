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
| `--set key=value` with key and type checks, plus a NOOP refusal | agents "testing" a change they never applied (found in a real Gemma 12B run, below) |
| Baseline must use the config as it was at `init`; git-dirty warning at init | an untested change hiding inside the baseline score (found in a Gemma 12B run) |
| `STOP:` after 6 runs in a row without a KEEP | burning budget on a plateau, fitting noise |
| Metric-aware `min_delta` (MRR: one rank-up) and a sized gate message (`small`/`large` drop) | rejecting real MRR gains; calling every one-question wobble "overfitting" (found on rag-doc-qa) |

No dependencies. Python 3.9+.

## Install

```bash
pip install git+https://github.com/Costas1992/rag-autotune
```
On a fresh Ubuntu/Debian you may need `sudo apt install python3-venv` first. If an agent can't find the command, link it somewhere every shell looks: `sudo ln -sf "$(which rag-autotune)" /usr/local/bin/`. Agent shells often don't read your `~/.zshrc`.

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
rag-autotune run --set expand_synonyms=True -m "synonym expansion: tracker->tag, money->refund"
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
| `run --set key=value ... -m "why" [--simplify]` | apply the change, score it, KEEP/DISCARD/CRASH. Refuses a no-op run. `--simplify` keeps a change that doesn't lower the score |
| `set key=value ...` | change config values only (validated keys and types, comments kept) |
| `reset-config` | put the config back to the last kept version |
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

## Field test: Gemma 4 12B on a 10GB GPU (Pi agent)

The first run with a small local model found two failure modes that the Claude run didn't:

1. **Pi's shell couldn't find `rag-autotune`** (PATH from `~/.zshrc` isn't loaded). The model then guessed misspelled command names and looped. Fix: the skill now checks the tool exists first and stops if it doesn't, and stops after the same command fails twice.
2. **Ten no-op experiments.** The model wrote `run -m "stem=True"`, treating the label as the change. All 10 runs scored the unchanged baseline, and the log made it look like "nothing helps". Fix: `--set` makes the change part of the command, and `run` refuses when the config is identical to the last kept one.

Lesson: design the agent's interface for the weakest model you support. A strong model works around a bad interface; a weak one shows you where it is.

### All Gemma runs so far (demo, recall@2, Ollama on an RX 6700 10GB, 32k context)

| Run | Version | Prompt | Outcome |
|---|---|---|---|
| 1 | v0.1 | short | couldn't find the CLI, misspelled it, looped; then 10 no-op experiments |
| 2 | v0.1 | step-by-step | 0.66 → 0.76 (synonym expansion), gate rejected one overfit |
| 3 | v0.2 | short | 0.72 → 0.82 in 10 experiments, gate rejected 3; ~35 min, one output truncation. **But** it changed `chunk_overlap` before the baseline, so the "0.72 baseline" hid an untested +0.06. Fixed in v0.3. |

Run 3 also ran 9 discards in a row, past the skill's plateau rule. Weak models skip rules in prose, so v0.3 prints the stop condition from the CLI itself.

## Field test: a real RAG app (rag-doc-qa, Qwen 3.5 9B, v0.3)

[rag-doc-qa](https://github.com/Costas1992/rag-doc-qa) adapter: production chunker, all-MiniLM-L6-v2, Chroma; 32 questions over an 8-section handbook; MRR@3 over sections.

| exp | change | train | holdout | read honestly |
|---|---|---|---|---|
| 0 | baseline | 0.848 | 1.00 | holdout already perfect: the gate can't confirm gains |
| 2 | chunk_size=300 | +0.038 | 1.00 | probably a real gain; v0.3 rejected it with a recall-sized `min_delta` |
| 5 | all-mpnet-base-v2 | +0.030 | 0.85 | lost 1.5 questions of rank: likely overfitting |
| 6 | chunk_overlap=50 | +0.038 | 0.90 | lost 1 question of rank: overfitting or noise |

The agent concluded "baseline is already optimal". The honest reading: **this corpus (~9 chunks) is too small to decide**. v0.4 fixes the three tool problems the run exposed.

## Limits (v0.4)

- The agent can still *read* `.autotune/holdout.jsonl` if it ignores the skill's instruction. The PASS/FAIL hiding is a soft guard. Run the harness outside the agent's filesystem for a hard one.
- Retrieval only. Answer quality (faithfulness, correctness) needs an LLM judge, which is noisier and gameable. Planned for v2, behind the same gate.
- Config-only edits. Adapter or code edits (new rerankers, new chunkers) are a human step for now.

## Tests

```bash
python -m unittest discover tests
```

MIT licensed.
