"""Command line: rag-autotune init | run | status | failures | report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core import AutotuneError, Project, FAILURES, PLATEAU


def _print_run(r: dict) -> None:
    if r.get("changes"):
        print("tested: " + "; ".join(r["changes"]))
    if r["decision"] == "CRASH":
        print(f"exp {r['exp']}: CRASH (config restored to last kept version)")
        print(r["error"])
        return
    line = (f"exp {r['exp']}: {r['decision']}  train={r['train_score']:.4f} "
            f"delta={r['delta']:+.4f}  mrr={r['mrr']:.4f}  holdout_gate={r['holdout_gate']}  "
            f"p95={r['p95_ms']:.0f}ms  misses={r['n_train_misses']}")
    print(line)
    for n in r.get("notes", []):
        print(f"NOTE: {n}")
    if r.get("reason"):
        print(f"  reason: {r['reason']}")
    if r["decision"] == "DISCARD":
        print("  config restored to last kept version")
    if r.get("streak", 0) >= PLATEAU:
        print(f"STOP: {r['streak']} runs in a row without a KEEP. You are on a plateau; more runs mostly "
              f"fit noise. Stop here and run: rag-autotune report")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rag-autotune",
                                 description="Autoresearch-style tuning loop for RAG retrieval.")
    ap.add_argument("--root", default=".", help="project folder (default: current)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("init", help="lock eval set + adapter, set the metric")
    i.add_argument("--eval", required=True, help="JSONL: {id, question, relevant_ids}")
    i.add_argument("--adapter", required=True, help="py file with build_index(config) and retrieve(index, q, config)")
    i.add_argument("--config", required=True, help="py file with CONFIG dict (the only file the agent edits)")
    i.add_argument("--k", type=int, default=5, help="locked cut-off for recall@k (default 5)")
    i.add_argument("--metric", default="recall", choices=["recall", "mrr", "hit"])
    i.add_argument("--holdout", type=float, default=0.3, help="holdout fraction (default 0.3)")
    i.add_argument("--min-delta", type=float, default=None, help="min train gain to KEEP (default 1/n_train)")
    i.add_argument("--latency-budget-ms", type=float, default=None, help="discard if p95 retrieval latency exceeds this")
    i.add_argument("--seed", default="rag-autotune")
    i.add_argument("--force", action="store_true", help="wipe .autotune/ and start over")

    r = sub.add_parser("run", help="apply --set changes, score the config, KEEP or DISCARD it")
    r.add_argument("-m", "--hypothesis", required=True,
                   help="one line: why you expect this change to help (a label only, it changes nothing)")
    r.add_argument("--set", dest="sets", nargs="+", metavar="KEY=VALUE", default=[],
                   help="config changes to apply before scoring, e.g. --set chunk_size=40 chunk_overlap=10")
    r.add_argument("--simplify", action="store_true",
                   help="this change only removes complexity: keep it if the score does not drop")

    st = sub.add_parser("set", help="change config values safely (validated keys and types)")
    st.add_argument("pairs", nargs="+", metavar="KEY=VALUE")

    sub.add_parser("reset-config", help="put the config back to the last kept version")

    sub.add_parser("status", help="run log so far")
    sub.add_parser("failures", help="train questions the current best config misses")
    sub.add_parser("report", help="final summary for humans (shows holdout numbers)")
    sub.add_parser("agents-md", help="print loop instructions for any agent: rag-autotune agents-md >> AGENTS.md")

    a = ap.parse_args(argv)
    proj = Project(Path(a.root))
    try:
        if a.cmd == "init":
            s = proj.init(a.eval, a.adapter, a.config, a.k, a.metric, a.holdout,
                          a.min_delta, a.latency_budget_ms, a.seed, a.force)
            print(f"initialised: {s['metric']}@{s['k']}, train={s['n_train']} holdout={s['n_holdout']}, "
                  f"min_delta={s['min_delta']}")
            print(f"locked: {', '.join(s['locked'])}   editable: {s['config']}")
            for w in s["warnings"]:
                print(f"WARNING: {w}")
            print("next: rag-autotune run -m 'baseline'")
        elif a.cmd == "set":
            proj.check_locks()
            for c in proj.set_values(a.pairs) or ["no change (values already set)"]:
                print(f"set {c}")
        elif a.cmd == "run":
            if a.sets:
                proj.check_locks()
                for c in proj.set_values(a.sets):
                    print(f"set {c}")
            res = proj.run(a.hypothesis, a.simplify)
            _print_run(res)
            return 1 if res["decision"] == "CRASH" else 0
        elif a.cmd == "reset-config":
            for ch in proj.reset_config() or ["already at the last kept version"]:
                print(f"reset {ch}")
        elif a.cmd == "status":
            hist = proj.history()
            best = proj.best()
            s = proj.settings
            print(f"project: {proj.root}   metric: {s['metric']}@{s['k']}   "
                  f"train={s['n_train']} holdout={s['n_holdout']}   min_delta={s['min_delta']}")
            for h in hist:
                print(f"{h['exp']:>3} {h['decision']:<8} train={h['train_score']:<7} {h['delta']:<8} "
                      f"gate={h['holdout_gate']:<4} {h['hypothesis']}")
            if best:
                print(f"best: exp {best['exp']} train={best['train']:.4f}")
        elif a.cmd == "failures":
            f = proj.p(FAILURES)
            print(f.read_text() if f.exists() else "[] (run an experiment first)")
        elif a.cmd == "report":
            print(json.dumps(proj.report(), indent=2))
        elif a.cmd == "agents-md":
            body = (Path(__file__).parent / "instructions.md").read_text(encoding="utf-8")
            print("<!-- rag-autotune: instructions for coding agents (Pi, Codex, Claude Code, Cursor...) -->")
            print(body.replace("# rag-autotune", "## rag-autotune: tuning loop", 1))
    except AutotuneError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
