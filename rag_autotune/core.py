"""Core of rag-autotune: load, split, score, gate, log.

Design rules (why the code looks like this):
- The scorer and eval set are locked by hash at `init`. If anything but the
  config file changes, `run` refuses. An agent that can edit its own scorer
  will "improve" the scorer instead of the system.
- The agent sees train scores and per-question train failures, but only a
  PASS/FAIL for the holdout gate. That makes it harder to overfit the holdout.
- Discarded experiments restore the last kept config automatically, so the
  config file always holds the best known state between runs.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import shutil
import statistics
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

STATE_DIR = ".autotune"
SETTINGS = "settings.json"
RESULTS = "results.tsv"
HOLDOUT_LOG = "holdout.jsonl"      # humans only; the skill tells the agent not to read it
FAILURES = "last_failures.json"
BEST_CONFIG = "best_config.py"
BEST_STATE = "best.json"

RESULTS_HEADER = "exp\ttime\tdecision\ttrain_score\tdelta\tmrr\tholdout_gate\tp95_ms\thypothesis\n"


class AutotuneError(Exception):
    """Expected, user-facing failure (bad input, tampering, missing init)."""


# ---------------------------------------------------------------- helpers

def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(path: Path, name: str):
    """Execute a .py file fresh from its source every time.

    Deliberately bypasses __pycache__: bytecode cache is keyed on (mtime in whole
    seconds, size), so two quick edits of the same size (False -> True) would
    silently reuse the previous experiment's config.
    """
    import types
    mod = types.ModuleType(f"{name}_{time.time_ns()}")
    mod.__file__ = str(path)
    code = compile(path.read_text(encoding="utf-8"), str(path), "exec")
    sys.path.insert(0, str(path.parent))
    try:
        exec(code, mod.__dict__)
    finally:
        sys.path.pop(0)
    return mod


def config_dict(path: Path) -> dict:
    mod = load_module(path, "config")
    cfg = getattr(mod, "CONFIG", None)
    if not isinstance(cfg, dict):
        raise AutotuneError(f"{path.name} must define a dict named CONFIG")
    return dict(cfg)


def config_changes(old: dict, new: dict) -> list[str]:
    keys = sorted(set(old) | set(new))
    return [f"{k}: {old.get(k, '<unset>')!r} -> {new.get(k, '<unset>')!r}"
            for k in keys if old.get(k, object()) != new.get(k, object())]


def parse_value(key: str, raw: str, current):
    """Parse 'raw' from the command line and check it has the same type as the current value.

    Strict on purpose: a weak agent writing chunk_size=big or stem=yes should get a clear
    error, not a silently wrong experiment.
    """
    import ast
    if isinstance(current, bool):
        low = raw.strip().lower()
        if low in ("true", "false"):
            return low == "true"
        raise AutotuneError(f"{key} is a bool: use True or False, not {raw!r}")
    try:
        val = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        val = raw  # bare word -> string, e.g. doc_aggregation=sum
    if isinstance(current, (int, float)):
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise AutotuneError(f"{key} is a number (currently {current!r}), got {raw!r}")
        if isinstance(current, int) and isinstance(val, float):
            if not val.is_integer():
                raise AutotuneError(f"{key} is an integer (currently {current!r}), got {raw!r}")
            val = int(val)
        if isinstance(current, float):
            val = float(val)
        return val
    if isinstance(current, str):
        if not isinstance(val, str):
            raise AutotuneError(f"{key} is text (currently {current!r}), got {raw!r}")
        return val
    return val  # lists, dicts, None: accept any Python literal


def load_eval_set(path: Path) -> list[dict]:
    items, seen = [], set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise AutotuneError(f"{path}:{n} is not valid JSON: {e}") from e
        for key in ("id", "question", "relevant_ids"):
            if key not in row:
                raise AutotuneError(f"{path}:{n} is missing '{key}'")
        if not isinstance(row["relevant_ids"], list) or not row["relevant_ids"]:
            raise AutotuneError(f"{path}:{n} 'relevant_ids' must be a non-empty list")
        if row["id"] in seen:
            raise AutotuneError(f"{path}:{n} duplicate id '{row['id']}'")
        seen.add(row["id"])
        row["relevant_ids"] = [str(x) for x in row["relevant_ids"]]
        items.append(row)
    if not items:
        raise AutotuneError(f"{path} has no questions")
    return items


def split(items: list[dict], holdout_frac: float, seed: str) -> tuple[list[dict], list[dict]]:
    """Deterministic split: rank questions by hash(seed:id), lowest fraction is holdout.

    Same ids + same seed -> same split on every run, independent of file order.
    """
    ranked = sorted(items, key=lambda it: hashlib.sha256(f"{seed}:{it['id']}".encode()).hexdigest())
    n_hold = max(1, round(len(items) * holdout_frac)) if len(items) > 1 else 0
    hold_ids = {it["id"] for it in ranked[:n_hold]}
    train = [it for it in items if it["id"] not in hold_ids]
    hold = [it for it in items if it["id"] in hold_ids]
    return train, hold


def dedupe(ids) -> list[str]:
    out, seen = [], set()
    for x in ids:
        x = str(x)
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ---------------------------------------------------------------- scoring

@dataclass
class Score:
    recall: float
    mrr: float
    hit: float
    p95_ms: float
    misses: list[dict] = field(default_factory=list)

    def primary(self, metric: str) -> float:
        return {"recall": self.recall, "mrr": self.mrr, "hit": self.hit}[metric]


def score(items: list[dict], index, adapter, config: dict, k: int) -> Score:
    recalls, rrs, hits, lat, misses = [], [], [], [], []
    for it in items:
        t0 = time.perf_counter()
        ranked = adapter.retrieve(index, it["question"], config)
        lat.append((time.perf_counter() - t0) * 1000)
        # Truncate to the LOCKED k. Raising top_k in config cannot inflate recall@k.
        top = dedupe(ranked)[:k]
        rel = set(it["relevant_ids"])
        found = rel.intersection(top)
        recalls.append(len(found) / len(rel))
        rank = next((i for i, d in enumerate(top, 1) if d in rel), None)
        rrs.append(1 / rank if rank else 0.0)
        hits.append(1.0 if found else 0.0)
        if len(found) < len(rel):
            misses.append({"id": it["id"], "question": it["question"],
                           "expected": sorted(rel), "got": top})
    p95 = sorted(lat)[max(0, math.ceil(0.95 * len(lat)) - 1)] if lat else 0.0
    return Score(statistics.fmean(recalls), statistics.fmean(rrs), statistics.fmean(hits), p95, misses)


# ---------------------------------------------------------------- project state

class Project:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.state = self.root / STATE_DIR

    def p(self, name: str) -> Path:
        return self.state / name

    @property
    def settings(self) -> dict:
        f = self.p(SETTINGS)
        if not f.exists():
            raise AutotuneError("Not initialised. Run: rag-autotune init --eval ... --adapter ... --config ...")
        return json.loads(f.read_text())

    def path(self, key: str) -> Path:
        return (self.root / self.settings[key]).resolve()

    # -- init
    def init(self, eval_path: str, adapter: str, config: str, k: int, metric: str,
             holdout_frac: float, min_delta: float | None, latency_budget_ms: float | None,
             seed: str, force: bool) -> dict:
        if self.p(SETTINGS).exists() and not force:
            raise AutotuneError(".autotune/ already exists. Use --force to start a fresh run log.")
        for f in (eval_path, adapter, config):
            if not (self.root / f).exists():
                raise AutotuneError(f"File not found: {f}")
        if metric not in ("recall", "mrr", "hit"):
            raise AutotuneError("metric must be recall, mrr or hit")
        items = load_eval_set(self.root / eval_path)
        train, hold = split(items, holdout_frac, seed)
        if min_delta is None:
            # One question's worth of improvement: smaller deltas are noise on small sets.
            min_delta = round(1 / len(train), 4)
        if self.state.exists():
            shutil.rmtree(self.state)
        self.state.mkdir()
        locked = {f: sha256_file(self.root / f) for f in (eval_path, adapter)}
        settings = {
            "eval": eval_path, "adapter": adapter, "config": config,
            "k": k, "metric": metric, "holdout_frac": holdout_frac, "seed": seed,
            "min_delta": min_delta, "latency_budget_ms": latency_budget_ms,
            "locked": locked, "n_train": len(train), "n_holdout": len(hold),
        }
        self.p(SETTINGS).write_text(json.dumps(settings, indent=2))
        self.p(RESULTS).write_text(RESULTS_HEADER)
        (self.state / ".gitignore").write_text("*\n")
        warnings = []
        if len(items) < 20:
            warnings.append(f"Only {len(items)} questions. Aim for 30+: below that one question moves the score by >3 points.")
        if len(hold) < 5:
            warnings.append(f"Holdout has {len(hold)} questions. The overfitting gate is weak below ~5.")
        settings["warnings"] = warnings
        return settings

    # -- guards
    def check_locks(self):
        s = self.settings
        for f, h in s["locked"].items():
            path = self.root / f
            if not path.exists() or sha256_file(path) != h:
                raise AutotuneError(
                    f"TAMPER: locked file '{f}' changed since init. Only '{s['config']}' may be edited. "
                    f"Revert it, or re-run init --force if a human changed it on purpose.")
        if (self.root / s["config"]).resolve() in {(self.root / f).resolve() for f in s["locked"]}:
            raise AutotuneError("Config file cannot also be a locked file.")

    def next_exp(self) -> int:
        lines = self.p(RESULTS).read_text().splitlines()[1:]
        return len(lines)

    def best(self) -> dict | None:
        f = self.p(BEST_STATE)
        return json.loads(f.read_text()) if f.exists() else None

    def restore_best(self):
        b = self.p(BEST_CONFIG)
        if b.exists():
            shutil.copyfile(b, self.path("config"))

    # -- changing the config without hand-editing
    def set_values(self, pairs: list[str]) -> list[str]:
        """Apply key=value pairs to the config file. All-or-nothing; verified by reloading."""
        import difflib
        import re
        path = self.path("config")
        current = config_dict(path)
        updates = {}
        for pair in pairs:
            if "=" not in pair:
                raise AutotuneError(f"expected key=value, got {pair!r}")
            key, raw = (x.strip() for x in pair.split("=", 1))
            if key not in current:
                hint = difflib.get_close_matches(key, list(current), n=1)
                msg = f"unknown key {key!r}" + (f", did you mean {hint[0]!r}?" if hint else "")
                raise AutotuneError(msg + f" Valid keys: {', '.join(sorted(current))}")
            updates[key] = parse_value(key, raw, current[key])

        original = path.read_text(encoding="utf-8")
        text = original
        for key, val in updates.items():
            pat = re.compile(r'(["\']' + re.escape(key) + r'["\']\s*:\s*)([^,#\n}]+)')
            text, n = pat.subn(lambda m: m.group(1) + repr(val), text, count=1)
            if n == 0:  # unusual formatting: fall back to an explicit override line
                text = text.rstrip("\n") + f"\nCONFIG[{key!r}] = {val!r}\n"
        path.write_text(text, encoding="utf-8")
        try:
            after = config_dict(path)
        except Exception as e:  # noqa: BLE001
            path.write_text(original, encoding="utf-8")
            raise AutotuneError(f"set produced an invalid config file, reverted: {e}") from e
        if any(after.get(k) != v for k, v in updates.items()):
            # Regex hit something unexpected (e.g. a value containing a comma). Use overrides.
            text = original.rstrip("\n") + "\n" + "".join(
                f"CONFIG[{k!r}] = {v!r}\n" for k, v in updates.items())
            path.write_text(text, encoding="utf-8")
            after = config_dict(path)
            if any(after.get(k) != v for k, v in updates.items()):
                path.write_text(original, encoding="utf-8")
                raise AutotuneError("could not apply the change safely; config left unchanged")
        return config_changes(current, after)

    # -- the experiment
    def run(self, hypothesis: str, simplify: bool = False) -> dict:
        s = self.settings
        self.check_locks()
        exp = self.next_exp()
        best = self.best()
        if best is None and exp > 0:
            raise AutotuneError("Run log exists but no baseline. Re-run init --force.")
        changes = []
        if best is not None:
            try:
                changes = config_changes(config_dict(self.p(BEST_CONFIG)), config_dict(self.path("config")))
            except Exception:  # noqa: BLE001 - broken config: let the run below report a CRASH
                changes = ["<config unreadable>"]
            if not changes:
                # A no-op run scores the same config again and burns an experiment for nothing.
                raise AutotuneError(
                    "NOOP: the config is identical to the last kept config, so this run would test "
                    "nothing. The -m text is only a label; it does not change anything. Apply a "
                    "change first, e.g.: rag-autotune run --set stem=True -m 'why'")
        items = load_eval_set(self.path("eval"))
        train, hold = split(items, s["holdout_frac"], s["seed"])
        out = {"exp": exp, "hypothesis": hypothesis, "changes": changes}
        try:
            adapter = load_module(self.path("adapter"), "adapter")
            config = config_dict(self.path("config"))
            t0 = time.perf_counter()
            index = adapter.build_index(config)
            out["index_s"] = round(time.perf_counter() - t0, 2)
            tr = score(train, index, adapter, config, s["k"])
            ho = score(hold, index, adapter, config, s["k"])
        except Exception:  # noqa: BLE001 - any adapter/config failure is a crashed experiment
            out.update(decision="CRASH", error=traceback.format_exc(limit=4))
            self.restore_best()
            self._log(out, None, None, "-")
            return out

        m = s["metric"]
        train_score, hold_score = tr.primary(m), ho.primary(m)
        out.update(train_score=round(train_score, 4), mrr=round(tr.mrr, 4), p95_ms=round(tr.p95_ms, 1))
        self.p(FAILURES).write_text(json.dumps(tr.misses, indent=2))
        out["n_train_misses"] = len(tr.misses)

        budget = s["latency_budget_ms"]
        over_budget = budget is not None and tr.p95_ms > budget

        if best is None:
            decision, gate, delta = "BASELINE", "PASS", 0.0
        else:
            delta = train_score - best["train"]
            gate = "PASS" if hold_score >= best["holdout"] - 1e-9 else "FAIL"
            improved = delta >= s["min_delta"] - 1e-9
            simpler_and_equal = simplify and delta >= -1e-9
            if over_budget:
                decision = "DISCARD"
                out["reason"] = f"p95 {tr.p95_ms:.0f}ms over budget {budget}ms"
            elif gate == "FAIL":
                decision = "DISCARD"
                out["reason"] = "holdout gate failed (train went up, unseen questions got worse = overfitting)" if delta > 0 else "holdout gate failed"
            elif improved or simpler_and_equal:
                decision = "KEEP"
            else:
                decision = "DISCARD"
                out["reason"] = f"delta {delta:+.4f} below min_delta {s['min_delta']}"
        out.update(delta=round(delta, 4), holdout_gate=gate, decision=decision)

        if decision in ("BASELINE", "KEEP"):
            shutil.copyfile(self.path("config"), self.p(BEST_CONFIG))
            self.p(BEST_STATE).write_text(json.dumps(
                {"exp": exp, "train": train_score, "holdout": hold_score, "mrr": tr.mrr}))
        else:
            self.restore_best()
        self._log(out, train_score, hold_score, gate)
        return out

    def _log(self, out: dict, train, hold, gate):
        row = [str(out["exp"]), time.strftime("%Y-%m-%dT%H:%M:%S"), out["decision"],
               f"{train:.4f}" if train is not None else "-",
               f"{out.get('delta', 0):+.4f}" if train is not None else "-",
               f"{out.get('mrr', 0):.4f}" if train is not None else "-",
               gate, f"{out.get('p95_ms', 0):.1f}" if train is not None else "-",
               out["hypothesis"].replace("\t", " ").replace("\n", " ")]
        with self.p(RESULTS).open("a") as f:
            f.write("\t".join(row) + "\n")
        with self.p(HOLDOUT_LOG).open("a") as f:
            f.write(json.dumps({"exp": out["exp"], "holdout": hold}) + "\n")

    # -- reporting
    def history(self) -> list[dict]:
        lines = self.p(RESULTS).read_text().splitlines()
        keys = lines[0].split("\t")
        return [dict(zip(keys, ln.split("\t"))) for ln in lines[1:]]

    def report(self) -> dict:
        s, hist, best = self.settings, self.history(), self.best()
        holdouts = [json.loads(x) for x in self.p(HOLDOUT_LOG).read_text().splitlines()] if self.p(HOLDOUT_LOG).exists() else []
        base_hold = next((h["holdout"] for h in holdouts if h["exp"] == 0), None)
        base = next((h for h in hist if h["decision"] == "BASELINE"), None)
        counts = {}
        for h in hist:
            counts[h["decision"]] = counts.get(h["decision"], 0) + 1
        return {
            "metric": f"{s['metric']}@{s['k']}", "n_train": s["n_train"], "n_holdout": s["n_holdout"],
            "experiments": len(hist), "decisions": counts,
            "baseline_train": float(base["train_score"]) if base else None,
            "best_train": best["train"] if best else None,
            "baseline_holdout": base_hold, "best_holdout": best["holdout"] if best else None,
            "best_exp": best["exp"] if best else None,
            "kept": [h for h in hist if h["decision"] == "KEEP"],
        }
