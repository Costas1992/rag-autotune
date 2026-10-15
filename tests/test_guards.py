"""Tests for the parts that keep the loop honest. Run: python -m unittest discover tests"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from rag_autotune.core import AutotuneError, Project, split, load_eval_set

DEMO = Path(__file__).resolve().parent.parent / "examples" / "demo"


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        for f in ("adapter.py", "config.py", "corpus.jsonl", "evalset.jsonl"):
            shutil.copy(DEMO / f, self.dir / f)
        self.proj = Project(self.dir)
        self.proj.init("evalset.jsonl", "adapter.py", "config.py", k=2, metric="recall",
                       holdout_frac=0.3, min_delta=None, latency_budget_ms=None,
                       seed="t", force=True)
        self.base = self.proj.run("baseline")

    def tearDown(self):
        shutil.rmtree(self.dir)

    def set_cfg(self, **kv):
        p = self.dir / "config.py"
        text = p.read_text()
        for k, v in kv.items():
            text += f"\nCONFIG[{k!r}] = {v!r}"
        p.write_text(text)

    def test_baseline_recorded(self):
        self.assertEqual(self.base["decision"], "BASELINE")
        self.assertIsNotNone(self.proj.best())

    def test_tamper_eval_refused(self):
        with open(self.dir / "evalset.jsonl", "a") as f:
            f.write(json.dumps({"id": "x", "question": "easy", "relevant_ids": ["battery"]}) + "\n")
        with self.assertRaises(AutotuneError):
            self.proj.run("cheat")

    def test_tamper_adapter_refused(self):
        with open(self.dir / "adapter.py", "a") as f:
            f.write("\n# edited\n")
        with self.assertRaises(AutotuneError):
            self.proj.run("cheat")

    def test_raising_top_k_cannot_inflate_recall_at_k(self):
        self.set_cfg(top_k=10)
        r = self.proj.run("bigger top_k")
        self.assertAlmostEqual(r["train_score"], self.base["train_score"], places=6)
        self.assertEqual(r["decision"], "DISCARD")

    def test_discard_restores_config(self):
        before = (self.proj.p("best_config.py")).read_text()
        self.set_cfg(chunk_size=1)  # very likely worse
        r = self.proj.run("tiny chunks")
        if r["decision"] == "DISCARD":
            self.assertEqual((self.dir / "config.py").read_text(), before)

    def test_crash_restores_config(self):
        before = (self.dir / "config.py").read_text()
        (self.dir / "config.py").write_text("CONFIG = {}\n")  # missing keys -> adapter KeyError
        r = self.proj.run("broken")
        self.assertEqual(r["decision"], "CRASH")
        self.assertEqual((self.dir / "config.py").read_text(), before)

    def test_keep_needs_min_delta(self):
        self.set_cfg(remove_stopwords=True, stem=True, prepend_title=True)
        r = self.proj.run("stopwords+stem+title")
        if r["decision"] == "KEEP":
            self.assertGreaterEqual(r["delta"], self.proj.settings["min_delta"] - 1e-9)
            self.assertEqual(r["holdout_gate"], "PASS")


    def test_fast_same_size_edits_are_not_cached(self):
        # Regression: __pycache__ keys on (mtime seconds, size). Two same-size edits within
        # one second used to re-run the previous experiment's config.
        p = self.dir / "config.py"
        orig = p.read_text()
        p.write_text(orig.replace('"remove_stopwords": False', '"remove_stopwords": True '))
        a = self.proj.run("a")
        p.write_text(orig.replace('"expand_synonyms": False', '"expand_synonyms": True '))
        b = self.proj.run("b")
        self.assertNotEqual((a["train_score"], a["mrr"]), (b["train_score"], b["mrr"]))


class SplitTests(unittest.TestCase):
    def test_split_deterministic_and_order_free(self):
        items = load_eval_set(DEMO / "evalset.jsonl")
        a = split(items, 0.3, "s")
        b = split(list(reversed(items)), 0.3, "s")
        self.assertEqual({i["id"] for i in a[1]}, {i["id"] for i in b[1]})
        self.assertEqual(len(a[1]), round(len(items) * 0.3))


class InstructionSyncTests(unittest.TestCase):
    def test_skill_body_matches_packaged_instructions(self):
        # One source of truth: SKILL.md (Claude/Pi/Codex skill) and `rag-autotune agents-md`
        # (AGENTS.md for any agent) must say the same thing.
        root = Path(__file__).resolve().parent.parent
        skill = (root / "skills" / "rag-autotune" / "SKILL.md").read_text()
        body = skill.split("---", 2)[2].lstrip("\n")
        packaged = (root / "rag_autotune" / "instructions.md").read_text()
        self.assertEqual(body, packaged)


if __name__ == "__main__":
    unittest.main()
