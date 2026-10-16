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


class SetAndNoopTests(unittest.TestCase):
    """Regression for a real Gemma-12B run: it wrote `run -m "stem=True"` ten times without
    changing the config, and the harness scored the same config ten times."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        for f in ("adapter.py", "config.py", "corpus.jsonl", "evalset.jsonl"):
            shutil.copy(DEMO / f, self.dir / f)
        self.proj = Project(self.dir)
        self.proj.init("evalset.jsonl", "adapter.py", "config.py", k=2, metric="recall",
                       holdout_frac=0.3, min_delta=None, latency_budget_ms=None, seed="t", force=True)
        self.proj.run("baseline")

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_noop_run_is_refused_and_not_logged(self):
        n = self.proj.next_exp()
        with self.assertRaisesRegex(AutotuneError, "NOOP"):
            self.proj.run("stem=True")
        self.assertEqual(self.proj.next_exp(), n)

    def test_set_applies_and_run_reports_change(self):
        self.proj.set_values(["expand_synonyms=True"])
        r = self.proj.run("synonyms")
        self.assertIn("expand_synonyms: False -> True", r["changes"])
        self.assertNotEqual(r.get("delta"), 0.0)

    def test_set_keeps_comments_and_layout(self):
        self.proj.set_values(["chunk_size=40"])
        text = (self.dir / "config.py").read_text()
        self.assertIn('"chunk_size": 40,          # words per chunk', text)

    def test_set_unknown_key_suggests_spelling(self):
        with self.assertRaisesRegex(AutotuneError, "did you mean 'chunk_size'"):
            self.proj.set_values(["chunk_sise=40"])

    def test_set_rejects_wrong_types(self):
        for bad in ("stem=yes", "chunk_size=big", "chunk_size=4.5", "doc_aggregation=3"):
            with self.assertRaises(AutotuneError, msg=bad):
                self.proj.set_values([bad])
        self.assertEqual((self.dir / "config.py").read_text(), (DEMO / "config.py").read_text())

    def test_set_is_all_or_nothing(self):
        with self.assertRaises(AutotuneError):
            self.proj.set_values(["stem=True", "nope=1"])
        self.assertEqual((self.dir / "config.py").read_text(), (DEMO / "config.py").read_text())

    def test_bare_word_string(self):
        changes = self.proj.set_values(["doc_aggregation=sum"])
        self.assertEqual(changes, ["doc_aggregation: 'max' -> 'sum'"])


class BaselineAndPlateauTests(unittest.TestCase):
    """Regressions from a Gemma 12B run on v0.2: it changed chunk_overlap before the baseline,
    and kept going after 9 discards in a row."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        for f in ("adapter.py", "config.py", "corpus.jsonl", "evalset.jsonl"):
            shutil.copy(DEMO / f, self.dir / f)
        self.proj = Project(self.dir)
        self.proj.init("evalset.jsonl", "adapter.py", "config.py", k=2, metric="recall",
                       holdout_frac=0.3, min_delta=None, latency_budget_ms=None, seed="t", force=True)

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_baseline_refused_if_config_changed_after_init(self):
        self.proj.set_values(["chunk_overlap=3"])
        with self.assertRaisesRegex(AutotuneError, "BASELINE.*chunk_overlap: 0 -> 3"):
            self.proj.run("baseline")
        self.assertEqual(self.proj.next_exp(), 0)

    def test_reset_config_then_baseline_then_change_as_experiment(self):
        self.proj.set_values(["chunk_overlap=3"])
        self.assertEqual(self.proj.reset_config(), ["chunk_overlap: 3 -> 0"])
        self.assertEqual(self.proj.run("baseline")["decision"], "BASELINE")
        self.proj.set_values(["chunk_overlap=3"])
        r = self.proj.run("overlap")
        self.assertIn("chunk_overlap: 0 -> 3", r["changes"])

    def test_streak_counts_runs_since_last_keep(self):
        self.proj.run("baseline")
        for v in (1.0, 1.3, 1.6):
            self.proj.set_values([f"bm25_k1={v}"])
            r = self.proj.run(f"k1 {v}")
            if r["decision"] == "KEEP":
                self.assertEqual(r["streak"], 0)
        self.assertEqual(r["streak"], self.proj.non_keep_streak())


class V04Tests(unittest.TestCase):
    """From a rag-doc-qa run (MRR@3, 22 train / 10 holdout): a +0.038 gain with an intact
    holdout was rejected by a recall-sized min_delta, the gate message called every drop
    "overfitting", and a perfect baseline holdout went unremarked."""

    def test_min_delta_is_metric_aware(self):
        from rag_autotune.core import default_min_delta
        self.assertEqual(default_min_delta("recall", 22), round(1 / 22, 4))
        self.assertEqual(default_min_delta("mrr", 22), round(0.5 / 22, 4))
        self.assertLess(default_min_delta("mrr", 22), 0.0379)  # the rejected rag-doc-qa gain

    def _proj(self, metric):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d)
        for f in ("adapter.py", "config.py", "corpus.jsonl", "evalset.jsonl"):
            shutil.copy(DEMO / f, d / f)
        p = Project(d)
        p.init("evalset.jsonl", "adapter.py", "config.py", k=2, metric=metric, holdout_frac=0.3,
               min_delta=None, latency_budget_ms=None, seed="rag-autotune", force=True)
        return p

    def test_baseline_notes_saturated_holdout(self):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d)
        for f in ("adapter.py", "config.py", "corpus.jsonl", "evalset.jsonl"):
            shutil.copy(DEMO / f, d / f)
        cfg = d / "config.py"  # demo holdout is 1.0 once synonyms are on
        cfg.write_text(cfg.read_text().replace('"expand_synonyms": False', '"expand_synonyms": True'))
        p = Project(d)
        p.init("evalset.jsonl", "adapter.py", "config.py", k=2, metric="recall", holdout_frac=0.3,
               min_delta=None, latency_budget_ms=None, seed="rag-autotune", force=True)
        r = p.run("baseline")
        self.assertTrue(any("can never confirm a gain" in n for n in r.get("notes", [])))

    def test_gate_size_buckets_match_rag_doc_qa_run(self):
        # holdout MRR over 10 questions: exp 6 lost 1.0 (borderline), exp 5 lost 1.5 (clear)
        for best_h, new_h, expect in ((1.0, 0.9, "small"), (1.0, 0.85, "large")):
            lost = (best_h - new_h) * 10
            self.assertEqual("large" if lost > 1.0 + 1e-9 else "small", expect)

    def test_gate_message_reports_size_not_just_overfitting(self):
        p = self._proj("recall")
        p.run("baseline")
        p.set_values(["remove_stopwords=True"])  # known demo holdout drop
        r = p.run("stopwords")
        self.assertEqual(r["holdout_gate"], "FAIL")
        self.assertRegex(r["reason"], "drop (small|large)")
        self.assertNotIn("= overfitting", r["reason"])


if __name__ == "__main__":
    unittest.main()
