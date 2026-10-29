"""MQM scoring and the linguist feedback loop."""

import csv
import shutil
import tempfile
import unittest
from pathlib import Path

from fakes import SAMPLES

from locqa.errors import LocQAError
from locqa.feedback import (FeedbackRow, analyse, effort_bucket, glossary_candidates, load_feedback,
                            qa_before_after, segment_metrics, substitutions, tm_leverage, update_tm,
                            write_candidates_csv)
from locqa.glossary import Glossary
from locqa.mqm import ReviewIssue, mqm_score
from locqa.tm import TranslationMemory

GLOSSARY = Glossary.load(SAMPLES / "glossary.csv")


def issue(severity, category="accuracy"):
    return ReviewIssue("s", category, severity, "x", "y")


class MQMTests(unittest.TestCase):
    def test_no_issues_is_100(self):
        result = mqm_score([], 250)
        self.assertEqual(result.score, 100.0)
        self.assertTrue(result.passed)

    def test_weights_and_normalization(self):
        result = mqm_score([issue("minor"), issue("major", "fluency"), issue("critical", "terminology")], 200)
        self.assertEqual(result.penalty, 31)
        self.assertEqual(result.score, 84.5)  # 100 * (1 - 31/200)
        self.assertFalse(result.passed)
        self.assertEqual(result.by_severity, {"minor": 1, "major": 1, "critical": 1})
        self.assertEqual(result.by_category["fluency"], 1)
        self.assertEqual(result.by_category["style"], 0)

    def test_threshold_and_floor(self):
        self.assertTrue(mqm_score([issue("major")], 100, threshold=95).passed)    # exactly 95.0
        self.assertFalse(mqm_score([issue("major")], 100, threshold=96).passed)
        self.assertEqual(mqm_score([issue("critical")] * 10, 50).score, 0.0)


class MetricsTests(unittest.TestCase):
    def test_segment_metrics(self):
        same = segment_metrics("Mehr fahren.", "Mehr fahren.")
        self.assertTrue(same["unchanged"])
        self.assertEqual(same["normalized_edit_distance"], 0)
        self.assertEqual(same["effort"], "none")
        m = segment_metrics("Versand innerhalb von 2 Arbeitstagen.", "Versand innerhalb von 2 Werktagen.")
        self.assertEqual(m["word_edits"], 1)
        self.assertEqual(m["edit_rate"], round(1 / 6, 3))
        self.assertGreater(m["char_edit_distance"], 0)
        self.assertEqual(m["effort"], "medium")  # 7 character edits / 37 = 0.19

    def test_effort_buckets(self):
        self.assertEqual([effort_bucket(x) for x in (0, 0.1, 0.15, 0.3, 0.41, 0.9)],
                         ["none", "light", "light", "medium", "heavy", "heavy"])

    def test_substitutions(self):
        self.assertEqual(substitutions("Lieferung innerhalb von 2 Arbeitstagen.", "Versand innerhalb von 2 Werktagen."),
                         [("Lieferung", "Versand"), ("Arbeitstagen", "Werktagen")])
        self.assertEqual(substitutions("30,9 мм и 31,8 мм", "30,9 мм и 31,6 мм"), [])  # number fixes are not terms


class FeedbackLoopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.rows = load_feedback(SAMPLES / "linguist_feedback.csv")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_load_and_validation(self):
        self.assertEqual(len(self.rows), 15)
        bad = self.tmp / "bad.csv"
        bad.write_text("segment_id,source\n1,a\n", encoding="utf-8")
        with self.assertRaisesRegex(LocQAError, "missing column"):
            load_feedback(bad)

    def test_candidates_separate_new_terms_from_ignored_glossary_terms(self):
        candidates = {(c.target_lang, c.mt_term, c.post_edit_term): c for c in glossary_candidates(self.rows, GLOSSARY)}
        ignored = candidates[("de", "Arbeitstagen", "Werktagen")]
        self.assertEqual((ignored.status, ignored.glossary_term, ignored.occurrences),
                         ("glossary-not-followed", "business days", 2))
        self.assertEqual(candidates[("bg", "крепления", "автомати")].glossary_term, "binding")
        new = candidates[("de", "Belüftungsöffnungen", "Lüftungsöffnungen")]
        self.assertEqual(new.status, "new-candidate")
        recurring = candidates[("bg", "Изпраща се в рамките на", "Доставка до")]
        self.assertEqual(recurring.occurrences, 2)
        out = self.tmp / "c.csv"
        write_candidates_csv(list(candidates.values()), out)
        with open(out, encoding="utf-8") as f:
            self.assertEqual(len(list(csv.DictReader(f))), len(candidates))

    def test_tm_update_raises_leverage(self):
        tm = TranslationMemory.load(SAMPLES / "tm.tmx")
        from locqa.formats import load_document
        sources = [s.source for s in load_document(SAMPLES / "product_copy.en.xlf").segments]
        before = tm_leverage(sources, tm, "de")
        counts = update_tm(self.rows, tm)
        after = tm_leverage(sources, tm, "de")
        self.assertEqual(counts, {"added": 13, "updated": 0, "unchanged": 2})
        self.assertEqual(before["exact"], 3)
        self.assertEqual(after["exact"], 10)
        self.assertEqual(sum(after.values()), len(sources))

    def test_qa_before_after_and_aggregates(self):
        qa = qa_before_after(self.rows, GLOSSARY)
        self.assertGreater(qa["de"]["mt_errors"], 0)
        self.assertEqual(qa["de"]["post_edit_errors"], 0)
        self.assertEqual(qa["bg"]["post_edit_errors"], 0)
        report = analyse(self.rows, GLOSSARY)
        self.assertEqual(report["overall"]["segments"], 15)
        self.assertEqual(report["overall"]["unchanged_pct"], 20.0)
        self.assertEqual(sum(report["overall"]["effort"].values()), 15)
        self.assertEqual(report["languages"]["de"]["categories"]["terminology"], 3)

    def test_rows_without_post_edit_are_ignored(self):
        path = self.tmp / "fb.csv"
        path.write_text("segment_id,target_lang,source,mt,post_edit\n1,de,A.,B.,\n2,de,C.,D.,E.\n", encoding="utf-8")
        self.assertEqual([r.segment_id for r in load_feedback(path)], ["2"])
        self.assertIsInstance(load_feedback(path)[0], FeedbackRow)


if __name__ == "__main__":
    unittest.main()
