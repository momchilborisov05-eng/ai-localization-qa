"""Claude integration with a fake client: batching, caching, repair round, review, pipeline, demo run."""

import io
import json
import re
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from fakes import SAMPLES, FakeClaude, echo_translator

from lxml import etree

from locqa import llm
from locqa.cli import main
from locqa.glossary import Glossary
from locqa.llm import LLM, LLMError, Reviewer, Translator, make_batches
from locqa.models import Segment
from locqa.pipeline import Options, Resources, qa_file, translate_file
from locqa.tags import protected
from locqa.tm import TranslationMemory
from locqa.xliff import NS12

GLOSSARY = Glossary.load(SAMPLES / "glossary.csv")


def segments(n):
    return [Segment(f"s{i}", f"Sentence number {i} with ⟦1⟧tag⟦/1⟧.") for i in range(n)]


class TranslatorTests(unittest.TestCase):
    def test_batches_and_cached_system_prompt(self):
        client = FakeClaude(echo_translator())
        translator = Translator(LLM(client), GLOSSARY, "en", "de", "STYLE", "CONTEXT", batch_size=4)
        outcome = translator.translate(segments(10))
        self.assertEqual(len(client.requests), 3)  # 4 + 4 + 2
        self.assertEqual(len(outcome.targets), 10)
        systems = [json.dumps(r["system"], ensure_ascii=False) for r in client.requests]
        self.assertEqual(len(set(systems)), 1)  # byte-identical prefix -> cache hits
        block = client.requests[0]["system"][0]
        self.assertEqual(block["cache_control"], {"type": "ephemeral"})
        self.assertIn("STYLE", block["text"])
        self.assertIn("| binding | Bindung |", block["text"])
        self.assertIn("| MIPS | DO NOT TRANSLATE", block["text"])
        fmt = client.requests[0]["output_config"]["format"]
        self.assertEqual(fmt["type"], "json_schema")
        self.assertEqual(translator.llm.usage.requests, 3)
        self.assertEqual(translator.llm.usage.cached_tokens, 2700)

    def test_fallback_parameters_for_default_model(self):
        client = FakeClaude(echo_translator())
        Translator(LLM(client), GLOSSARY, "en", "de").translate(segments(1))
        request = client.requests[0]
        self.assertEqual(request["model"], llm.MODEL)
        if llm.MODEL in llm.FALLBACK_MODELS:
            self.assertEqual(request["fallbacks"], "default")
            self.assertEqual(request["betas"], ["server-side-fallback-2026-07-01"])

    def test_payload_has_glossary_hits_tm_matches_and_limits(self):
        tm = TranslationMemory.load(SAMPLES / "tm.tmx")
        seg = Segment("ps-009", "Recommended rider weight: 70–95 kg.", max_chars=40, note="Spec line")
        seg2 = Segment("ps-010", "Pair it with our ⟦1⟧Lockdown bindings⟦/1⟧.")
        client = FakeClaude(echo_translator())
        matches = {"ps-009": tm.lookup(seg.source, "de")}
        Translator(LLM(client), GLOSSARY, "en", "de").translate([seg, seg2], matches)
        items = {i["id"]: i for i in client.payloads()[0]["segments"]}
        self.assertEqual(items["ps-009"]["max_chars"], 40)
        self.assertEqual(items["ps-009"]["context"], "Spec line")
        self.assertEqual(items["ps-009"]["tm_matches"][0]["target"], "Empfohlenes Fahrergewicht: 60–85 kg.")
        self.assertEqual({g["source"] for g in items["ps-010"]["glossary"]}, {"binding", "Lockdown"})

    def test_repair_round_fixes_lost_tags(self):
        def answer(payload):
            out = []
            for s in payload["segments"]:
                if "previous_attempt" in s:
                    self.assertIn("missing ⟦/1⟧", s["problem"])
                    out.append({"id": s["id"], "target": "Fixed ⟦1⟧tag⟦/1⟧."})
                elif s["id"] == "s1":
                    out.append({"id": s["id"], "target": "Broken ⟦1⟧tag."})
                else:
                    out.append({"id": s["id"], "target": s["source"]})
            return {"translations": out}
        client = FakeClaude(answer)
        outcome = Translator(LLM(client), GLOSSARY, "en", "de").translate(segments(3))
        self.assertEqual(outcome.repaired, ["s1"])
        self.assertEqual(outcome.unresolved, [])
        self.assertEqual(outcome.targets["s1"], "Fixed ⟦1⟧tag⟦/1⟧.")
        self.assertEqual(len(client.requests), 2)

    def test_missing_segment_is_requested_again_and_unresolved_is_reported(self):
        client = FakeClaude(lambda p: {"translations": []})
        outcome = Translator(LLM(client), GLOSSARY, "en", "de").translate(segments(2))
        self.assertEqual(sorted(outcome.unresolved), ["s0", "s1"])

    def test_refusal_and_truncation_raise_clear_errors(self):
        with self.assertRaisesRegex(LLMError, "declined"):
            Translator(LLM(FakeClaude(echo_translator(), "refusal")), GLOSSARY, "en", "de").translate(segments(1))
        with self.assertRaisesRegex(LLMError, "batch-size"):
            Translator(LLM(FakeClaude(echo_translator(), "max_tokens")), GLOSSARY, "en", "de").translate(segments(1))

    def test_make_batches_respects_character_budget(self):
        segs = [Segment(str(i), "x" * 400) for i in range(5)]
        self.assertEqual([len(b) for b in make_batches(segs, 10, max_chars=1000)], [2, 2, 1])


class ReviewerTests(unittest.TestCase):
    def test_review_issues_are_parsed_and_unknown_ids_ignored(self):
        def answer(payload):
            return {"segments": [
                {"id": "a", "issues": [{"category": "accuracy", "severity": "critical", "explanation": "Wrong number.",
                                        "suggested_fix": "31,6 mm"}]},
                {"id": "ghost", "issues": [{"category": "style", "severity": "minor", "explanation": "x",
                                            "suggested_fix": "y"}]},
            ]}
        client = FakeClaude(answer)
        issues = Reviewer(LLM(client), GLOSSARY, "en", "bg").review(
            [Segment("a", "31.6 mm", "31,8 мм"), Segment("b", "Empty", "")])
        self.assertEqual([(i.segment_id, i.severity) for i in issues], [("a", "critical")])
        self.assertEqual([s["id"] for s in client.payloads()[0]["segments"]], ["a"])  # empty targets not sent
        schema = client.requests[0]["output_config"]["format"]["schema"]
        enum = schema["properties"]["segments"]["items"]["properties"]["issues"]["items"]["properties"]["severity"]["enum"]
        self.assertEqual(enum, ["minor", "major", "critical"])


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.resources = Resources(glossary=GLOSSARY, tm=TranslationMemory.load(SAMPLES / "tm.tmx"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_exact_tm_matches_skip_the_llm_and_tags_survive(self):
        client = FakeClaude(echo_translator("DE "))
        run = translate_file(SAMPLES / "product_copy.en.xlf", "de", self.tmp, self.resources, client)
        sent = [s["id"] for p in client.payloads() if p["task"] == "translate" for s in p["segments"]]
        self.assertNotIn("ps-006", sent)
        self.assertEqual(len(sent), 27)
        by_id = {s.id: s for s in run.segments}
        self.assertEqual(by_id["ps-006"].origin, "tm")
        self.assertEqual(by_id["ps-006"].target, "Kostenloser Versand ab 100 € Bestellwert.")
        self.assertEqual(run.summary()["tm_reused"], 3)
        self.assertEqual(run.mqm.score, 100.0)
        self.assertEqual(Path(run.output_file).name, "product_copy.de.xlf")
        out = self.tmp / "product_copy.de.xlf"
        root = etree.parse(str(out)).getroot()
        for tu in root.iter(f"{{{NS12}}}trans-unit"):
            src = etree.tostring(tu.find(f"{{{NS12}}}source"))
            tgt = etree.tostring(tu.find(f"{{{NS12}}}target"))
            self.assertEqual(src.count(b"<g "), tgt.count(b"<g "))
            self.assertEqual(src.count(b"<x "), tgt.count(b"<x "))

    def test_no_review_option(self):
        client = FakeClaude(echo_translator())
        run = translate_file(SAMPLES / "ui_strings.en.json", "bg", self.tmp, self.resources, client,
                             Options(review=False))
        self.assertIsNone(run.mqm)
        self.assertTrue(all(p["task"] == "translate" for p in client.payloads()))
        for seg in run.segments:
            self.assertEqual(protected(seg.source), protected(seg.target))

    def test_qa_only_mode_needs_no_client(self):
        run = qa_file(SAMPLES / "ui_strings.en.json", "de", self.resources, target_path=SAMPLES / "ui_strings.en.json")
        self.assertIsNone(run.mqm)
        self.assertGreater(run.summary()["qa_warnings"], 0)  # identical to source


class DemoEndToEndTests(unittest.TestCase):
    """Runs the real CLI demo (canned responses, no network) and checks every output."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        with redirect_stdout(io.StringIO()) as out:
            cls.code = main(["demo", "--out", str(cls.tmp)])
        cls.stdout = out.getvalue()
        cls.report = json.loads((cls.tmp / "report.json").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_exit_code_and_outputs(self):
        self.assertEqual(self.code, 0)
        for name in ("product_copy.bg.xlf", "product_copy.de.xlf", "ui_strings.bg.json", "ui_strings.de.json",
                     "report.html", "report.json", "feedback_report.json", "tm_updated.tmx", "glossary_candidates.csv"):
            self.assertTrue((self.tmp / name).exists(), name)
        self.assertIn("demo mode", self.stdout)

    def test_report_numbers(self):
        self.assertTrue(self.report["demo"])
        runs = {(Path(r["input_file"]).name, r["target_lang"]): r["summary"] for r in self.report["runs"]}
        bg, de = runs[("product_copy.en.xlf", "bg")], runs[("product_copy.en.xlf", "de")]
        self.assertEqual((bg["segments"], bg["tm_reused"], bg["repaired"]), (30, 3, 1))
        self.assertEqual(bg["qa_by_check"], {"glossary": 2, "numbers": 2, "length": 1, "untranslated": 1})
        self.assertFalse(bg["mqm"]["passed"])
        self.assertEqual(de["qa_by_check"], {"glossary": 2, "consistency": 2, "whitespace": 1})
        self.assertTrue(de["mqm"]["passed"])
        self.assertEqual(runs[("ui_strings.en.json", "de")]["repaired"], 1)

    def test_translated_files_keep_tags_and_placeholders(self):
        from locqa.formats import load_document
        source = load_document(SAMPLES / "product_copy.en.xlf")
        target = load_document(self.tmp / "product_copy.bg.xlf")
        self.assertEqual(target.target_lang, "bg")
        for s, t in zip(source.segments, target.segments):
            self.assertEqual(s.tags, t.tags)
            self.assertEqual(sorted(protected(s.source).elements()), sorted(protected(t.target).elements()), s.id)
        data = json.loads((self.tmp / "ui_strings.de.json").read_text(encoding="utf-8"))
        self.assertEqual(data["cart"]["items"], "{count} Artikel")

    def test_html_report_is_standalone(self):
        html = (self.tmp / "report.html").read_text(encoding="utf-8")
        self.assertNotIn("__REPORT_DATA__", html)
        # data is embedded; the only external scripts allowed are optional CDN ones (GSAP animation)
        for src in re.findall(r'<script src="([^"]+)"', html):
            self.assertTrue(src.startswith("https://cdnjs.cloudflare.com/"), src)
        self.assertIn("Linguist feedback", html)
        self.assertIn('"demo": true', html.replace('"demo":true', '"demo": true'))


if __name__ == "__main__":
    unittest.main()
