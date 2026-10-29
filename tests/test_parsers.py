"""XLIFF 1.2 / 2.0 and JSON: parsing, tag masking, round trips and output naming."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from fakes import SAMPLES

from lxml import etree

from locqa.errors import LocQAError
from locqa.formats import load_document, output_name
from locqa.xliff import NS12, NS20

XLIFF12_WITH_TARGETS = """<?xml version="1.0" encoding="UTF-8"?>
<xliff version="1.2" xmlns="urn:oasis:names:tc:xliff:document:1.2" xmlns:m="http://www.memsource.com/mxlf/2.0">
  <file original="a.docx" source-language="en" target-language="de" datatype="x-undefined">
    <body>
      <trans-unit id="1">
        <source>Click <g id="5" ctype="bold">here</g> or <x id="7"/> now &amp; save.</source>
        <target>Klicke <g id="5" ctype="bold">hier</g> oder <x id="7"/> jetzt &amp; spare.</target>
      </trans-unit>
      <trans-unit id="2" m:locked="true">
        <source>Locked segment</source>
      </trans-unit>
      <trans-unit id="3" translate="no">
        <source>Not translatable</source>
      </trans-unit>
      <trans-unit id="4">
        <source>Before <ph id="1">&lt;br/&gt;</ph> after</source>
      </trans-unit>
    </body>
  </file>
</xliff>
"""


class TempDirTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class Xliff12Tests(TempDirTest):
    def test_parses_sample_with_notes_limits_and_tags(self):
        doc = load_document(SAMPLES / "product_copy.en.xlf")
        self.assertEqual(doc.format_name, "xliff12")
        self.assertEqual(doc.source_lang, "en")
        self.assertEqual(len(doc.segments), 30)
        first = doc.segments[0]
        self.assertEqual(first.id, "ps-001")
        self.assertEqual(first.max_chars, 40)
        self.assertIn("Product title", first.note)
        seg3 = doc.segments[2]
        self.assertTrue(seg3.source.startswith("⟦1⟧Key features:⟦/1⟧ 4/3 mm"))
        self.assertEqual(seg3.tags["⟦1⟧"], '<g id="1" ctype="bold">')
        self.assertEqual(seg3.tags["⟦/1⟧"], "</g>")
        seg18 = next(s for s in doc.segments if s.id == "ps-018")
        self.assertEqual(seg18.tags["⟦1⟧"], '<x id="1" ctype="x-icon"/>')

    def test_round_trip_preserves_inline_tags_exactly(self):
        doc = load_document(SAMPLES / "product_copy.en.xlf")
        for seg in doc.segments:
            seg.target = "DE " + seg.source
        out = self.tmp / "out.xlf"
        self.assertEqual(doc.save(out, "de"), [])

        root = etree.parse(str(out)).getroot()
        q = lambda n: f"{{{NS12}}}{n}"  # noqa: E731
        self.assertEqual(root.find(q("file")).get("target-language"), "de")
        for tu in root.iter(q("trans-unit")):
            source, target = tu.find(q("source")), tu.find(q("target"))
            self.assertIsNotNone(target, tu.get("id"))
            self.assertEqual(target.get("state"), "translated")
            self.assertEqual([(etree.QName(c).localname, dict(c.attrib)) for c in source.iter()][1:],
                             [(etree.QName(c).localname, dict(c.attrib)) for c in target.iter()][1:])
        # reading the output again gives the same masked targets
        again = load_document(out)
        self.assertEqual([s.target for s in again.segments], ["DE " + s.source for s in doc.segments])

    def test_existing_targets_get_the_source_token_numbers(self):
        path = self.tmp / "bi.xlf"
        path.write_text(XLIFF12_WITH_TARGETS, encoding="utf-8")
        doc = load_document(path)
        self.assertEqual([s.id for s in doc.segments], ["1", "4"])  # locked and translate="no" skipped
        seg = doc.segments[0]
        self.assertEqual(seg.source, "Click ⟦1⟧here⟦/1⟧ or ⟦2⟧ now & save.")
        self.assertEqual(seg.target, "Klicke ⟦1⟧hier⟦/1⟧ oder ⟦2⟧ jetzt & spare.")
        self.assertEqual(doc.segments[1].tags["⟦1⟧"], '<ph id="1">&lt;br/&gt;</ph>')

    def test_special_characters_are_escaped_on_write(self):
        path = self.tmp / "bi.xlf"
        path.write_text(XLIFF12_WITH_TARGETS, encoding="utf-8")
        doc = load_document(path)
        doc.segments[0].target = "A < B & ⟦1⟧C⟦/1⟧ ⟦2⟧"
        doc.save(self.tmp / "o.xlf", "de")
        self.assertIn("A &lt; B &amp; <g", (self.tmp / "o.xlf").read_text(encoding="utf-8"))
        self.assertEqual(load_document(self.tmp / "o.xlf").segments[0].target, "A < B & ⟦1⟧C⟦/1⟧ ⟦2⟧")

    def test_broken_tags_leave_target_empty_instead_of_invalid_xml(self):
        doc = load_document(SAMPLES / "product_copy.en.xlf")
        seg = doc.segments[2]
        seg.target = "⟦/1⟧broken⟦1⟧ order"
        doc.segments[1].needs_review = True
        doc.segments[1].target = "fine"
        out = self.tmp / "o.xlf"
        self.assertEqual(doc.save(out, "bg"), ["ps-003"])
        root = etree.parse(str(out)).getroot()  # still valid XML
        units = {tu.get("id"): tu for tu in root.iter(f"{{{NS12}}}trans-unit")}
        self.assertEqual(units["ps-003"].find(f"{{{NS12}}}target").get("state"), "needs-translation")
        self.assertEqual(units["ps-002"].find(f"{{{NS12}}}target").get("state"), "needs-review-translation")

    def test_invalid_xml_and_unsupported_files_give_clear_errors(self):
        bad = self.tmp / "bad.xlf"
        bad.write_text("<xliff><file>", encoding="utf-8")
        with self.assertRaisesRegex(LocQAError, "not valid XML"):
            load_document(bad)
        other = self.tmp / "x.xlf"
        other.write_text("<root/>", encoding="utf-8")
        with self.assertRaisesRegex(LocQAError, "not XLIFF"):
            load_document(other)
        with self.assertRaisesRegex(LocQAError, "Unsupported file type"):
            load_document(SAMPLES / "glossary.csv")
        with self.assertRaisesRegex(LocQAError, "not found"):
            load_document(self.tmp / "missing.xlf")


class Xliff20Tests(TempDirTest):
    def test_parses_units_segments_nested_pc_and_ph(self):
        doc = load_document(SAMPLES / "care_guide.en.xlf")
        self.assertEqual(doc.format_name, "xliff20")
        self.assertEqual((doc.source_lang, doc.target_lang), ("en", "de"))
        ids = [s.id for s in doc.segments]
        self.assertEqual(ids, ["cg-1", "cg-2/s1", "cg-2/s2", "cg-3"])
        self.assertEqual(doc.segments[0].max_chars, 30)
        self.assertEqual(doc.segments[1].note, "Instructions shown on the product page.")
        self.assertEqual(doc.segments[1].source, "Rinse with ⟦1⟧fresh water⟦/1⟧ after every session.")
        self.assertEqual(doc.segments[2].tags["⟦1⟧"], '<ph id="2" dataRef="d1"/>')
        self.assertEqual(doc.segments[3].source, "Never use a ⟦1⟧⟦2⟧hot⟦/2⟧ wash⟦/1⟧ or a dryer.")

    def test_round_trip_and_states(self):
        doc = load_document(SAMPLES / "care_guide.en.xlf")
        doc.segments[0].target = "Neoprenpflege in 3 Schritten"
        doc.segments[1].target = "Nach jeder Session mit ⟦1⟧Süßwasser⟦/1⟧ ausspülen."
        doc.segments[2].target = "Auf links trocknen, ohne direkte Sonne.⟦1⟧"
        doc.segments[3].target = "Nie ⟦1⟧⟦2⟧heiß⟦/2⟧ waschen⟦/1⟧ oder in den Trockner."
        doc.segments[3].needs_review = True
        out = self.tmp / "care.de.xlf"
        self.assertEqual(doc.save(out, "de"), [])
        text = out.read_text(encoding="utf-8")
        self.assertIn('<pc id="1" type="fmt"><pc id="2" type="fmt">heiß</pc> waschen</pc>', text)
        self.assertIn('<ph id="2" dataRef="d1"/>', text)
        self.assertIn('<data id="d1">&lt;br/&gt;</data>', text)  # originalData untouched
        root = etree.parse(str(out)).getroot()
        segs = list(root.iter(f"{{{NS20}}}segment"))
        self.assertTrue(all(s.get("state") == "translated" for s in segs))
        self.assertEqual(segs[-1].get("subState"), "locqa:needs-review")
        again = load_document(out)
        self.assertEqual([s.target for s in again.segments], [s.target for s in doc.segments])


class JsonTests(TempDirTest):
    def test_nested_json_round_trip_keeps_structure_and_non_strings(self):
        src = self.tmp / "app.en.json"
        src.write_text(json.dumps({"a": {"b": "Hello {name}", "n": 3}, "list": ["One", "Two"], "flag": True}),
                       encoding="utf-8")
        doc = load_document(src)
        self.assertEqual([s.id for s in doc.segments], ["a.b", "list.0", "list.1"])
        doc.segments[0].target = "Hallo {name}"
        doc.segments[1].target = "Eins"
        out = self.tmp / "app.de.json"
        doc.save(out, "de")
        data = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(data, {"a": {"b": "Hallo {name}", "n": 3}, "list": ["Eins", "Two"], "flag": True})

    def test_bilingual_json_qa_input(self):
        doc = load_document(SAMPLES / "ui_strings.en.json", target_path=SAMPLES / "ui_strings.en.json")
        self.assertTrue(all(s.target == s.source for s in doc.segments))
        self.assertEqual(len(doc.segments), 9)

    def test_invalid_json(self):
        bad = self.tmp / "bad.json"
        bad.write_text("{nope", encoding="utf-8")
        with self.assertRaisesRegex(LocQAError, "not valid JSON"):
            load_document(bad)


class OutputNameTests(unittest.TestCase):
    def test_replaces_source_language_suffix(self):
        self.assertEqual(output_name(Path("product_copy.en.xlf"), "en", "bg"), "product_copy.bg.xlf")
        self.assertEqual(output_name(Path("strings.json"), "en", "de"), "strings.de.json")
        self.assertEqual(output_name(Path("x.en-US.mxliff"), "en-US", "de"), "x.de.mxliff")


if __name__ == "__main__":
    unittest.main()
