"""Every deterministic QA check, with a passing and a failing case."""

import unittest

from fakes import SAMPLES

from locqa.glossary import Glossary
from locqa.models import Segment
from locqa.qa_checks import (check_consistency, check_glossary, check_length, check_numbers, check_punctuation,
                             check_tags, check_units, check_untranslated, check_whitespace, numbers_in, run_qa)
from locqa.tags import protected_diff, tag_order_problem

GLOSSARY = Glossary.load(SAMPLES / "glossary.csv")


def seg(source, target, **kw):
    return Segment("s1", source, target, **kw)


def checks(issues):
    return [(i.check, i.severity) for i in issues]


class TagTests(unittest.TestCase):
    def test_tokens_and_placeholders(self):
        self.assertEqual(check_tags(seg("Hi ⟦1⟧{name}⟦/1⟧, %s", "Hallo ⟦1⟧{name}⟦/1⟧, %s"), "de", GLOSSARY), [])
        missing = check_tags(seg("Hi ⟦1⟧there⟦/1⟧", "Hallo ⟦1⟧da"), "de", GLOSSARY)
        self.assertEqual(checks(missing), [("tags", "error")])
        self.assertIn("⟦/1⟧", missing[0].message)
        renamed = check_tags(seg("{count} items", "{anzahl} Artikel"), "de", GLOSSARY)
        self.assertEqual(len(renamed), 2)  # {count} missing and {anzahl} unexpected
        html = check_tags(seg("<b>Sale:</b> now", "<b>Sale: jetzt"), "de", GLOSSARY)
        self.assertEqual(checks(html), [("tags", "error")])

    def test_tag_order(self):
        self.assertIsNone(tag_order_problem("⟦1⟧a⟦2⟧b⟦/2⟧⟦/1⟧ ⟦3⟧"))
        self.assertIsNotNone(tag_order_problem("⟦/1⟧a⟦1⟧"))
        self.assertIsNotNone(tag_order_problem("⟦1⟧a⟦2⟧b⟦/1⟧⟦/2⟧"))
        issues = check_tags(seg("⟦1⟧a⟦/1⟧", "⟦/1⟧a⟦1⟧"), "de", GLOSSARY)
        self.assertIn("Tag order", issues[0].message)

    def test_protected_diff(self):
        self.assertEqual(protected_diff("a %1$s ${x}", "b ${x}"), (["%1$s"], []))


class NumberAndUnitTests(unittest.TestCase):
    def test_locale_number_formats_compare_equal(self):
        self.assertEqual(numbers_in("20,000 mm and 30.9 mm"), numbers_in("20.000 mm und 30,9 mm"))
        self.assertEqual(numbers_in("20,000 mm"), numbers_in("20 000 мм"))
        self.assertEqual(numbers_in("Size {size} ⟦12⟧"), numbers_in("Größe {size} ⟦12⟧"))  # tag/placeholder digits ignored

    def test_number_mismatch(self):
        ok = seg("Compatible with 30.9 mm and 31.6 mm.", "Kompatibel mit 30,9 mm und 31,6 mm.")
        self.assertEqual(check_numbers(ok, "de", GLOSSARY), [])
        bad = seg("Compatible with 30.9 mm and 31.6 mm.", "Съвместима с 30,9 мм и 31,8 мм.")
        self.assertEqual(checks(check_numbers(bad, "bg", GLOSSARY)), [("numbers", "error")] * 2)

    def test_units_accept_cyrillic_symbols(self):
        self.assertEqual(check_units(seg("Weighs 340 g, 150 mm travel.", "Тежи 340 г, ход 150 мм."), "bg", GLOSSARY), [])
        self.assertEqual(check_units(seg("12–16 °C, 30% off", "12–16 °C, 30 % Rabatt"), "de", GLOSSARY), [])
        issues = check_units(seg("Weighs 340 g.", "Wiegt 340."), "de", GLOSSARY)
        self.assertEqual(checks(issues), [("units", "error")])


class PunctuationWhitespaceTests(unittest.TestCase):
    def test_ending_punctuation(self):
        self.assertEqual(check_punctuation(seg("Ride more.", "Mehr fahren."), "de", GLOSSARY), [])
        self.assertEqual(check_punctuation(seg("Questions? ⟦1⟧Help⟦/1⟧.", "Fragen? ⟦1⟧Hilfe⟦/1⟧."), "de", GLOSSARY), [])
        self.assertEqual(checks(check_punctuation(seg("Ride more!", "Mehr fahren"), "de", GLOSSARY)),
                         [("punctuation", "warning")])
        self.assertEqual(checks(check_punctuation(seg("A (b) c", "A (b c"), "de", GLOSSARY)),
                         [("punctuation", "warning")])

    def test_whitespace(self):
        self.assertEqual(check_whitespace(seg("Press ⟦1⟧ to drop.", "Drücke ⟦1⟧, um abzusenken."), "de", GLOSSARY), [])
        self.assertEqual(checks(check_whitespace(seg("A, b", "A,  b"), "de", GLOSSARY)), [("whitespace", "warning")])
        self.assertEqual(checks(check_whitespace(seg("A, b", "A , b"), "de", GLOSSARY)), [("whitespace", "warning")])
        self.assertEqual(checks(check_whitespace(seg("A b ", "A b"), "de", GLOSSARY)), [("whitespace", "warning")])


class LengthGlossaryUntranslatedTests(unittest.TestCase):
    def test_length_counts_visible_characters_only(self):
        self.assertEqual(check_length(seg("x", "⟦1⟧12345⟦/1⟧", max_chars=5), "de", GLOSSARY), [])
        issues = check_length(seg("x", "123456", max_chars=5), "de", GLOSSARY)
        self.assertEqual(issues[0].message, "6 characters, limit is 5")

    def test_glossary_and_dnt(self):
        src = "Pair it with our ⟦1⟧Lockdown bindings⟦/1⟧ for edge control on hardpack."
        good = seg(src, "Комбинирай го с нашите ⟦1⟧Lockdown автомати⟦/1⟧ за контрол върху твърд сняг.")
        self.assertEqual(check_glossary(good, "bg", GLOSSARY), [])
        bad = seg(src, "Комбинирай го с нашите ⟦1⟧Локдаун крепления⟦/1⟧ за контрол върху твърд сняг.")
        self.assertEqual(sorted(checks(check_glossary(bad, "bg", GLOSSARY))), [("dnt", "error"), ("glossary", "error")])

    def test_untranslated(self):
        self.assertEqual(checks(check_untranslated(seg("Hello there", ""), "de", GLOSSARY)), [("untranslated", "error")])
        self.assertEqual(checks(check_untranslated(seg("Shred 150 Dropper Post", "Shred 150 Dropper Post"), "bg", GLOSSARY)),
                         [("untranslated", "warning")])
        # identical is fine when it only contains DNT names and terms that stay the same in German
        self.assertEqual(check_untranslated(seg("Drift Pro Snowboard", "Drift Pro Snowboard"), "de", GLOSSARY), [])
        self.assertEqual(check_untranslated(seg("XS - XXL", "XS - XXL"), "de", GLOSSARY), [])


class FileLevelTests(unittest.TestCase):
    def test_consistency(self):
        segments = [Segment("a", "Ships in 2 days.", "Versand in 2 Tagen."),
                    Segment("b", "Ships  in 2 days.", "Lieferung in 2 Tagen."),
                    Segment("c", "Other.", "Anderes.")]
        issues = check_consistency(segments)
        self.assertEqual(sorted(i.segment_id for i in issues), ["a", "b"])
        segments[1].target = "Versand in 2 Tagen."
        self.assertEqual(check_consistency(segments), [])

    def test_run_qa_on_clean_segment(self):
        clean = [Segment("x", "Free shipping on orders over €100.", "Kostenloser Versand ab 100 € Bestellwert.")]
        self.assertEqual(run_qa(clean, "de", GLOSSARY), [])


if __name__ == "__main__":
    unittest.main()
