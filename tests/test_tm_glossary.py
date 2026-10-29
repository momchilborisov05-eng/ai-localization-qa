"""Fuzzy matching, translation memory and glossary."""

import shutil
import tempfile
import unittest
from pathlib import Path

from fakes import SAMPLES

from locqa.errors import LocQAError
from locqa.fuzzy import fuzzy_score, levenshtein, tokenize
from locqa.glossary import Glossary, GlossaryTerm
from locqa.tm import TMEntry, TranslationMemory, same_language


class FuzzyTests(unittest.TestCase):
    def test_levenshtein_on_characters_and_words(self):
        self.assertEqual(levenshtein("kitten", "sitting"), 3)
        self.assertEqual(levenshtein("", "abc"), 3)
        self.assertEqual(levenshtein(["a", "b", "c"], ["a", "c"]), 1)

    def test_tokenizer_keeps_numbers_tags_and_placeholders_whole(self):
        self.assertEqual(tokenize("Fits 30.9 mm ⟦1⟧{size}⟦/1⟧."), ["Fits", "30.9", "mm", "⟦1⟧", "{size}", "⟦/1⟧", "."])

    def test_scores(self):
        self.assertEqual(fuzzy_score("Ships within 2 days.", "Ships within 2 days."), 100)
        self.assertEqual(fuzzy_score("Ships  within 2 days.", "Ships within 2 days."), 99)
        one_word = fuzzy_score("Available in sizes XS to XXL.", "Available in sizes XS to XL.")
        self.assertTrue(85 <= one_word <= 95, one_word)
        self.assertLess(fuzzy_score("Machine wash cold.", "Free shipping on orders over €100."), 40)
        self.assertEqual(fuzzy_score("", "x"), 0)

    def test_more_differences_score_lower(self):
        base = "Recommended rider weight: 70–95 kg."
        self.assertGreater(fuzzy_score(base, "Recommended rider weight: 70–90 kg."),
                           fuzzy_score(base, "Recommended board length: 150–160 cm."))


class TranslationMemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.tm = TranslationMemory.load(SAMPLES / "tm.tmx")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_tmx_loading_skips_native_code(self):
        self.assertEqual(len(self.tm), 24)  # 12 units x 2 target languages
        entry = next(e for e in self.tm.entries if e.source.startswith("Ideal for") and e.target_lang == "de")
        self.assertEqual(entry.source, "Ideal for beginners and intermediate riders.")
        self.assertEqual(entry.target, "Ideal für Einsteiger und fortgeschrittene Rider.")

    def test_lookup_exact_and_fuzzy_per_language(self):
        exact = self.tm.lookup("Free shipping on orders over €100.", "bg")
        self.assertEqual(exact[0].score, 100)
        self.assertEqual(exact[0].entry.target, "Безплатна доставка за поръчки над 100 €.")
        fuzzy = self.tm.lookup("Recommended rider weight: 70–95 kg.", "de-DE")
        self.assertEqual(len(fuzzy), 1)
        self.assertTrue(75 <= fuzzy[0].score < 100)
        self.assertEqual(fuzzy[0].entry.target_lang, "de")
        self.assertEqual(self.tm.lookup("Something completely different here.", "de"), [])
        self.assertEqual(self.tm.lookup("Free shipping on orders over €100.", "fr"), [])

    def test_add_update_and_tmx_round_trip(self):
        self.assertEqual(self.tm.add("New sentence.", "Neuer Satz.", "en", "de"), "added")
        self.assertEqual(self.tm.add("New sentence.", "Neuer Satz.", "en", "de"), "unchanged")
        self.assertEqual(self.tm.add("New sentence.", "Ein neuer Satz.", "en", "de"), "updated")
        out = self.tmp / "tm.tmx"
        self.tm.save_tmx(out)
        again = TranslationMemory.load(out)
        self.assertEqual(len(again), 25)
        self.assertEqual(again.lookup("New sentence.", "de")[0].entry.target, "Ein neuer Satz.")
        self.assertIn("linguist-feedback", out.read_text(encoding="utf-8"))

    def test_feedback_entries_win_ties(self):
        tm = TranslationMemory([TMEntry("A b c.", "old", "en", "de", "tm"), TMEntry("A b c!", "new", "en", "de", "feedback")])
        tm.entries.append(TMEntry("A b c.", "fb", "en", "de", "feedback"))
        self.assertEqual(tm.lookup("A b c.", "de")[0].entry.target, "fb")

    def test_csv_tm_and_errors(self):
        csv_path = self.tmp / "tm.csv"
        csv_path.write_text("source,target,target_lang\nHello.,Hallo.,de\n,skip,de\n", encoding="utf-8")
        tm = TranslationMemory.load(csv_path)
        self.assertEqual(len(tm), 1)
        bad = self.tmp / "bad.csv"
        bad.write_text("source,target\nHello.,Hallo.\n", encoding="utf-8")
        with self.assertRaisesRegex(LocQAError, "target_lang"):
            TranslationMemory.load(bad)
        with self.assertRaisesRegex(LocQAError, "not found"):
            TranslationMemory.load(self.tmp / "missing.tmx")

    def test_same_language(self):
        self.assertTrue(same_language("de", "de-DE"))
        self.assertTrue(same_language("de_AT", "DE"))
        self.assertFalse(same_language("bg", "de"))


class GlossaryTests(unittest.TestCase):
    def setUp(self):
        self.glossary = Glossary.load(SAMPLES / "glossary.csv")

    def term(self, source):
        return next(t for t in self.glossary.terms if t.source == source)

    def test_loading(self):
        self.assertEqual(len(self.glossary), 30)
        self.assertTrue(self.term("MIPS").dnt)
        self.assertEqual(self.term("seam").target_for("de"), ["Naht", "Nähte"])
        self.assertEqual(self.term("binding").preferred("bg"), "автомати")
        self.assertEqual(self.term("binding").preferred("de-AT"), "Bindung")

    def test_source_matching_plurals_and_word_boundaries(self):
        found = {t.source for t in self.glossary.find_in("Pair it with our Lockdown bindings on hardpack.", "bg")}
        self.assertEqual(found, {"binding", "Lockdown", "hardpack"})
        self.assertFalse(self.term("leash").found_in("unleashed power"))
        self.assertTrue(self.term("fin box").found_in("Includes FCS II fin boxes"))
        self.assertFalse(self.term("MIPS").found_in("mips"))  # case-sensitive term

    def test_target_compliance_with_inflection_and_variants(self):
        wetsuit = self.term("wetsuit")
        self.assertTrue(wetsuit.satisfied_by("Неопреновият костюм е топъл.", "bg"))
        self.assertTrue(wetsuit.satisfied_by("bis zu 30 % Rabatt auf Neoprenanzüge", "de"))
        self.assertTrue(self.term("seam").satisfied_by("verklebte Nähte", "de"))
        self.assertTrue(self.term("business days").satisfied_by("Versand in 2 Werktagen.", "de"))
        self.assertFalse(self.term("business days").satisfied_by("Versand in 2 Arbeitstagen.", "de"))
        self.assertTrue(self.term("Lockdown").satisfied_by("unserer Lockdown-Bindung", "de"))
        self.assertFalse(self.term("Lockdown").satisfied_by("unserer Lockdown-Bindung".lower(), "de"))

    def test_dnt_terms_count_for_every_language(self):
        self.assertIn(self.term("MIPS"), self.glossary.for_language("fr"))
        self.assertNotIn(self.term("wetsuit"), self.glossary.for_language("fr"))

    def test_longest_terms_first_and_errors(self):
        g = Glossary([GlossaryTerm("base", {"de": ["Basis"]}), GlossaryTerm("sintered base", {"de": ["Sinterbelag"]})])
        self.assertEqual(g.terms[0].source, "sintered base")
        with self.assertRaisesRegex(LocQAError, "not found"):
            Glossary.load(Path("nope.csv"))


if __name__ == "__main__":
    unittest.main()
