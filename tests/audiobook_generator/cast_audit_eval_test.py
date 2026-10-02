import json
import tempfile
import unittest
from pathlib import Path

from tests.audiobook_generator import cast_audit_eval as evaluator


def _row(document, line, speaker):
    return {"document": document, "line": line, "speaker_id": speaker, "confidence": "high",
            "uncertainty": None, "review_status": "reviewed", "supporting_paragraphs": []}


# An invented book: Wren narrates; Rob and Dex talk in chapter 1, Mara in chapter 2.
REFERENCE = {"narrator_speaker_id": "wren", "lines": [
    _row(1, 1, "rob"), _row(1, 2, "rob"), _row(1, 3, "wren"), _row(1, 4, "dex"), _row(1, 5, "wren"),
    _row(2, 1, "wren"), _row(2, 2, "mara"), _row(2, 3, "mara"),
]}


@unittest.skipUnless(evaluator.REFERENCE.is_file(), "the private source-linked reference is not on this machine")
class TestPrivateReference(unittest.TestCase):

    def test_reference_covers_every_line_and_links_clean_source_paragraphs(self):
        data = json.loads(evaluator.REFERENCE.read_text(encoding="utf-8"))
        paragraphs = {doc["document"]: {p["id"]: p["text"] for p in doc["paragraphs"]}
                      for doc in data["source_documents"]}
        self.assertEqual(len({(row["document"], row["line"]) for row in data["lines"]}), len(data["lines"]))
        for row in data["lines"]:
            self.assertTrue(row["speaker_id"])
            self.assertIn(row["confidence"], {"high", "medium", "low"})
            self.assertIn("uncertainty", row)
            self.assertIn(row["review_status"], {"reviewed", "reviewed_ambiguous"})
            for pid in row["supporting_paragraphs"]:
                self.assertTrue(paragraphs[row["document"]][pid])
        self.assertFalse(any("assigned " in text and "voice " in text
                             for doc in paragraphs.values() for text in doc.values()))


class TestEvaluator(unittest.TestCase):

    def test_reports_attribution_splits_merges_unresolved_and_routes_separately(self):
        predictions = [
            {"document": 1, "line": 1, "speaker": "rob", "voice": "A.wav"},
            {"document": 1, "line": 2, "speaker": "alternate rob", "voice": "B.wav"},
            {"document": 1, "line": 4, "speaker": "rob", "voice": "A.wav"},
            {"document": 2, "line": 2, "speaker": None},
            {"document": 1, "line": 3, "speaker": "wren", "voice": "Voice-1.wav"},
            {"document": 1, "line": 5, "speaker": "narrator", "voice": "Voice-2.wav"},
        ]
        result = evaluator.evaluate(REFERENCE, predictions, {"wren": "Chosen.wav"})
        self.assertGreater(result["misattributions"]["count"], 0)
        self.assertGreater(result["identity_splits"]["pair_count"], 0)
        self.assertGreater(result["false_merges"]["pair_count"], 0)
        self.assertEqual(result["unresolved_lines"]["count"], 1)
        self.assertFalse(result["narrator_consistency"]["speaker_consistent"])
        self.assertFalse(result["narrator_consistency"]["voice_consistent"])
        self.assertEqual(result["wrong_voice_routes"]["count"], 2)
        self.assertEqual(result["wrong_voice_routes"]["checked_lines"], 2)

    def test_any_single_narrator_voice_is_consistent(self):
        predictions = [{"document": row["document"], "line": row["line"], "speaker": row["speaker_id"],
                        "voice": "Gianna.wav" if row["speaker_id"] == "wren" else None}
                       for row in REFERENCE["lines"]]
        result = evaluator.evaluate(REFERENCE, predictions)
        self.assertTrue(result["narrator_consistency"]["voice_consistent"])
        self.assertEqual(result["narrator_consistency"]["voices"], ["Gianna.wav"])
        self.assertTrue(result["narrator_consistency"]["speaker_consistent"])
        self.assertEqual(result["wrong_voice_routes"]["checked_lines"], 0)

    def test_narrator_missing_coverage_is_not_reported_as_consistent(self):
        narrator = evaluator.evaluate(REFERENCE, [])["narrator_consistency"]
        self.assertFalse(narrator["speaker_consistent"])
        self.assertFalse(narrator["voice_consistent"])
        self.assertEqual(narrator["scored_lines"], 0)
        self.assertEqual(narrator["missing_lines"], narrator["expected_lines"])

    def test_saved_cast_keys_and_aliases_match_canonical_identity_without_hiding_splits(self):
        prediction = {
            "narrator_voice": "RunNarrator.wav",
            "characters": {
                "wren": {"name": "Wren", "voice": None, "aliases": ["I", "narrator"]},
                "mara_1": {"name": "Mara", "voice": "Gianna.wav"},
                "mara_2": {"name": "Mara", "voice": "Lucy.wav"},
            },
            "book_tone": {"point_of_view": "first", "pov_key": "wren"},
            "chapters": {
                "a": {"number": 1, "lines": {"3": "wren"}},
                "b": {"number": 2, "narrator": "mara_1", "lines": {"2": "mara_1", "3": "mara_2"}},
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cast.json"
            path.write_text(json.dumps(prediction), encoding="utf-8")
            rows, routes, aliases = evaluator._load_predictions(path, REFERENCE)
        result = evaluator.evaluate(REFERENCE, rows, routes, aliases)
        self.assertEqual(result["misattributions"]["count"], 0)
        self.assertGreater(result["identity_splits"]["pair_count"], 0)
        self.assertEqual(result["wrong_voice_routes"]["count"], 0)
        mara_routes = {(row["document"], int(row["line"])): row["voice"] for row in rows if row["document"] == 2}
        self.assertEqual(mara_routes, {(2, 2): "Gianna.wav", (2, 3): "Lucy.wav"})

    def _scored(self, prediction, reference, aliases=None):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cast.json"
            path.write_text(json.dumps(prediction), encoding="utf-8")
            rows, routes, speaker_aliases = evaluator._load_predictions(path, reference, aliases)
        return evaluator.evaluate(reference, rows, routes, speaker_aliases)

    def test_alias_file_unnamed_narrator_and_chapter_narrators(self):
        # Two invented stories: Wren tells chapter 1, an unnamed "I" (really Dex) tells chapter 2.
        reference = {"narrator_speaker_id": None, "chapter_narrators": {"1": "wren", "2": "dex"}, "lines": [
            _row(1, 1, "rob"), _row(1, 2, "wren"), _row(2, 1, "dex"), _row(2, 2, "dex"), _row(2, 3, "mara")]}
        prediction = {"characters": {"wren": {"name": "Wren"}, "robbie": {"name": "Robbie"},
                                     "narrator 2": {"name": "The Narrator"}, "mara": {"name": "Mara"}},
                      "chapters": {"a": {"number": 1, "narrator": "wren", "lines": {"1": "robbie", "2": "wren"}},
                                   "b": {"number": 2, "narrator": "narrator 2",
                                         "lines": {"1": "narrator 2", "2": "narrator 2", "3": "narrator 2"}}}}
        result = self._scored(prediction, reference, {"Robbie": "rob"})
        self.assertEqual(result["misattributions"]["lines"],  # the unnamed "I" is Dex; Mara's line is not his
                         [{"document": 2, "line": 3, "expected": "mara", "actual": "dex", "confidence": "high"}])
        self.assertEqual(result["chapter_narrators"], {"checked": 2, "wrong": []})
        prediction["chapters"]["b"]["narrator"] = "wren"
        self.assertEqual(self._scored(prediction, reference, {"Robbie": "rob"})["chapter_narrators"]["wrong"],
                         [{"document": 2, "expected": "dex", "actual": "wren"}])
        self.assertEqual(evaluator.summary(result), "wrong 1 of 5, unresolved 0, split pairs 0, merged pairs 2, "
                                                    "wrong chapter narrators 0 of 2, wrong voice routes 0")


if __name__ == "__main__":
    unittest.main()
