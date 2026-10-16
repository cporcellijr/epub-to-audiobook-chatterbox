import importlib.util
import math
import unittest
from pathlib import Path

from audiobook_generator.core.cast_llm import PROMPTS

MODULE = Path(__file__).resolve().parents[2] / "docs" / "chatterbox-edition" / "experiments" / "cast_decision" / "decision_client.py"
spec = importlib.util.spec_from_file_location("cast_decision_client", MODULE)
dc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dc)


def window_prompt(roster, narrator, passage, ids):
    return PROMPTS["window"].format(roster=roster, narrator=narrator, passage=passage, ids=ids,
                                    moods_shape="", moods_rule="")


def tok(text, p, top=()):
    return {"token": text, "logprob": math.log(p), "bytes": list(text.encode()),
            "top_logprobs": [{"token": t, "logprob": math.log(q), "bytes": list(t.encode())} for t, q in top]}


class CandidatesTest(unittest.TestCase):
    def test_window_prompt_gives_roster_keys_narrator_labels_and_always(self):
        text = window_prompt(
            "Rory, @k2 = Tom Baker (male; aliases: Tom, Mr. Baker), Scarlett",
            PROMPTS["narrator"].format(name="Rory", aliases=" (aliases: Aurora)"),
            '[Mom] "Dinner." She waited.\n\n[#3] "Coming!" [Scarlett?] "Me too." [sic] he wrote. [#4] “No.”',
            "3, 4")
        self.assertEqual(dc.candidates_from_prompt(text), ["Rory", "@k2", "Scarlett", "Mom", "I", "unknown"])
        self.assertEqual(dc.ids_from_prompt(text), [3, 4])

    def test_empty_roster_and_review_header(self):
        text = PROMPTS["review"].format(roster=PROMPTS["roster_empty"], narrator="",
                                        passage='[Ann] "Hi." [#7] "Hello."', ids="7")
        self.assertEqual(dc.candidates_from_prompt(text), ["Ann", "I", "unknown"])
        self.assertEqual(dc.ids_from_prompt(text), [7])

    def test_prompt_without_ids(self):
        self.assertEqual(dc.ids_from_prompt("no ids here"), [])

    def test_schema_constrains_every_id_and_keeps_reply_order(self):
        schema = dc.speaker_schema([3, 4], ["Rory", "unknown"])
        self.assertEqual(list(schema["properties"]), ["speakers", "characters"])
        speakers = schema["properties"]["speakers"]
        self.assertEqual(speakers["required"], ["3", "4"])
        self.assertEqual(speakers["properties"]["4"], {"enum": ["Rory", "unknown"]})
        self.assertFalse(speakers["additionalProperties"])


class ConfidenceTest(unittest.TestCase):
    CANDIDATES = ["Rory", "Scarlett", "Mom", "I", "unknown"]
    HEAD = [tok('{"', 1), tok("spe", 1), tok("akers", 1), tok('":', 1), tok(' {"', 1), tok("3", 1), tok('":', 1),
            tok(' "', 1)]
    TAIL = [tok('"},', 1, [('"},', 1)]), tok(' "', 1), tok("characters", 1), tok('":', 1), tok(" []}", 1)]
    REPLY = '{"speakers": {"3": "Rory"}, "characters": []}'

    def test_decision_probability_renormalises_over_candidates(self):
        tokens = self.HEAD + [tok("R", 0.6, [("R", 0.6), ("Sc", 0.2), ("Mom", 0.1), ("X", 0.05)]),
                              tok("ory", 0.99, [("ory", 0.99)])] + self.TAIL
        result = dc.answer_confidences(self.REPLY, tokens, self.CANDIDATES)[3]
        self.assertEqual(result["answer"], "Rory")
        self.assertTrue(result["is_candidate"])
        self.assertAlmostEqual(result["p_first"], 0.6)
        self.assertAlmostEqual(result["p_seq"], 0.6 * 0.99)
        self.assertAlmostEqual(result["p_decision"], 0.6 / 0.9)  # "X" starts no candidate
        self.assertEqual(result["runner_up"], "Scarlett")
        self.assertAlmostEqual(result["p_runner_up"], 0.2 / 0.9)

    def test_constrained_choice_outside_the_top_list_adds_its_own_mass(self):
        reply = self.REPLY.replace("Rory", "Mom")
        tokens = self.HEAD + [tok("Mom", 0.01, [("Rory", 0.7), ("X", 0.2)])] + self.TAIL
        result = dc.answer_confidences(reply, tokens, self.CANDIDATES)[3]
        self.assertAlmostEqual(result["p_decision"], 0.01 / 0.71)
        self.assertEqual(result["runner_up"], "Rory")

    def test_new_name_has_no_decision_probability(self):
        reply = self.REPLY.replace("Rory", "Peter")
        tokens = self.HEAD + [tok("Peter", 0.5, [("Peter", 0.5), ("R", 0.3)])] + self.TAIL
        result = dc.answer_confidences(reply, tokens, self.CANDIDATES)[3]
        self.assertFalse(result["is_candidate"])
        self.assertIsNone(result["p_decision"])
        self.assertAlmostEqual(result["p_seq"], 0.5)

    def test_tokens_that_do_not_spell_the_reply_give_nothing(self):
        self.assertEqual(dc.answer_confidences(self.REPLY + " ", self.HEAD + self.TAIL, self.CANDIDATES), {})


spec = importlib.util.spec_from_file_location("cast_decision_analysis", MODULE.with_name("analyse_confidence.py"))
ac = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ac)


class AnalysisTest(unittest.TestCase):
    def test_auroc_counts_ties_as_half(self):
        self.assertEqual(ac.auroc([0.1], [0.9]), 1.0)
        self.assertEqual(ac.auroc([0.9], [0.1]), 0.0)
        self.assertEqual(ac.auroc([1.0, 0.2], [1.0, 1.0]), 0.75)
        self.assertIsNone(ac.auroc([], [1.0]))

    def test_last_answer_is_the_review_and_missing_decision_counts_as_zero(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "decisions.jsonl"
            first = {"answer": "A", "p_seq": 0.9, "p_first": 0.9, "p_decision": 0.95}
            review = {"answer": "B", "p_seq": 0.4, "p_first": 0.4, "p_decision": None}
            path.write_text("\n".join(json.dumps(r) for r in (
                {"ids": [5], "chapter": 2, "answers": {"5": first}},
                {"ids": [], "chapter": 2, "answers": {}},
                {"ids": [5], "chapter": 2, "answers": {"5": review}})), encoding="utf-8")
            entry = ac.load_confidences(path)[(2, 5)]
        self.assertEqual((entry["first"]["answer"], entry["last"]["answer"], entry["answers"]), ("A", "B", 2))
        self.assertEqual(ac.confidence(entry, "p_decision", "last"), 0.0)
        self.assertEqual(ac.confidence(entry, "p_seq", "first"), 0.9)

    def test_matched_flags_the_same_number_of_lowest_confidence_lines(self):
        pairs = [(0.2, True, False), (0.3, False, True), (0.9, True, True), (0.99, False, False)]
        result = ac.matched(pairs)
        self.assertEqual(result["baseline"], {"flagged": 2, "wrong_caught": 1, "right_flagged": 1})
        self.assertEqual(result["confidence"], {"flagged": 2, "wrong_caught": 1, "right_flagged": 1})
        rows = ac.threshold_rows([(c, w) for c, w, _ in pairs], (0.5,))
        self.assertEqual((rows[0]["flagged"], rows[0]["wrong_caught"], rows[0]["right_flagged"]), (2, 1, 1))


if __name__ == "__main__":
    unittest.main()
