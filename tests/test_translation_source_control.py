import unittest

from scripts.translation_source_control import controls


class SourceControlTests(unittest.TestCase):
    def test_source_independent_collapse_cannot_pass_control(self):
        result = controls(["one repeated sentence"] * 3, ["a b c d", "one repeated sentence", "other text"])
        self.assertEqual(result["unique_predictions"], 1)
        for summary in result["wrong_source"].values():
            self.assertEqual(summary["at_least_aligned"], 2)

    def test_perfect_distinct_outputs_beat_all_wrong_source_rotations(self):
        phrases = ["alpha beta gamma delta", "one two three four", "red green blue yellow"]
        result = controls(phrases, phrases)
        self.assertEqual(result["wrong_source_rotations"], 2)
        for key in ("corpus_bleu4", "chrf_plus_plus", "exact_match"):
            self.assertEqual(result["wrong_source"][key]["at_least_aligned"], 0)
