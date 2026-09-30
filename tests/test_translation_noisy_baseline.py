from __future__ import annotations

import unittest

import torch

from scripts.translation_noisy_baseline import evaluate, greedy_decode


class _FixedLogitModel(torch.nn.Module):
    def __init__(self, token_sequence: list[int], vocab_size: int = 8):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.token_sequence = token_sequence
        self.vocab_size = vocab_size

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, **_: object) -> torch.Tensor:
        step = min(tgt.size(1) - 1, len(self.token_sequence) - 1)
        logits = torch.full((1, tgt.size(1), self.vocab_size), -100.0, device=tgt.device)
        logits[:, -1, self.token_sequence[step]] = 100.0
        return logits


class TranslationNoisyBaselineTests(unittest.TestCase):
    def test_nonfinite_logits_fail_before_argmax(self):
        class NonfiniteModel(_FixedLogitModel):
            def forward(self, src, tgt, **kwargs):
                return super().forward(src, tgt, **kwargs).fill_(float("nan"))
        with self.assertRaises(FloatingPointError):
            greedy_decode(NonfiniteModel([2]), [1, 7, 2], bos_id=1, eos_id=2, pad_id=0)

    def test_metrics_reject_missing_or_misaligned_rows(self):
        for predictions, references in [([], []), (["a"], ["a", "b"]), ([], ["a"])]:
            with self.assertRaises(ValueError):
                evaluate(predictions, references)

    def test_greedy_decoder_starts_bos_and_stops_at_eos(self):
        model = _FixedLogitModel([4, 2, 5])
        generated = greedy_decode(model, [1, 7, 2], bos_id=1, eos_id=2, pad_id=0, max_len=5, device=torch.device("cpu"))
        self.assertEqual(generated, [1, 4, 2])
        self.assertEqual(model.training, False)

    def test_greedy_decoder_respects_max_len_without_eos(self):
        model = _FixedLogitModel([4, 5, 6])
        generated = greedy_decode(model, [1, 7, 2], bos_id=1, eos_id=2, pad_id=0, max_len=2, device=torch.device("cpu"))
        self.assertEqual(generated, [1, 4, 5])

    def test_metrics_are_corpus_metrics_with_signatures(self):
        metrics = evaluate(["a b", "same"], ["a b", "different"])
        # SacreBLEU's unsmoothed corpus BLEU-4 is zero here because the tiny
        # corpus has no matching 4-gram; chrF++ still captures character overlap.
        self.assertEqual(metrics["corpus_bleu4"], 0.0)
        self.assertGreater(metrics["chrf_plus_plus"], 0.0)
        self.assertEqual(metrics["exact_match"], 0.5)
        self.assertIn("sacrebleu", metrics["metric_versions"])
        self.assertTrue(metrics["metric_signatures"]["bleu"])
        self.assertTrue(metrics["metric_signatures"]["chrf_plus_plus"])


if __name__ == "__main__":
    unittest.main()
