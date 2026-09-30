from __future__ import annotations

import base64
import json
import tempfile
import unittest
import zlib
from pathlib import Path

from scripts.team23_tokenizer import Team23Tokenizer, load_embedded_vocab


class Team23TokenizerTests(unittest.TestCase):
    def test_embedded_ids_and_eos_are_preserved(self):
        vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "hello": 4}
        encoded = base64.b64encode(zlib.compress(json.dumps(vocab).encode())).decode()
        notebook = {"cells": [{"source": [f'_EMBEDDED_VOCAB_ASDF_B64 = "{encoded}"\n']}]}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "boilerplate.ipynb"
            path.write_text(json.dumps(notebook), encoding="utf-8")
            restored = load_embedded_vocab(path, "source")
        tokenizer = Team23Tokenizer(restored, is_source=True)
        self.assertEqual(tokenizer.encode("hello"), [1, 4, 2])
        self.assertEqual(tokenizer.decode([1, 4, 2, 4]), "hello")
        self.assertNotEqual(tokenizer.bos_id, tokenizer.eos_id)

    def test_invalid_special_ids_are_rejected(self):
        vocab = {"<pad>": 0, "<bos>": 2, "<eos>": 1, "<unk>": 3}
        encoded = base64.b64encode(zlib.compress(json.dumps(vocab).encode())).decode()
        notebook = {"cells": [{"source": [f'_EMBEDDED_VOCAB_ASDF_B64 = "{encoded}"\n']}]}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "boilerplate.ipynb"
            path.write_text(json.dumps(notebook), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "special-token IDs"):
                load_embedded_vocab(path, "source")


if __name__ == "__main__":
    unittest.main()
