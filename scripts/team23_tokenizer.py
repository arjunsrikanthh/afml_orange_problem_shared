"""Load the supplied embedded vocabularies while preserving their token IDs."""

from __future__ import annotations

import ast
import base64
import json
import re
import zlib
from pathlib import Path

EXPECTED_SPECIAL = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
EMBEDDED_NAMES = {
    "source": "_EMBEDDED_VOCAB_ASDF_B64",
    "target": "_EMBEDDED_VOCAB_ENG_B64",
}


def load_embedded_vocab(notebook_path: Path, language: str) -> dict[str, int]:
    """Extract one vocabulary from the actual assigned notebook, not a rebuilt corpus."""
    assignment = EMBEDDED_NAMES[language]
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if assignment not in source:
            continue
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == assignment for target in node.targets
            ):
                encoded = ast.literal_eval(node.value)
                vocab = json.loads(zlib.decompress(base64.b64decode(encoded)).decode("utf-8"))
                if not isinstance(vocab, dict) or any(
                    not isinstance(token, str) or not isinstance(index, int) for token, index in vocab.items()
                ):
                    raise ValueError(f"{language} vocabulary has invalid entries")
                if {name: vocab.get(name) for name in EXPECTED_SPECIAL} != EXPECTED_SPECIAL:
                    raise ValueError(f"{language} special-token IDs differ from supplied contract")
                indices = sorted(vocab.values())
                if indices != list(range(len(vocab))):
                    raise ValueError(f"{language} vocabulary IDs are not unique and contiguous")
                return vocab
    raise ValueError(f"{language} embedded vocabulary not found")


class Team23Tokenizer:
    def __init__(self, vocab: dict[str, int], *, is_source: bool):
        self.word2idx = dict(vocab)
        self.idx2word = {index: word for word, index in vocab.items()}
        self.is_source = is_source
        self.pad_id = vocab["<pad>"]
        self.bos_id = vocab["<bos>"]
        self.eos_id = vocab["<eos>"]
        self.unk_id = vocab["<unk>"]

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)

    def tokenize(self, text: str) -> list[str]:
        if self.is_source:
            return [token.strip() for token in str(text).strip().split() if token.strip()]
        tokens = []
        for token in str(text).strip().split():
            cleaned = re.sub(r"[^\w'-]", "", token, flags=re.UNICODE).lower()
            if cleaned:
                tokens.append(cleaned)
        return tokens

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        ids = [self.word2idx.get(token, self.unk_id) for token in self.tokenize(text)]
        return [self.bos_id, *ids, self.eos_id] if add_special_tokens else ids

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        words = []
        for index in ids:
            if skip_special_tokens and index == self.eos_id:
                break
            if skip_special_tokens and index in {self.pad_id, self.bos_id, self.unk_id}:
                continue
            word = self.idx2word.get(index)
            if word is not None:
                words.append(word)
        return " ".join(words)


def load_tokenizers(notebook_path: Path) -> tuple[Team23Tokenizer, Team23Tokenizer]:
    return (
        Team23Tokenizer(load_embedded_vocab(notebook_path, "source"), is_source=True),
        Team23Tokenizer(load_embedded_vocab(notebook_path, "target"), is_source=False),
    )
