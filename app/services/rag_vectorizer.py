from __future__ import annotations

import hashlib
import math
import re


RAG_VECTOR_SIZE = 96


def vectorize_text(text: str) -> list[float]:
    vector = [0.0] * RAG_VECTOR_SIZE
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    if not tokens:
        return vector

    frequencies: dict[str, int] = {}
    for token in tokens:
        frequencies[token] = frequencies.get(token, 0) + 1

    for token, frequency in frequencies.items():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:2], "big") % RAG_VECTOR_SIZE
        weight = 1.0 + math.log1p(frequency)
        vector[bucket] += weight

    for left, right in zip(tokens, tokens[1:]):
        digest = hashlib.sha256(f"{left}_{right}".encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:2], "big") % RAG_VECTOR_SIZE
        vector[bucket] += 0.7

    for token in tokens:
        if len(token) < 5:
            continue
        for index in range(0, len(token) - 2):
            trigram = token[index : index + 3]
            digest = hashlib.sha256(f"tri:{trigram}".encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:2], "big") % RAG_VECTOR_SIZE
            vector[bucket] += 0.15

    magnitude = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [round(value / magnitude, 6) for value in vector]
