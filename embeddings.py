"""Embedding algorithms — pure stdlib, deterministic across processes."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Callable, Dict, List

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _blake(key: bytes, data: bytes, n: int) -> int:
    h = hashlib.blake2b(data, digest_size=8, key=key)
    return int.from_bytes(h.digest(), "big") % n


_K_IDX = b"idx-key-0001"
_K_SIGN = b"sgn-key-0001"
_K_PROJ = b"prj-key-0001"


def _idx(token: str, dim: int) -> int:
    return _blake(_K_IDX, token.encode("utf-8"), dim)


def _sign(token: str) -> int:
    return 1 if _blake(_K_SIGN, token.encode("utf-8"), 2) == 0 else -1


def _l2_normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


def _char_ngrams(token: str, ns=(3, 4, 5)) -> List[str]:
    wrapped = "<" + token + ">"
    grams: List[str] = []
    for n in ns:
        if len(wrapped) < n:
            continue
        for i in range(len(wrapped) - n + 1):
            grams.append(wrapped[i : i + n])
    return grams


# --- Model 1: word-level hashing trick (Weinberger et al. 2009) -------------

DIM_BOW = 256


def embed_hash_bow(text: str) -> List[float]:
    vec = [0.0] * DIM_BOW
    for tok in _tokenize(text):
        vec[_idx(tok, DIM_BOW)] += _sign(tok)
    return _l2_normalize(vec)


# --- Model 2: char n-gram hashing (FastText-style sub-word) -----------------

DIM_NGRAM = 512


def embed_hash_ngram(text: str) -> List[float]:
    vec = [0.0] * DIM_NGRAM
    for tok in _tokenize(text):
        for gram in _char_ngrams(tok):
            vec[_idx(gram, DIM_NGRAM)] += _sign(gram)
    return _l2_normalize(vec)


# --- Model 3: random projection of bag-of-words (Johnson–Lindenstrauss) -----

DIM_PROJ = 128


def _rademacher(token: str, j: int) -> int:
    """Deterministic ±1 entry of the projection matrix at (token, j)."""
    data = token.encode("utf-8") + b"|" + j.to_bytes(2, "big")
    return 1 if _blake(_K_PROJ, data, 2) == 0 else -1


def embed_random_proj(text: str) -> List[float]:
    counts: Dict[str, int] = {}
    for tok in _tokenize(text):
        counts[tok] = counts.get(tok, 0) + 1
    vec = [0.0] * DIM_PROJ
    scale = 1.0 / math.sqrt(DIM_PROJ)
    for tok, c in counts.items():
        for j in range(DIM_PROJ):
            vec[j] += c * _rademacher(tok, j) * scale
    return _l2_normalize(vec)


# --- Perf models: cheap, fixed-size, no real computation -------------------
# Useful for benchmarking client / network / serialization overhead without
# the embedding algorithm itself being on the hot path.

DIM_PERF_TINY = 8
DIM_PERF_SMALL = 1536   # matches OpenAI text-embedding-3-small
DIM_PERF_LARGE = 3072   # matches OpenAI text-embedding-3-large

_PERF_TINY = [0.0] * DIM_PERF_TINY
_PERF_SMALL = [0.0] * DIM_PERF_SMALL
# Non-zero constant so JSON byte size reflects the dim — each float
# serializes to ~20 chars rather than collapsing to "0.0".
_PERF_LARGE_FILL = 1.0 / math.sqrt(DIM_PERF_LARGE)
_PERF_LARGE = [_PERF_LARGE_FILL] * DIM_PERF_LARGE


def embed_perf_zero_tiny(text: str) -> List[float]:
    return _PERF_TINY


def embed_perf_zero_small(text: str) -> List[float]:
    return _PERF_SMALL


def embed_perf_fixed_large(text: str) -> List[float]:
    return _PERF_LARGE


# --- Registry ---------------------------------------------------------------

EmbedFn = Callable[[str], List[float]]

MODELS: Dict[str, EmbedFn] = {
    "hash-bow-256": embed_hash_bow,
    "hash-ngram-512": embed_hash_ngram,
    "random-proj-128": embed_random_proj,
    "perf-zero-8": embed_perf_zero_tiny,
    "perf-zero-1536": embed_perf_zero_small,
    "perf-fixed-3072": embed_perf_fixed_large,
}

MODEL_DIMS: Dict[str, int] = {
    "hash-bow-256": DIM_BOW,
    "hash-ngram-512": DIM_NGRAM,
    "random-proj-128": DIM_PROJ,
    "perf-zero-8": DIM_PERF_TINY,
    "perf-zero-1536": DIM_PERF_SMALL,
    "perf-fixed-3072": DIM_PERF_LARGE,
}


def count_tokens(text: str) -> int:
    return len(_tokenize(text))
