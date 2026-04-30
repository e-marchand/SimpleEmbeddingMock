"""Optional plugin: real semantic embeddings via sentence-transformers.

Activates only if `sentence_transformers` is importable. Models are loaded
lazily on first embed call (the library downloads weights on first use, which
can take a moment and requires network).

Configure via env var (comma-separated HuggingFace model names):
    EMBEDMOCK_ST_MODELS=all-MiniLM-L6-v2,all-mpnet-base-v2

Default: all-MiniLM-L6-v2 (~23M params, 384 dims, fast).
"""

from __future__ import annotations

import os
from importlib.util import find_spec
from typing import Dict, List

OWNER = "sentence-transformers"
DEFAULT_MODELS = ("all-MiniLM-L6-v2",)


def register() -> Dict[str, dict]:
    if find_spec("sentence_transformers") is None:
        return {}

    raw = os.environ.get("EMBEDMOCK_ST_MODELS", "")
    names = [n.strip() for n in raw.split(",") if n.strip()] or list(DEFAULT_MODELS)

    cache: Dict[str, object] = {}

    def _load(name: str):
        if name not in cache:
            from sentence_transformers import SentenceTransformer  # lazy

            cache[name] = SentenceTransformer(name)
        return cache[name]

    def _make_embed(name: str):
        def embed(text: str) -> List[float]:
            model = _load(name)
            vec = model.encode(text, normalize_embeddings=True)
            return [float(x) for x in vec]

        return embed

    return {
        name: {"embed": _make_embed(name), "owned_by": OWNER}
        for name in names
    }
