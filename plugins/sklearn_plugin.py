"""Optional plugin: scikit-learn HashingVectorizer.

Mirrors the core `hash-bow-256` algorithm at higher dimension using sklearn's
own implementation — useful as a parity/sanity reference.
"""

from __future__ import annotations

import math
from importlib.util import find_spec
from typing import Dict, List

OWNER = "scikit-learn"
DIM = 1024


def register() -> Dict[str, dict]:
    if find_spec("sklearn") is None:
        return {}

    from sklearn.feature_extraction.text import HashingVectorizer

    vectorizer = HashingVectorizer(
        n_features=DIM,
        alternate_sign=True,
        norm="l2",
        analyzer="word",
        lowercase=True,
    )

    def embed(text: str) -> List[float]:
        x = vectorizer.transform([text])
        # x is a 1xDIM scipy sparse matrix; densify and unwrap.
        dense = x.toarray()[0]
        return [float(v) for v in dense]

    return {
        f"sklearn-hashing-{DIM}": {"embed": embed, "owned_by": OWNER},
    }
