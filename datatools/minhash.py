"""MinHash + LSH near-duplicate detection, implemented directly on numpy.

Exact hashing only catches byte-identical documents. Web corpora are full of
documents that differ by a boilerplate header or a changed date, and those are
what inflate a pretraining set without adding information.

The pipeline is the textbook one:

1. Represent a document as its set of character n-grams (character-level so it
   works for Chinese, where whitespace tokenization does not).
2. Reduce that set to a fixed-length MinHash signature. The probability that
   two signatures agree at a given position equals the Jaccard similarity of
   the underlying sets, so ``mean(sig_a == sig_b)`` estimates Jaccard.
3. Split signatures into ``bands`` bands of ``rows`` rows and bucket by band.
   Two documents become candidates if any band matches exactly, which turns an
   O(n^2) comparison into a hash join with a tunable S-curve.

Hashing detail: shingles are hashed to 31 bits and permuted with
``(a*h + b) mod (2^31 - 1)``. Operands are bounded so the uint64 arithmetic
never wraps, which keeps this an honest universal hash family rather than
relying on overflow behaviour.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

MERSENNE_PRIME = np.uint64((1 << 31) - 1)
_MAX_COEFF = 1 << 30  # keeps a*h + b below 2**63, so uint64 never wraps
_SHINGLE_CHUNK = 4096  # bound peak memory at num_perm x chunk


def shingles(text: str, ngram: int = 5) -> list[str]:
    """Character n-grams of a document. Short documents fall back to themselves."""
    if ngram < 1:
        raise ValueError("ngram must be >= 1")
    if not text:
        return []
    if len(text) <= ngram:
        return [text]
    return [text[i : i + ngram] for i in range(len(text) - ngram + 1)]


def hash_shingles(items: Iterable[str]) -> np.ndarray:
    """Hash shingles into ``[0, MERSENNE_PRIME)``, deduplicated (MinHash is set-based)."""
    unique = {s for s in items}
    if not unique:
        return np.empty(0, dtype=np.uint64)
    digests = [
        int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=4).digest(), "big")
        for s in unique
    ]
    return np.array(digests, dtype=np.uint64) % MERSENNE_PRIME


@dataclass
class MinHasher:
    """Fixed random permutation family; the same seed gives reproducible signatures."""

    num_perm: int = 128
    ngram: int = 5
    seed: int = 0

    def __post_init__(self):
        if self.num_perm < 1:
            raise ValueError("num_perm must be >= 1")
        rng = np.random.default_rng(self.seed)
        self.a = rng.integers(1, _MAX_COEFF, self.num_perm, dtype=np.uint64)
        self.b = rng.integers(0, _MAX_COEFF, self.num_perm, dtype=np.uint64)

    def signature(self, text: str) -> np.ndarray:
        """MinHash signature of one document. Empty documents get a constant signature."""
        return self.signature_from_hashes(hash_shingles(shingles(text, self.ngram)))

    def signature_from_hashes(self, hashes: np.ndarray) -> np.ndarray:
        if hashes.size == 0:
            return np.full(self.num_perm, MERSENNE_PRIME, dtype=np.uint64)
        best = np.full(self.num_perm, MERSENNE_PRIME, dtype=np.uint64)
        for start in range(0, hashes.size, _SHINGLE_CHUNK):
            chunk = hashes[start : start + _SHINGLE_CHUNK]
            permuted = (self.a[:, None] * chunk[None, :] + self.b[:, None]) % MERSENNE_PRIME
            np.minimum(best, permuted.min(axis=1), out=best)
        return best

    def signatures(self, texts: Sequence[str]) -> np.ndarray:
        """(n_docs, num_perm) signature matrix."""
        if not len(texts):
            return np.empty((0, self.num_perm), dtype=np.uint64)
        return np.stack([self.signature(t) for t in texts])


def estimate_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Fraction of agreeing positions, which is an unbiased Jaccard estimate."""
    if sig_a.shape != sig_b.shape:
        raise ValueError(f"signature shapes differ: {sig_a.shape} vs {sig_b.shape}")
    return float(np.mean(sig_a == sig_b))


def choose_bands(num_perm: int, threshold: float) -> tuple[int, int]:
    """Pick (bands, rows) whose S-curve midpoint sits closest to ``threshold``.

    The probability that two documents with Jaccard ``s`` share at least one band
    is ``1 - (1 - s**rows)**bands``; its midpoint is near ``(1/bands)**(1/rows)``.
    """
    if not 0 < threshold < 1:
        raise ValueError("threshold must be in (0, 1)")
    best, best_error = (num_perm, 1), float("inf")
    for bands in range(1, num_perm + 1):
        if num_perm % bands:
            continue
        rows = num_perm // bands
        error = abs(lsh_threshold(bands, rows) - threshold)
        if error < best_error:
            best, best_error = (bands, rows), error
    return best


def lsh_threshold(bands: int, rows: int) -> float:
    """Approximate Jaccard at which a candidate pair becomes more likely than not."""
    return (1.0 / bands) ** (1.0 / rows)


def lsh_candidate_pairs(signatures: np.ndarray, bands: int, rows: int) -> set[tuple[int, int]]:
    """Index pairs sharing at least one identical band."""
    n_docs, num_perm = signatures.shape
    if bands * rows != num_perm:
        raise ValueError(f"bands*rows ({bands}*{rows}) must equal num_perm ({num_perm})")
    pairs: set[tuple[int, int]] = set()
    for band in range(bands):
        buckets: dict[bytes, list[int]] = {}
        block = signatures[:, band * rows : (band + 1) * rows]
        for index in range(n_docs):
            buckets.setdefault(block[index].tobytes(), []).append(index)
        for members in buckets.values():
            if len(members) < 2:
                continue
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    pairs.add((members[i], members[j]))
    return pairs


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        root_x, root_y = self.find(x), self.find(y)
        if root_x != root_y:
            # Keep the lower index as the root so clusters report in file order.
            low, high = sorted((root_x, root_y))
            self.parent[high] = low

    def groups(self) -> dict[int, list[int]]:
        clusters: dict[int, list[int]] = {}
        for index in range(len(self.parent)):
            clusters.setdefault(self.find(index), []).append(index)
        return clusters


def cluster_near_duplicates(
    signatures: np.ndarray,
    threshold: float = 0.8,
    bands: int | None = None,
    rows: int | None = None,
) -> list[list[int]]:
    """Group near-duplicate documents. Returns clusters of size >= 2, in file order.

    LSH proposes candidates; every candidate is then verified against the full
    signature, so the band configuration only trades recall for speed and cannot
    introduce pairs below ``threshold``.
    """
    if signatures.shape[0] < 2:
        return []
    num_perm = signatures.shape[1]
    if bands is None or rows is None:
        bands, rows = choose_bands(num_perm, threshold)

    union = UnionFind(signatures.shape[0])
    for i, j in lsh_candidate_pairs(signatures, bands, rows):
        if estimate_jaccard(signatures[i], signatures[j]) >= threshold:
            union.union(i, j)
    clusters = [sorted(members) for members in union.groups().values() if len(members) > 1]
    return sorted(clusters, key=lambda members: members[0])
