import numpy as np
import pytest

from datatools.minhash import (
    MERSENNE_PRIME,
    MinHasher,
    UnionFind,
    choose_bands,
    cluster_near_duplicates,
    estimate_jaccard,
    hash_shingles,
    lsh_candidate_pairs,
    lsh_threshold,
    shingles,
)


def true_jaccard(a: str, b: str, ngram: int = 5) -> float:
    sa, sb = set(shingles(a, ngram)), set(shingles(b, ngram))
    return len(sa & sb) / len(sa | sb)


def test_shingles_slide_by_one_character():
    assert shingles("abcde", 3) == ["abc", "bcd", "cde"]


def test_shingles_of_short_text_fall_back_to_the_whole_string():
    assert shingles("ab", 5) == ["ab"]
    assert shingles("", 5) == []


def test_shingles_reject_invalid_ngram():
    with pytest.raises(ValueError):
        shingles("abc", 0)


def test_hash_shingles_stay_in_range_and_deduplicate():
    hashes = hash_shingles(["a", "b", "a", "b"])
    assert hashes.dtype == np.uint64
    assert hashes.size == 2
    assert np.all(hashes < MERSENNE_PRIME)


def test_permutation_arithmetic_never_overflows_uint64():
    """a*h + b must stay below 2**63 or the modulus becomes meaningless."""
    hasher = MinHasher(num_perm=64, seed=3)
    worst = hasher.a.max().astype(object) * int(MERSENNE_PRIME) + hasher.b.max().astype(object)
    assert worst < 2**63


def test_identical_documents_get_identical_signatures():
    hasher = MinHasher(num_perm=64, seed=0)
    text = "海绵宝宝住在比奇堡的一个菠萝里面，他喜欢抓水母。"
    assert np.array_equal(hasher.signature(text), hasher.signature(text))


def test_signatures_are_reproducible_across_instances():
    text = "reproducibility matters for data pipelines"
    assert np.array_equal(MinHasher(seed=7).signature(text), MinHasher(seed=7).signature(text))


def test_empty_document_signature_is_constant():
    hasher = MinHasher(num_perm=16)
    assert np.all(hasher.signature("") == MERSENNE_PRIME)


def test_signature_estimates_true_jaccard():
    """The defining property: agreement rate between signatures tracks set overlap."""
    hasher = MinHasher(num_perm=512, ngram=5, seed=0)
    a = "The quick brown fox jumps over the lazy dog near the riverbank at dawn."
    b = "The quick brown fox jumps over the lazy cat near the riverbank at dusk."

    estimated = estimate_jaccard(hasher.signature(a), hasher.signature(b))

    assert estimated == pytest.approx(true_jaccard(a, b), abs=0.08)


def test_unrelated_documents_have_near_zero_similarity():
    hasher = MinHasher(num_perm=256, seed=0)
    a = "蟹堡王的秘方汉堡是比奇堡最受欢迎的食物。"
    b = "Gradient descent converges when the learning rate is small enough."
    assert estimate_jaccard(hasher.signature(a), hasher.signature(b)) < 0.05


def test_estimate_jaccard_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        estimate_jaccard(np.zeros(4, dtype=np.uint64), np.zeros(8, dtype=np.uint64))


def test_signatures_matrix_shape():
    hasher = MinHasher(num_perm=32)
    assert hasher.signatures(["a", "b", "c"]).shape == (3, 32)
    assert hasher.signatures([]).shape == (0, 32)


def test_long_document_chunking_matches_unchunked_result():
    """Signatures are computed in chunks to bound memory; the result must not depend
    on where the chunk boundaries fall."""
    hasher = MinHasher(num_perm=32, ngram=3, seed=1)
    hashes = np.random.default_rng(0).integers(
        0, int(MERSENNE_PRIME), size=9000, dtype=np.uint64
    )
    assert hashes.size > 4096  # actually exercises more than one chunk

    chunked = hasher.signature_from_hashes(hashes)
    reference = (
        (hasher.a[:, None] * hashes[None, :] + hasher.b[:, None]) % MERSENNE_PRIME
    ).min(axis=1)

    assert np.array_equal(chunked, reference)


def test_lsh_threshold_and_band_choice_are_consistent():
    assert lsh_threshold(1, 128) == pytest.approx(1.0)
    bands, rows = choose_bands(128, 0.8)
    assert bands * rows == 128
    assert lsh_threshold(bands, rows) == pytest.approx(0.8, abs=0.1)


def test_choose_bands_rejects_impossible_threshold():
    with pytest.raises(ValueError):
        choose_bands(128, 1.5)


def test_lsh_candidate_pairs_requires_consistent_band_shape():
    with pytest.raises(ValueError):
        lsh_candidate_pairs(np.zeros((2, 10), dtype=np.uint64), bands=3, rows=4)


def test_lsh_finds_identical_documents_as_candidates():
    signatures = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [9, 9, 9, 9]], dtype=np.uint64)
    assert lsh_candidate_pairs(signatures, bands=2, rows=2) == {(0, 1)}


def test_union_find_merges_transitively_and_roots_at_lowest_index():
    union = UnionFind(5)
    union.union(3, 1)
    union.union(1, 0)

    groups = union.groups()

    assert sorted(groups[0]) == [0, 1, 3]
    assert groups[2] == [2] and groups[4] == [4]


def test_cluster_near_duplicates_groups_only_similar_documents():
    base = "海绵宝宝住在比奇堡的一个菠萝里面，他每天都去蟹堡王上班，最喜欢的事情是抓水母。"
    texts = [
        base,
        base,                                    # exact repeat
        base.replace("抓水母", "吹泡泡"),          # small edit -> near duplicate
        "深度学习模型的训练需要大量高质量的语料数据，数据质量往往比模型结构更重要。",  # unrelated
    ]
    signatures = MinHasher(num_perm=256, ngram=5, seed=0).signatures(texts)

    clusters = cluster_near_duplicates(signatures, threshold=0.7)

    assert clusters == [[0, 1, 2]]


def test_cluster_respects_the_threshold():
    base = "海绵宝宝住在比奇堡的一个菠萝里面，他每天都去蟹堡王上班，最喜欢的事情是抓水母。"
    texts = [base, base.replace("抓水母", "吹泡泡")]
    signatures = MinHasher(num_perm=256, ngram=5, seed=0).signatures(texts)

    assert cluster_near_duplicates(signatures, threshold=0.7) == [[0, 1]]
    assert cluster_near_duplicates(signatures, threshold=0.99) == []


def test_cluster_handles_degenerate_inputs():
    hasher = MinHasher(num_perm=32)
    assert cluster_near_duplicates(hasher.signatures([]), 0.8) == []
    assert cluster_near_duplicates(hasher.signatures(["only one"]), 0.8) == []
