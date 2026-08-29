"""Embeddings are normalised on write regardless of what the provider returns.

This test asserts unit norm so that if a provider silently changes its
behaviour, a test fails loudly instead of the accuracy numbers drifting
mysteriously across arms.
"""
import math

from src.ingest.chunks import normalize


def test_normalize_produces_unit_norm():
    vector = [3.0, 4.0]  # 3-4-5 triangle, norm = 5
    result = normalize(vector)
    assert math.isclose(math.sqrt(sum(x * x for x in result)), 1.0, rel_tol=1e-9)


def test_normalize_arbitrary_vector():
    vector = [0.1, -2.5, 7.3, -0.001, 42.0, 0.0, -13.37]
    result = normalize(vector)
    assert math.isclose(math.sqrt(sum(x * x for x in result)), 1.0, rel_tol=1e-9)


def test_normalize_preserves_direction():
    vector = [1.0, 1.0]
    result = normalize(vector)
    assert math.isclose(result[0], result[1], rel_tol=1e-9)
