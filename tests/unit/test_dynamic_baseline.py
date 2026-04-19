"""
test_dynamic_baseline.py – Replaced by test_kmeans_classifier.py

The DynamicBaseline class (Z-score + EMA) has been removed from classification.py
in favour of the 5-step K-Means pipeline. All tests that covered DynamicBaseline
internals have been moved to:

    tests/unit/test_kmeans_classifier.py

This file is kept as a placeholder to avoid stale import errors in any CI
that might still reference it.
"""
# No tests here – see test_kmeans_classifier.py
