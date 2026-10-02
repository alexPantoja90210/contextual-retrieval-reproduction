#!/usr/bin/env python3
"""Checks the reranking path without Cohere, a key, or a network call.

The quota is what makes these worth writing. A cache keyed too loosely reuses
one embedder's ranking for another and the three columns stop being three
measurements -- silently, with plausible numbers. A cache written only at the
end turns one crash into a spent month. Neither raises.

  python test_rerank.py
"""
import json
import os
import sys
import tempfile

import rerank
from rerank import Reranker, candidate_key, document_text

CASES = []


def case(fn):
    CASES.append(fn)
    return fn


class FakeResult:
    def __init__(self, index):
        self.index = index


class FakeResponse:
    def __init__(self, order):
        self.results = [FakeResult(i) for i in order]


class FakeCohere:
    """Answers rerank. Records every call so a test can count them."""

    def __init__(self, fail_times=0, error="429 rate limit exceeded"):
        self.calls = []
        self.fail_times = fail_times
        self.error = error

    def rerank(self, model, query, documents, top_n):
        self.calls.append({"model": model, "query": query,
                           "documents": list(documents), "top_n": top_n})
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError(self.error)
        # reversed, so a test can tell the model's order from the input order
        return FakeResponse(list(range(len(documents)))[::-1])


def make(tmp, budget=10, client=None, name="cache.json"):
    r = Reranker(os.path.join(tmp, name), budget, client or FakeCohere())
    rerank.SECONDS_PER_CALL = 0      # no real waiting inside the suite
    return r


@case
def the_document_format_matches_the_guide():
    """original, blank line, 'Context: ', generated context. Read, not invented."""
    meta = {"original_content": "def f():\n    pass"}
    got = document_text(meta, "This function does nothing.")
    assert got == "def f():\n    pass\n\nContext: This function does nothing.", repr(got)
    return "f(original)\\n\\nContext: {context}"


@case
def the_cache_key_separates_different_candidate_lists():
    """Keying on the query alone would serve one embedder's ranking to another."""
    a = candidate_key("how do I log in", [("d1", 0), ("d2", 1)])
    b = candidate_key("how do I log in", [("d2", 1), ("d1", 0)])
    c = candidate_key("how do I log in", [("d1", 0), ("d2", 2)])
    assert a != b, "same chunks in a different order shared a key"
    assert a != c, "different chunks shared a key"
    return "order and membership both change the key"


@case
def the_cache_key_is_stable_for_the_same_inputs():
    ids = [("d1", 0), ("d2", 1)]
    assert candidate_key("q", ids) == candidate_key("q", list(ids))
    return "same query and same candidates, same key"


@case
def a_cached_ranking_costs_no_call():
    with tempfile.TemporaryDirectory() as tmp:
        c = FakeCohere()
        r = make(tmp, client=c)
        docs = ["a", "b", "c"]
        first = r.rank("k1", "q", docs)
        second = r.rank("k1", "q", docs)
        assert first == second, (first, second)
        assert len(c.calls) == 1, f"{len(c.calls)} calls for two ranks"
        assert r.calls == 1 and r.hits == 1, (r.calls, r.hits)
    return "two ranks, one call"


@case
def the_cache_survives_a_crash_after_each_call():
    """Written per call, not at the end. 744 of 1,000 calls allows no replay."""
    with tempfile.TemporaryDirectory() as tmp:
        r = make(tmp)
        r.rank("k1", "q", ["a", "b"])
        on_disk = json.load(open(r.cache_path, encoding="utf-8"))
        assert "k1" in on_disk, on_disk
        fresh = Reranker(r.cache_path, budget=10, client=FakeCohere())
        assert fresh.rank("k1", "q", ["a", "b"]) == r.cache["k1"]
        assert fresh.calls == 0, "a new process paid for a cached ranking"
    return "a second process reuses the first's answers"


@case
def the_budget_stops_the_run_before_it_overspends():
    with tempfile.TemporaryDirectory() as tmp:
        r = make(tmp, budget=2)
        r.rank("k1", "q1", ["a"])
        r.rank("k2", "q2", ["a"])
        try:
            r.rank("k3", "q3", ["a"])
        except SystemExit as exc:
            assert "budget" in str(exc).lower(), exc
            return "third call refused at a budget of 2"
    raise AssertionError("the budget was exceeded without stopping")


@case
def missing_counts_only_what_is_not_cached():
    with tempfile.TemporaryDirectory() as tmp:
        r = make(tmp)
        work = [("k1", "q1", ["a"]), ("k2", "q2", ["b"]), ("k3", "q3", ["c"])]
        assert r.missing(work) == 3, r.missing(work)
        r.rank(*work[0])
        assert r.missing(work) == 2, r.missing(work)
    return "the pre-flight count tracks the cache"


@case
def the_returned_order_is_the_models_order():
    with tempfile.TemporaryDirectory() as tmp:
        r = make(tmp)
        order = r.rank("k", "q", ["a", "b", "c", "d"])
        assert order == [3, 2, 1, 0], order
    return "indices come back as the model ranked them, not as sent"


@case
def a_fully_cached_run_never_builds_a_client():
    """So a rerun needs no key at all, and cannot spend quota by accident."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cache.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"k1": [2, 0, 1]}, f)
        r = Reranker(path, budget=0)          # no client, no key
        assert r.rank("k1", "q", ["a", "b", "c"]) == [2, 0, 1]
        assert r.calls == 0
    return "no client constructed, no key read"


@case
def a_rate_limit_error_is_retried():
    with tempfile.TemporaryDirectory() as tmp:
        c = FakeCohere(fail_times=2)
        r = make(tmp, client=c)
        assert r.rank("k", "q", ["a", "b"]) == [1, 0]
        assert len(c.calls) == 3, f"{len(c.calls)} attempts"
        assert r.retries == 2, r.retries
        assert r.calls == 1, "a retry counted as a second spent call"
    return "two 429s, one ranking, one call counted"


@case
def an_unrecognised_error_is_not_retried():
    """A bad key must fail at once, not burn five attempts."""
    with tempfile.TemporaryDirectory() as tmp:
        c = FakeCohere(fail_times=99, error="401 invalid api token")
        r = make(tmp, client=c)
        try:
            r.rank("k", "q", ["a"])
        except RuntimeError:
            assert len(c.calls) == 1, f"{len(c.calls)} attempts on a 401"
            return "401 raises on the first attempt"
    raise AssertionError("an authentication error was swallowed")


@case
def the_call_asks_for_every_candidate_back():
    """top_n must cover the pool, or Pass@20 reads a truncated ranking."""
    with tempfile.TemporaryDirectory() as tmp:
        c = FakeCohere()
        r = make(tmp, client=c)
        docs = [f"doc {i}" for i in range(200)]
        r.rank("k", "q", docs)
        assert c.calls[0]["top_n"] == 200, c.calls[0]["top_n"]
        assert c.calls[0]["model"] == rerank.MODEL, c.calls[0]["model"]
    return f"top_n=200 to {rerank.MODEL}"


def main():
    print(f"\n{'=' * 62}")
    print("  Reranking path and its cache, offline")
    print(f"{'=' * 62}")
    failed = 0
    for fn in CASES:
        name = fn.__name__.replace("_", " ")
        try:
            detail = fn()
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {name}\n          {type(exc).__name__}: {exc}")
        else:
            print(f"  ok    {name}")
            if detail:
                print(f"          {detail}")
    print(f"{'-' * 62}")
    print(f"  {failed} of {len(CASES)} checks failed" if failed
          else f"  {len(CASES)} checks passed")
    print(f"{'=' * 62}\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
