#!/usr/bin/env python3
"""Cohere reranking for the fourth row, with a cache the trial quota depends on.

WHAT THIS IS NOT
----------------
The guide over-retrieves ten times k and reranks that many candidates, so it
makes three calls per query: 50 documents for k=5, 100 for k=10, 200 for k=20.
Across 248 queries and three embedders that is 2,232 calls.

Cohere's trial key allows 1,000 API calls a month. This module makes ONE call
per query with a fixed 200-candidate pool and reads Pass@5, Pass@10 and Pass@20
out of that single ranking: 744 calls for all three embedders, inside the free
allowance. That is a deliberate deviation and it is not free of consequence.
Reranking 200 candidates and keeping the top 5 is not the same as reranking 50
and keeping all of them -- the larger pool gives the model more chances to find
the golden chunk and more distractors to reject at once. It most likely helps
Pass@5 rather than hurting it, which is the direction that flatters the result,
which is why it is written here and in RESULTS.md rather than left implicit.

WHAT IS MATCHED
---------------
  * rerank-english-v3.0        the guide's model, still served.
  * the document format        f"{original_content}\n\nContext: {context}".
                               Original first, generated context second behind a
                               "Context: " label -- read out of the guide rather
                               than reconstructed.
  * the candidate source       the contextual store's own dense ranking.

THE QUOTA IS THE REAL CONSTRAINT
--------------------------------
At 744 of 1,000 calls a month, one failed run that gets retried from scratch
would spend the rest. So every ranking is written to disk the moment it arrives,
keyed by the query and by the exact candidate list it ranked, and a rerun reads
the cache instead of the API. A run also refuses to start if it would need more
new calls than the budget it was given.

The key is read from COHERE_API_KEY and never printed, logged or cached.
"""
import hashlib
import json
import os
import sys
import time

MODEL = "rerank-english-v3.0"
CANDIDATES = 200
TRIAL_CALLS_PER_MONTH = 1000
TRIAL_RPM = 10
SECONDS_PER_CALL = 60 / TRIAL_RPM + 0.5     # a little margin under 10 per minute
MAX_TRIES = 5


def document_text(meta, context):
    """The guide's document format, byte for byte."""
    return f"{meta['original_content']}\n\nContext: {context}"


def candidate_key(query, chunk_ids):
    """Identifies a ranking by its query AND the exact list it ranked.

    Keying on the query alone would reuse one embedder's ranking for another,
    silently, and the three columns would stop being three measurements.
    """
    h = hashlib.sha256()
    h.update(query.encode("utf-8"))
    for doc_id, index in chunk_ids:
        h.update(f"\x00{doc_id}\x01{index}".encode("utf-8"))
    return h.hexdigest()


class Reranker:
    """Ranks a candidate list, remembers every answer, and counts what it spends."""

    def __init__(self, cache_path, budget, client=None):
        self.cache_path = cache_path
        self.budget = budget
        self.calls = 0
        self.hits = 0
        self.retries = 0
        self.cache = {}
        if os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                self.cache = json.load(f)
        self._client = client
        self._last = 0.0

    def client(self):
        """Built on first miss. A fully cached run needs no key at all."""
        if self._client is None:
            try:
                import cohere
            except ImportError:
                sys.exit("cohere is not installed:\n  python -m pip install cohere")
            key = os.getenv("COHERE_API_KEY")
            if not key:
                sys.exit(
                    "COHERE_API_KEY is not set.\n"
                    "  A free trial key from dashboard.cohere.com allows 1,000 calls\n"
                    "  a month, which is what this stage is sized for.\n"
                    '    $env:COHERE_API_KEY = "<your key>"')
            self._client = cohere.Client(key)
        return self._client

    def save(self):
        tmp = self.cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.cache, f)
        os.replace(tmp, self.cache_path)

    def missing(self, work):
        """How many of these (query, candidates) pairs are not cached yet."""
        return sum(1 for key, _, _ in work if key not in self.cache)

    def rank(self, key, query, documents):
        """Indices of `documents`, best first. Cached answers cost nothing."""
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        if self.calls >= self.budget:
            sys.exit(f"budget of {self.budget} new calls reached. Everything ranked so "
                     f"far is cached, so re-running continues where this stopped.")

        for attempt in range(MAX_TRIES):
            wait = self._last + SECONDS_PER_CALL - time.time()
            if wait > 0:
                time.sleep(wait)
            try:
                self._last = time.time()
                r = self.client().rerank(model=MODEL, query=query,
                                         documents=documents, top_n=len(documents))
                break
            except Exception as exc:
                text = str(exc).lower()
                retryable = any(s in text for s in
                                ("429", "rate limit", "timeout", "502", "503", "504"))
                if not retryable or attempt == MAX_TRIES - 1:
                    raise
                self.retries += 1
                back = SECONDS_PER_CALL * (2 ** attempt)
                print(f"\n  {type(exc).__name__}, waiting {back:.0f}s and retrying")
                time.sleep(back)

        order = [item.index for item in r.results]
        self.cache[key] = order
        self.calls += 1
        # Written on every call, not at the end. A crash after 200 calls must not
        # cost 200 calls of a 1,000-call monthly allowance.
        self.save()
        return order

    def report(self):
        return (f"  {self.calls} new Cohere calls, {self.hits} served from cache"
                + (f", {self.retries} retries" if self.retries else ""))
