#!/usr/bin/env python3
"""Checks the embedding path against a server that refuses oversized requests.

The first version of this script retried an identical over-budget request with
a longer wait each time, which cannot succeed: the request is the problem, not
the timing. This test fails that design and passes the one that splits.

No API key and no network: the client is a stub.

    python test_batching.py
"""
import importlib.util
import json
import os
import sys

os.environ["REPRODUCE_QUIET"] = "1"

HERE = os.path.dirname(os.path.abspath(__file__))


def load_module():
    os.environ.setdefault("VOYAGE_API_KEY", "test")
    spec = importlib.util.spec_from_file_location("rep", os.path.join(HERE, "reproduce.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["rep"] = m
    spec.loader.exec_module(m)
    return m


class RefusingEmbedder:
    """An embedder whose real limit is below what the caller believes.

    It implements the same interface as the providers in embedders.py, so the
    batching loop cannot tell it apart from a real one.
    """

    key, label, suffix = "stub", "stub", "_stub"
    max_texts = None
    default_rpm, default_tpm = 10_000, 10_000_000

    def __init__(self, limit_tokens):
        self.limit = limit_tokens
        self.calls, self.refusals, self.max_accepted = 0, 0, 0

    def count(self, text):
        # Deliberately an estimate rather than a real tokenizer. What is under
        # test is that the run converges however wrong the count is, so the
        # count being wrong is the condition, not a flaw in the test.
        return max(1, int(len(text) / 2.5))

    def embed(self, texts, kind="passage"):
        self.calls += 1
        tokens = sum(self.count(t) for t in texts)
        if tokens > self.limit:
            self.refusals += 1
            raise RuntimeError(
                "rate limit exceeded: you have not yet added your payment method")
        self.max_accepted = max(self.max_accepted, tokens)
        return [[0.0] * 4 for _ in texts]

    def report(self):
        return None


def main():
    m = load_module()

    texts = [c["content"]
             for doc in json.load(open(os.path.join(HERE, "data", "codebase_chunks.json"),
                                       encoding="utf-8"))
             for c in doc["chunks"]]
    print(f"{len(texts)} chunks\n")

    failures = 0
    # The server's true limit is deliberately set BELOW what the script thinks a
    # request may hold, which is exactly the situation that broke the first
    # version: the token count was wrong in the unsafe direction.
    # 1,500 is below the largest single chunk, so it cannot be satisfied by
    # splitting. That case must fail with a clear message, not spin forever.
    impossible = 1_500
    for true_limit in (10_000, 6_000, 3_000, impossible):
        emb = RefusingEmbedder(true_limit)
        db = m.VectorDB(emb, rpm=10_000, tpm=10_000_000)  # timing out of the way
        db.request_cap = 5_500                            # what it believes

        try:
            out = db.embed_all(texts, "chunks")
            ok = len(out) == len(texts)
        except SystemExit as exc:
            if true_limit == impossible:
                print(f"  server limit {true_limit:>6}: refused cleanly, as it should   ok")
            else:
                print(f"  server limit {true_limit:>6}: GAVE UP: {str(exc)[:70]}")
                failures += 1
            continue
        except Exception as exc:
            print(f"  server limit {true_limit:>6}: RAISED {type(exc).__name__}: {exc}")
            failures += 1
            continue
        if true_limit == impossible:
            print(f"  server limit {true_limit:>6}: completed, which should not happen")
            failures += 1
            continue

        c = emb
        status = "ok" if ok else "WRONG COUNT"
        print(f"  server limit {true_limit:>6}: {len(out)} vectors  "
              f"{c.calls} calls, {c.refusals} refused, "
              f"largest accepted {c.max_accepted} tokens   {status}")
        if not ok or c.max_accepted > true_limit:
            failures += 1

    print()
    if failures:
        print(f"{failures} case(s) failed. The embedding path does not converge.")
        return 1
    print("Every case completed. A refused request is split until it fits,")
    print("so the run does not depend on the token count being right.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
