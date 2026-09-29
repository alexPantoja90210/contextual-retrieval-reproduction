#!/usr/bin/env python3
"""Checks the Bedrock embedding path without AWS, a key, or a network call.

The properties tested here are the ones whose failure would be invisible in
the results. A batch reassembled in the wrong order still scores; it just
scores the wrong chunk against the query, and every number downstream looks
plausible. A silently dropped retry produces a short run that reports a
Pass@k as if it were complete. Neither shows up as an error.

The credentials check and the boto3 client are bypassed by constructing the
embedder directly, so this runs anywhere:

  python test_embedders.py
"""
import json
import sys
import threading

import embedders
from embedders import BedrockEmbedder, DIMENSIONS


class FakeError(Exception):
    """Shaped like botocore's ClientError: the code lives in .response."""

    def __init__(self, code, message=""):
        super().__init__(message or code)
        self.response = {"Error": {"Code": code}}


class FakeBody:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class FakeBedrock:
    """Answers InvokeModel. Each call is recorded so the test can inspect it."""

    def __init__(self, throttle_first=0, fail_with=None):
        self.calls = []
        self.throttle_first = throttle_first
        self.fail_with = fail_with
        self.seen = {}
        self.lock = threading.Lock()

    def invoke_model(self, modelId=None, body=None):
        req = json.loads(body)
        text = req["inputText"]
        with self.lock:
            self.calls.append(req)
            n = self.seen.get(text, 0)
            self.seen[text] = n + 1
        if self.fail_with:
            raise FakeError(self.fail_with)
        if n < self.throttle_first:
            raise FakeError("ThrottlingException")
        # The vector encodes the text, so a misordered result is detectable.
        vec = [float(len(text))] * DIMENSIONS
        return {"body": FakeBody({"embedding": vec,
                                  "inputTextTokenCount": max(1, len(text) // 4)})}


def make(client, threads=4):
    e = BedrockEmbedder.__new__(BedrockEmbedder)
    e.region, e.client, e.threads = "us-east-1", client, threads
    e.lock = threading.Lock()
    e.real_tokens = e.est_tokens = e.retries = 0
    return e


CASES = []


def case(fn):
    CASES.append(fn)
    return fn


@case
def order_is_preserved_across_threads():
    """Forty texts of distinct lengths, fanned out over four threads."""
    texts = ["x" * (i + 1) for i in range(40)]
    e = make(FakeBedrock())
    out = e.embed(texts)
    got = [int(v[0]) for v in out]
    assert got == [len(t) for t in texts], f"order lost: {got[:8]}"
    assert len(out) == len(texts), "count changed"
    return f"{len(texts)} texts returned in order"


@case
def every_vector_has_the_declared_width():
    e = make(FakeBedrock())
    out = e.embed(["a", "bb", "ccc"])
    assert all(len(v) == DIMENSIONS for v in out), "width varies"
    return f"all vectors are {DIMENSIONS} wide"


@case
def the_request_pins_dimensions_and_normalization():
    """Both matter. A default that changes upstream would change the result
    without changing this repository, and an unnormalized vector makes the
    dot-product scoring measure length as well as direction."""
    c = FakeBedrock()
    make(c).embed(["hello"])
    req = c.calls[0]
    assert req["dimensions"] == DIMENSIONS, req
    assert req["normalize"] is True, req
    return f"dimensions={req['dimensions']}, normalize={req['normalize']}"


@case
def throttling_is_retried_until_it_succeeds():
    c = FakeBedrock(throttle_first=2)
    e = make(c, threads=2)
    out = e.embed(["a", "bb", "ccc"])
    assert [int(v[0]) for v in out] == [1, 2, 3], "order lost under retry"
    assert e.retries == 6, f"expected 6 retries, counted {e.retries}"
    return f"{e.retries} retries, all three texts recovered"


@case
def persistent_throttling_raises_rather_than_returning_short():
    c = FakeBedrock(throttle_first=99)
    e = make(c, threads=1)
    try:
        e.embed(["a"])
    except FakeError:
        return f"gave up after {embedders.MAX_TRIES} tries and raised"
    raise AssertionError("returned instead of raising")


@case
def access_denied_points_at_the_causes_that_still_exist():
    """Bedrock's Model access page has been retired: serverless foundation
    models enable themselves on first invocation. A message that still sends
    the reader to that page costs them a trip to a console screen that no
    longer exists, so the text is asserted rather than trusted."""
    e = make(FakeBedrock(fail_with="AccessDeniedException"), threads=1)
    try:
        e.embed(["a"])
    except SystemExit as exc:
        msg = str(exc)
        assert "bedrock:InvokeModel" in msg, msg
        assert "retired" in msg, msg
        assert "us-east-1" in msg, msg
        assert "Model access page" in msg, msg
        return "names IAM, region and account policy; says the page is retired"
    raise AssertionError("did not exit")


@case
def token_counts_are_reported_against_the_estimate():
    c = FakeBedrock()
    e = make(c)
    texts = ["word " * 200] * 5
    for t in texts:
        e.count(t)
    e.embed(texts)
    line = e.report()
    assert "Titan reported" in line, line
    assert e.real_tokens > 0 and e.est_tokens > 0
    return line.strip()


@case
def the_batch_cap_is_honoured_by_the_caller():
    """embed_all sizes batches; the provider must never be handed more than
    it declares it can take."""
    import reproduce

    class Stub:
        key, label, suffix = "stub", "stub", "_stub"
        max_texts, default_rpm, default_tpm = 7, 10_000, 10_000_000

        def __init__(self):
            self.sizes = []

        def count(self, text):
            return 1

        def embed(self, texts, kind="passage"):
            self.sizes.append(len(texts))
            return [[0.0] * DIMENSIONS for _ in texts]

        def report(self):
            return None

    stub = Stub()
    db = reproduce.VectorDB(stub)
    out = db.embed_all([f"t{i}" for i in range(30)], "test")
    assert len(out) == 30, len(out)
    assert max(stub.sizes) <= 7, stub.sizes
    assert sum(stub.sizes) == 30, stub.sizes
    return f"30 texts in batches of {stub.sizes}, cap 7"


class _FixedTokenizer:
    """Returns a scripted token count per call, so length is under test."""

    def __init__(self, counts=None):
        self.counts, self.i = counts, 0

    def encode(self, text, add_special_tokens=None):
        if self.counts is None:
            return [0] * max(1, len(text) // 4)
        n = self.counts[self.i]
        self.i += 1
        return [0] * n


class FakeSentenceTransformer:
    """Stands in for the downloaded model: records how each text was encoded.

    encode() and tokenizer.encode() are different methods with the same name
    in sentence-transformers, so the tokenizer is a separate object here. An
    earlier version of this stub defined both on one class and the second
    silently replaced the first.
    """

    def __init__(self, dim=DIMENSIONS, limit=8192):
        self.dim, self.max_seq_length = dim, limit
        self.seen = []
        self.tokenizer = _FixedTokenizer()

    def encode(self, texts, normalize_embeddings=None, show_progress_bar=None,
               prompt_name=None):
        import types
        assert normalize_embeddings is True, "vectors must be normalized"
        self.seen.append((prompt_name, list(texts)))
        return [types.SimpleNamespace(tolist=lambda n=len(x): [float(n)] * self.dim)
                for x in texts]

    def get_sentence_embedding_dimension(self):
        return self.dim


def make_local(limit=8192, dim=DIMENSIONS):
    """A LocalEmbedder wired to the fake model, with no download."""
    from embedders import LocalEmbedder
    e = LocalEmbedder.__new__(LocalEmbedder)
    e.st = FakeSentenceTransformer(dim=dim, limit=limit)
    e.limit, e.truncated, e.longest = limit, 0, 0
    return e


@case
def the_query_prefix_goes_on_queries_and_not_on_passages():
    """Arctic asks for a prefix on queries only. Both mistakes — applying it
    everywhere, or nowhere — cost retrieval accuracy silently and look exactly
    like the model being worse than it is."""
    e = make_local()
    e.embed(["a chunk of code"], kind="passage")
    e.embed(["how do I parse this"], kind="query")
    prompts = [p for p, _ in e.st.seen]
    assert prompts == [None, "query"], prompts
    return "passage: no prompt; query: prompt_name='query'"


@case
def passage_is_the_default_so_a_missed_kind_cannot_prefix_a_chunk():
    e = make_local()
    e.embed(["a chunk"])
    assert e.st.seen[0][0] is None, e.st.seen
    return "default kind is passage"


@case
def truncation_is_counted_and_named_rather_than_silent():
    """A model whose window is shorter than the corpus biases the comparison,
    because contextual chunks are longer than baseline ones. The run has to
    say so; silence here would look like a clean result."""
    e = make_local(limit=100)
    e.st.tokenizer = _FixedTokenizer([150, 50, 300])
    for t in ("a", "b", "c"):
        e.count(t)
    line = e.report()
    assert e.truncated == 2, e.truncated
    assert "truncated" in line and "understates" in line, line
    return "2 of 3 over the limit, and the report says the bias direction"


def main():
    print(f"\n{'=' * 62}")
    print("  Embedding providers — offline checks")
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
    if failed:
        print(f"  {failed} of {len(CASES)} checks failed")
    else:
        print(f"  {len(CASES)} checks passed")
    print(f"{'=' * 62}\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
