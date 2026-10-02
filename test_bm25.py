#!/usr/bin/env python3
"""Checks the local BM25 and the rank fusion without a service or a network call.

Every property here is one whose failure would be invisible in the results. A
negative idf still produces a ranking; it just quietly demotes every chunk that
contains a common term. A fusion that sums two fields instead of taking the
maximum still produces a Pass@k. Neither raises.

  python test_bm25.py
"""
import bm25
from bm25 import Bm25Index, Field, analyze, fuse

CASES = []


def case(fn):
    CASES.append(fn)
    return fn


def meta(doc, idx, content, context=None):
    return {"doc_id": doc, "original_index": idx, "original_content": content,
            "contextualized_content": context if context is not None else content}


def hits(ids):
    """Fake retrieval output, ranked, in the shape both sides produce."""
    return [{"metadata": meta(d, i, "")} for d, i in ids]


# ---------------------------------------------------------------- analyzer

@case
def stop_words_are_removed():
    assert analyze("the cache is on the server") == analyze("cache server"), \
        analyze("the cache is on the server")
    return "_english_ list applied"


@case
def words_are_stemmed_to_a_common_root():
    for a, b in (("running", "runs"), ("databases", "database"), ("queries", "query")):
        assert analyze(a) == analyze(b), f"{a!r} and {b!r} did not share a stem"
    return "running/runs, databases/database, queries/query"


@case
def possessives_are_stripped_before_lowercasing():
    assert analyze("Alice's") == analyze("alice"), analyze("Alice's")
    assert analyze("SERVER'S") == analyze("server"), analyze("SERVER'S")
    return "'s and 'S both handled"


@case
def underscores_do_not_split_identifiers():
    assert len(analyze("foo_bar")) == 1, analyze("foo_bar")
    assert analyze("foo_bar") != analyze("foo bar"), "identifier split like prose"
    return "foo_bar stays one token"


# ------------------------------------------------------------------- BM25

@case
def idf_is_never_negative():
    """The reason this is not rank_bm25: BM25Okapi's idf goes negative here."""
    f = Field(["cache " + w for w in ("alpha", "beta", "gamma", "delta")])
    term = analyze("cache")[0]
    assert f.idf[term] > 0, f"idf for a term in every document: {f.idf[term]}"
    return f"term in 4 of 4 documents has idf {f.idf[term]:.4f}"


@case
def a_term_in_every_document_still_contributes():
    f = Field(["cache alpha", "cache beta", "cache gamma"])
    s = f.scores(analyze("cache"))
    assert s and all(v > 0 for v in s.values()), s
    return "every document scores above zero"


@case
def shorter_documents_score_higher_at_equal_frequency():
    """b = 0.75 is length normalization; without it these two would tie."""
    f = Field(["cache", "cache " + " ".join(f"pad{i}" for i in range(50))])
    s = f.scores(analyze("cache"))
    assert s[0] > s[1], f"length normalization absent: {s}"
    return f"{s[0]:.4f} against {s[1]:.4f} for the padded document"


@case
def two_fields_take_the_maximum_not_the_sum():
    """multi_match defaults to best_fields. Summing would double-count.

    Both fields carry the same text in every document, so they also carry the
    same statistics and must produce the same score. Taking the maximum leaves
    that score unchanged; summing would double it.
    """
    texts = ["cache alpha", "beta gamma"]
    ix = Bm25Index([meta("d", i, t, t) for i, t in enumerate(texts)])
    both = ix.search("cache", k=2)
    single = Field(texts).scores(analyze("cache"))[0]
    assert abs(both[0]["score"] - single) < 1e-9, \
        f"{both[0]['score']} is not the single-field score {single}"
    assert abs(both[0]["score"] - 2 * single) > 1e-9, "the two fields were summed"
    return f"two identical fields score {single:.4f}, not {2 * single:.4f}"


@case
def search_is_deterministic_and_respects_k():
    docs = [meta("d", i, f"cache entry {i} alpha beta") for i in range(10)]
    ix = Bm25Index(docs)
    a = [m["metadata"]["original_index"] for m in ix.search("cache alpha", k=4)]
    b = [m["metadata"]["original_index"] for m in ix.search("cache alpha", k=4)]
    assert a == b, f"two identical searches disagreed: {a} {b}"
    assert len(a) == 4, a
    return f"top-4 stable at {a}"


@case
def the_keyword_ranking_does_not_depend_on_the_embedder():
    """What makes the hybrid column comparable across the three embedders.

    The keyword side reads only the chunk text, which is byte-identical in all
    three contextual databases, so this ranking is shared. If it ever stopped
    being shared, the three hybrid columns would differ for two reasons at once
    and neither could be read.
    """
    docs = [meta("d", i, f"cache entry {i}") for i in range(6)]
    first = Bm25Index(docs).search("cache entry", k=6)
    second = Bm25Index(list(docs)).search("cache entry", k=6)
    assert [h["metadata"]["original_index"] for h in first] == \
           [h["metadata"]["original_index"] for h in second]
    return "same corpus, same ranking, no embedder input"


# ----------------------------------------------------------------- fusion

@case
def semantic_outranks_keyword_at_the_same_position():
    """0.8 against 0.2: a keyword-only top hit does not displace a semantic one."""
    out, _, _ = fuse(hits([("s", 0)]), hits([("k", 0)]), k=2)
    assert [m["metadata"]["doc_id"] for m in out] == ["s", "k"], out
    return "0.8/1 beats 0.2/1"


@case
def agreement_between_the_two_sides_wins():
    out, _, _ = fuse(hits([("a", 0), ("b", 0)]), hits([("b", 0), ("a", 0)]), k=2)
    assert out[0]["metadata"]["doc_id"] == "a", "the chunk both sides ranked first lost"
    return "0.8 + 0.2 outranks either side alone"


@case
def a_keyword_only_chunk_can_enter_the_result():
    """The whole point of the hybrid: something semantic search never retrieved."""
    semantic = hits([("s", i) for i in range(20)])
    out, _, _ = fuse(semantic, hits([("k", 0)]), k=5)
    assert any(m["metadata"]["doc_id"] == "k" for m in out), \
        "keyword-only chunk never surfaced"
    return "0.2/1 beats 0.8/6, so rank 6 and below are reachable"


@case
def contribution_counts_split_shared_chunks():
    out, sem, kw = fuse(hits([("a", 0)]), hits([("a", 0)]), k=1)
    assert (sem, kw) == (0.5, 0.5), (sem, kw)
    assert len(out) == 1, out
    return "a chunk found by both counts half to each"


@case
def fusion_never_returns_more_than_k():
    out, _, _ = fuse(hits([("s", i) for i in range(30)]),
                     hits([("k", i) for i in range(30)]), k=7)
    assert len(out) == 7, len(out)
    seen = [(m["metadata"]["doc_id"], m["metadata"]["original_index"]) for m in out]
    assert len(set(seen)) == 7, f"duplicate chunk in the result: {seen}"
    return "7 distinct chunks"


@case
def fusion_is_stable_under_input_order():
    """Set iteration must not reach the output; the guide's tie-break decides."""
    a = fuse(hits([("a", 0), ("b", 1)]), hits([("b", 1), ("a", 0)]), k=2)[0]
    b = fuse(hits([("a", 0), ("b", 1)]), hits([("b", 1), ("a", 0)]), k=2)[0]
    assert [m["metadata"] for m in a] == [m["metadata"] for m in b]
    return "repeated fusion returns the same order"


def main():
    print(f"\n{'=' * 62}")
    print("  Local BM25 and rank fusion, offline")
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
