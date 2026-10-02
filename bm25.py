#!/usr/bin/env python3
"""Lucene-compatible BM25 over the chunk corpus, for the hybrid stage.

WHAT THIS IS NOT
----------------
The published guide runs the keyword side on Elasticsearch. This file does not
call Elasticsearch, so the hybrid column it feeds is NOT a reproduction of
Anthropic's BM25 row and is never labelled as one. It is the guide's fusion
recipe, over the guide's chunks, with the keyword engine rewritten locally so
the whole stage runs offline with no service and no credentials.

WHAT IS MATCHED, DELIBERATELY
-----------------------------
Running `rank_bm25` instead would have introduced three differences that have
nothing to do with the question being asked, so they are closed here:

  * k1 = 1.2, b = 0.75          Lucene's defaults. rank_bm25's BM25Okapi uses
                                k1 = 1.5.
  * idf = ln(1 + (N-df+0.5)/(df+0.5))
                                Lucene's form, which is positive for every term.
                                BM25Okapi's idf goes NEGATIVE for a term that
                                appears in more than half the corpus, which on a
                                737-chunk corpus of one language's source code is
                                not a corner case.
  * best_fields over two fields The guide's query is a `multi_match` with no
                                explicit type, so Elasticsearch scores `content`
                                and `contextualized_content` separately and keeps
                                the MAXIMUM, not the sum. Each field carries its
                                own document frequencies and its own average
                                length, so they are two indexes here as well.

WHAT IS NOT MATCHED, AND CANNOT BE
----------------------------------
  * Length-norm quantization.   Lucene stores each field's length in a single
                                byte, so document lengths are rounded to a coarse
                                scale before they reach the formula. This file
                                uses the exact length. That makes these scores
                                more precise than Elasticsearch's, not less, but
                                it does mean two chunks whose lengths Lucene would
                                have collapsed to the same bucket can rank in a
                                different order here.
  * The standard tokenizer.     Lucene implements the full Unicode UAX#29 word
                                break. `analyze` below approximates it. On ASCII
                                source code the two agree; on unusual punctuation
                                they may not.

Both are stated in RESULTS.md rather than left for a reader to discover.
"""
import math
import re
from collections import Counter

import snowballstemmer

K1 = 1.2
B = 0.75

# Elasticsearch's _english_ stop word list, verbatim.
STOPWORDS = frozenset("""
a an and are as at be but by for if in into is it no not of on or such
that the their then there these they this to was will with
""".split())

_TOKEN = re.compile(r"\w+(?:['’]\w+)*", re.UNICODE)
_POSSESSIVE = re.compile(r"['’][sS]$")
_STEMMER = snowballstemmer.stemmer("english")


def analyze(text):
    """Approximate Elasticsearch's `english` analyzer.

    Its filter chain is: standard tokenizer, possessive stemmer, lowercase,
    stop words, Porter2. The order matters -- the possessive stemmer runs
    before lowercasing, so it has to handle both 's and 'S.
    """
    out = []
    for raw in _TOKEN.findall(text):
        token = _POSSESSIVE.sub("", raw).lower()
        if not token or token in STOPWORDS:
            continue
        out.append(_STEMMER.stemWord(token))
    return out


class Field:
    """One BM25-scored field, with its own statistics.

    Separate instances because Elasticsearch keeps document frequencies and
    average length per field, and best_fields compares their scores as they
    stand.
    """

    def __init__(self, texts):
        self.tfs = [Counter(analyze(t)) for t in texts]
        self.lengths = [sum(tf.values()) for tf in self.tfs]
        n = len(self.tfs)
        self.avgdl = (sum(self.lengths) / n) if n else 0.0
        df = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.postings = {}
        for i, tf in enumerate(self.tfs):
            for term in tf:
                self.postings.setdefault(term, []).append(i)

    def scores(self, terms):
        """BM25 score per document for these already-analyzed query terms."""
        out = {}
        for term in terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i in self.postings[term]:
                tf = self.tfs[i][term]
                norm = K1 * (1 - B + B * (self.lengths[i] / self.avgdl)) if self.avgdl else K1
                out[i] = out.get(i, 0.0) + idf * (tf * (K1 + 1)) / (tf + norm)
        return out


class Bm25Index:
    """The guide's multi_match: two fields, best_fields, max of the two."""

    FIELDS = ("original_content", "contextualized_content")

    def __init__(self, metadata):
        self.metadata = metadata
        self.fields = [Field([m.get(f, "") for m in metadata]) for f in self.FIELDS]

    def search(self, query, k):
        terms = analyze(query)
        best = {}
        for field in self.fields:
            for i, s in field.scores(terms).items():
                if s > best.get(i, 0.0):
                    best[i] = s
        # Elasticsearch breaks score ties by internal document order, which for a
        # single bulk load is insertion order. Index order reproduces that.
        ranked = sorted(best, key=lambda i: (-best[i], i))[:k]
        return [{"metadata": self.metadata[i], "score": best[i]} for i in ranked]


def chunk_key(meta):
    return (meta["doc_id"], meta["original_index"])


def fuse(semantic, keyword, k, semantic_weight=0.8, bm25_weight=0.2):
    """The guide's rank fusion, including its tie-break.

    Each side contributes weight / (rank + 1). The guide then re-sorts by
    (score, doc_id, original_index) descending, so the tie-break runs on the
    document identifier rather than on anything about the match. Reproduced
    rather than improved: changing it would change the numbers for a reason
    that is not the one under study.
    """
    sem_ids = [chunk_key(r["metadata"]) for r in semantic]
    kw_ids = [chunk_key(r["metadata"]) for r in keyword]
    sem_rank = {}
    kw_rank = {}
    for i, cid in enumerate(sem_ids):
        sem_rank.setdefault(cid, i)
    for i, cid in enumerate(kw_ids):
        kw_rank.setdefault(cid, i)

    score = {}
    for cid in set(sem_ids) | set(kw_ids):
        s = 0.0
        if cid in sem_rank:
            s += semantic_weight * (1 / (sem_rank[cid] + 1))
        if cid in kw_rank:
            s += bm25_weight * (1 / (kw_rank[cid] + 1))
        score[cid] = s

    order = sorted(score, key=lambda c: (score[c], c[0], c[1]), reverse=True)[:k]

    by_id = {}
    for r in list(semantic) + list(keyword):
        by_id.setdefault(chunk_key(r["metadata"]), r["metadata"])

    from_semantic = from_keyword = 0.0
    out = []
    for cid in order:
        in_s, in_k = cid in sem_rank, cid in kw_rank
        out.append({"metadata": by_id[cid]})
        if in_s and in_k:
            from_semantic += 0.5
            from_keyword += 0.5
        elif in_s:
            from_semantic += 1
        else:
            from_keyword += 1
    return out, from_semantic, from_keyword
