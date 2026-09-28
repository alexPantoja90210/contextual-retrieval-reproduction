# Reproducing Anthropic's Contextual Retrieval benchmark

Anthropic published a retrieval benchmark with the data included and the numbers
stated. This repository runs it and prints your results next to theirs, with a
verdict: reproduced, or not.

That is the whole point. A benchmark you can only read is a claim. A benchmark
you can run is evidence.

## What is being measured

737 chunks from 9 codebases, and 248 queries. Each query has a known correct
chunk. The metric is **Pass@k**: for a given query, is the correct chunk among
the top k retrieved?

| | Pass@5 | Pass@10 | Pass@20 |
|---|---|---|---|
| Baseline RAG | 80.92% | 87.15% | 90.06% |
| + Contextual Embeddings | 88.12% | 92.34% | 94.29% |
| + Hybrid BM25 | 86.43% | 93.21% | 94.99% |
| + Reranking | 92.15% | 95.26% | 97.45% |

The retrieval and scoring code here is the guide's own, unchanged: voyage-2
embeddings, dot-product similarity, top-k, and the same scoring function. What
this repository adds is the comparison and the verdict.

## What "contextual embeddings" does

A chunk pulled out of a file loses what it was about. `def handle(req)` means
one thing in an auth module and another in a payment gateway, and the embedding
cannot tell.

So before embedding, Claude reads the whole document and writes one line
situating the chunk in it. That line is prepended to the chunk. The document is
sent with every request and cached, so it is billed once per document instead of
once per chunk — the script prints what share of document tokens came from the
cache rather than being re-billed.

That is the entire idea. Its effect is the gap between the first two rows.

## Running it

```bash
pip install voyageai anthropic numpy
```

Stage 1 needs one key. Stage 2 needs two.

```bash
# Stage 1 — baseline.  ~$0.05, a few minutes.  VOYAGE_API_KEY only.
python reproduce.py baseline

# Stage 2 — contextual.  ~$1-3, 15-30 minutes.  Adds ANTHROPIC_API_KEY.
python reproduce.py contextual
```

Keys come from [voyageai.com](https://www.voyageai.com/) and
[console.anthropic.com](https://console.anthropic.com/).

```powershell
$env:VOYAGE_API_KEY    = "..."
$env:ANTHROPIC_API_KEY = "..."
```

Each stage caches its vector database to `data/`, so a second run costs nothing
and re-scores in seconds. Each writes `result_<stage>.json`.

Stages 3 and 4 of the guide — hybrid BM25 and reranking — need Elasticsearch in
Docker and a Cohere key. They are not run here. The first two stages are where
the idea being tested lives; the last two are retrieval engineering on top of it.

## What the number means, and what it does not

Pass@20 asks one question with an unambiguous answer: **is the correct chunk in
the top twenty?** Yes or no.

It does not ask whether the generated answer was right, whether it was
faithful to the source, or whether an expert would agree with it. Those are
harder questions and they score lower — which is why a retrieval benchmark can
report 97% while end-to-end evaluations on hard corpora report figures in the
forties.

Both kinds of number are honest. They answer different questions. Quoting one
while implying the other is where RAG reporting usually goes wrong.

## Source

The data, the retrieval code and the published figures are Anthropic's:

- [Contextual Retrieval in AI Systems](https://www.anthropic.com/engineering/contextual-retrieval)
- [Cookbook guide](https://github.com/anthropics/claude-cookbooks/blob/main/capabilities/contextual-embeddings/guide.ipynb)

`guide.ipynb` is included for reference. `data/` is downloaded from that
repository unchanged.
