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

## Results

Both stages run here reproduced the published figures.

| Stage | | Pass@5 | Pass@10 | Pass@20 |
|---|---|---|---|---|
| Baseline | published | 80.92% | 87.15% | 90.06% |
| Baseline | **this run** | **80.92%** | **87.15%** | **90.06%** |
| Contextual | published | 88.12% | 92.34% | 94.29% |
| Contextual | **this run** | **87.45%** | **92.10%** | **94.99%** |
| | gain measured here | +6.53 | +4.95 | +4.93 |

The baseline matched to the hundredth — 80.91878, 87.14958, 90.06336. Embeddings
are deterministic, so an unchanged method on unchanged data has one answer, and
landing on it is what makes the rest of the table worth reading.

The contextual stage scatters about half a point in both directions, inside a
tolerance of two. That stage puts a language model in the pipeline and the line
it writes for a chunk is not guaranteed to be identical twice, so a match to the
hundredth here would be the suspicious result rather than the good one.

Raw figures are in `result_baseline.json` and `result_contextual.json`.

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
python -m pip install voyageai anthropic numpy
```

Through `python -m` rather than `pip` directly: it installs into the same
interpreter that will run the script, which matters when more than one Python
is on the machine, and it works when `pip` is not on PATH.

Stage 1 needs one key. Stage 2 needs two.

```bash
# Stage 1 — baseline.  VOYAGE_API_KEY only.
python reproduce.py baseline

# Stage 2 — contextual.  Adds ANTHROPIC_API_KEY.
python reproduce.py contextual
```

### On Voyage's free tier

Without a payment method on file, Voyage allows **3 requests and 10,000 tokens
per minute**. The corpus is about **199,000 tokens**, so stage 1 takes roughly
**25 minutes** and costs nothing. The script holds to both limits and prints
what it is waiting for rather than appearing to hang.

Three things make that work:

- Requests are sized by token count, not by a fixed number of chunks, and capped
  well below the per-minute budget rather than at it.
- All 246 unique queries are embedded in one batch up front. One at a time is
  248 requests, which on this tier is over an hour of waiting for a few thousand
  tokens of text.
- **A refused request is rebuilt smaller rather than retried unchanged.** Token
  counts are never exact, and retrying an identical over-budget request cannot
  succeed however long the wait. On a refusal the cap is halved and the batch is
  rebuilt from the same position, so the run does not depend on the count being
  right — only on the error being fixable. `test_batching.py` checks this
  against a stub server whose real limit is below what the script believes.

With a payment method the limits rise sharply and the free token allowance still
applies, so it still costs nothing — pass `--rpm=300 --tpm=1000000` to run it in
about a minute.

```bash
python test_batching.py     # no key, no network
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

## Does the gain survive a different embedding model?

Anthropic published this technique measured on one embedder, voyage-2. Whether
the gain belongs to the technique or to that particular model is a separate
question, and it is not one the guide answers.

The harness can answer it, because the baseline above reproduced the published
figures. An instrument that reads true on a known answer can be pointed at an
unknown one. So the embedding provider is swappable, and a second provider is
included: **Amazon Titan Text Embeddings V2**, through Bedrock.

```bash
python reproduce.py baseline   --embedder=titan
python reproduce.py contextual --embedder=titan
python consolidate.py
```

Everything except the embedder is held fixed — the same 737 chunks, the same
248 queries, the same golden chunks, the same dot-product-over-top-k scoring,
and the same 1024 dimensions. The contextual stage reuses the contextualized
text the voyage-2 run already wrote to disk rather than regenerating it, so
Claude is not called a second time and the text being embedded is byte-identical
across the two runs. Changing one thing is the only way the difference means
anything.

These runs are **not** a reproduction and the script does not grade them. There
is no published Titan figure for this benchmark, so there is nothing to match.
What they produce is a measurement placed next to another measurement:

| Embedder | @5 | @10 | @20 |
|---|---|---|---|
| Anthropic, published | +7.20 | +5.19 | +4.23 |
| voyage-2, this repo | +6.53 | +4.95 | +4.93 |
| titan-embed-v2 | — | — | — |

`consolidate.py` fills the last row and writes `RESULTS.md`.

Setup is in [BEDROCK-SETUP.md](BEDROCK-SETUP.md): a scoped IAM user limited to
one action on one model in one region, and the CLI profile. There is no model
activation step — Bedrock's Model access page has been retired and serverless
foundation models enable themselves on first invocation.
Titan Embed v2 is $0.02 per million input tokens, so both stages together cost
a few cents. Nothing is left running — there is no vector store, no index and
no infrastructure to delete.

```bash
python test_embedders.py    # no AWS, no credentials, no network
```

That test covers the failures that would not look like failures: a batch
reassembled in the wrong order still scores, and a silently dropped retry
produces a short run that reports Pass@k as if it were complete.

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

## What this changes from the guide, and what it does not

The measured path is the guide's: voyage-2 embeddings, dot-product similarity,
top-k, the same scoring function, the same data, the same prompts byte for byte
(including the eight-space indentation the guide's class definition gives them —
that whitespace reaches the model, so it is kept rather than tidied away).

What is different is everything around the measurement, none of which touches
the result:

- It is a script with stages rather than a notebook, and it caches each stage.
- Requests are sized by token count, throttled to the account's rate limits, and
  rebuilt smaller when refused, so it completes on a free tier.
- Queries are embedded in one batch instead of one request each.
- Each document's first chunk is sent alone so the document is written to the
  prompt cache once rather than raced.
- Spending is tracked against a budget and the run stops if it is exceeded.
- The result is printed next to the published figures with a verdict.

The exact match on stage 1 — 80.918.../87.149.../90.063... against a published
80.92/87.15/90.06 — is itself the evidence that the measured path was not
altered. A changed method would not land on the same hundredths.

## Where the published guide no longer runs as written

Two things in the cookbook have aged, and both are handled here:

- **`temperature` on `messages.create()`.** The guide passes it as a named
  argument. Anthropic's SDK 1.9 dropped it from that signature, so the guide as
  published raises `TypeError` on a current install. The API still accepts the
  field, so it is sent through `extra_body`; if the server refuses it too, the
  run continues at the default temperature and says so, because that is a small
  loss of determinism rather than a reason to stop.
- **The prompt-caching beta header.** Prompt caching is generally available, so
  the `anthropic-beta` header the guide sets is no longer needed.

Neither changes what is measured.

## License and attribution

This repository's own code — `reproduce.py`, `embedders.py`, `consolidate.py`,
`test_batching.py`, `test_embedders.py` — is MIT licensed. The data files and `guide.ipynb` come from
[anthropics/claude-cookbooks](https://github.com/anthropics/claude-cookbooks),
also MIT licensed, copyright Anthropic, and are included unchanged so the
benchmark runs from a clone. The published figures being matched are theirs.
See [LICENSE](LICENSE).

## Source

The data, the retrieval code and the published figures are Anthropic's:

- [Contextual Retrieval in AI Systems](https://www.anthropic.com/engineering/contextual-retrieval)
- [Cookbook guide](https://github.com/anthropics/claude-cookbooks/blob/main/capabilities/contextual-embeddings/guide.ipynb)

`guide.ipynb` is included for reference. `data/` is downloaded from that
repository unchanged.
