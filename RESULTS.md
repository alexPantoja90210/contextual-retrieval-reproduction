# Results

Pass@k on 737 chunks from 9 codebases, 248 queries, each with a
known golden chunk. Retrieval is dot-product over normalized
1024-dimension vectors, top-k. The scoring is the guide's own.

## Every run

| Embedder | Chunking | Pass@5 | Pass@10 | Pass@20 |
| --- | --- | --- | --- | --- |
| Anthropic, published | baseline | 80.92 | 87.15 | 90.06 |
| Anthropic, published | contextual | 88.12 | 92.34 | 94.29 |
| voyage-2 | baseline | 80.92 | 87.15 | 90.06 |
| voyage-2 | contextual | 87.45 | 92.10 | 94.99 |
| arctic-embed-l-v2 | baseline | 76.78 | 82.74 | 88.79 |
| arctic-embed-l-v2 | contextual | 86.10 | 91.10 | 94.62 |
| text-embedding-3-large | baseline | 83.89 | 88.49 | 91.85 |
| text-embedding-3-large | contextual | 86.66 | 91.11 | 94.76 |

## What contextual retrieval adds, per embedder

| Embedder | @5 | @10 | @20 |
| --- | --- | --- | --- |
| Anthropic, published | +7.20 | +5.19 | +4.23 |
| voyage-2 | +6.53 | +4.95 | +4.93 |
| arctic-embed-l-v2 | +9.33 | +8.36 | +5.84 |
| text-embedding-3-large | +2.78 | +2.61 | +2.90 |

## The same gains, adjusted for headroom

A model that starts lower has more room to gain, so the raw differences above are partly arithmetic. Each figure here is the share of the error still available that contextual retrieval removed: (contextual - baseline) / (100 - baseline).

| Embedder | @5 | @10 | @20 |
| --- | --- | --- | --- |
| Anthropic, published | 37.7% | 40.4% | 42.6% |
| voyage-2 | 34.2% | 38.6% | 49.6% |
| arctic-embed-l-v2 | 40.2% | 48.4% | 52.1% |
| text-embedding-3-large | 17.2% | 22.7% | 35.7% |

## Reading

**The gain survives every change of embedding model.** Contextual retrieval raises Pass@20 by +5.84 on arctic-embed-l-v2, +4.93 on voyage-2, +2.90 on text-embedding-3-large. Same 737 chunks, same contextualized text byte for byte, same 248 queries, same scoring, same 1024 dimensions. The embedding model is the only variable.

**The weaker the embedder, the more the context is worth — and headroom does not account for it.** Ranked by baseline Pass@5, arctic-embed-l-v2 then voyage-2 then text-embedding-3-large runs weakest to strongest. The share of remaining error that contextual retrieval closes falls in exactly that order at every k: 40.2% / 34.2% / 17.2% at k=5, 48.4% / 38.6% / 22.7% at k=10, 52.1% / 49.6% / 35.7% at k=20. The headroom correction was written into this script before any figure past voyage-2 existed, so it is not a post-hoc rescue; it divides out the arithmetic advantage of starting low and the ordering is still there.

**The embedders converge.** The spread across the 3 of them narrows from 7.11 / 5.76 / 3.07 points at baseline to 1.34 / 1.01 / 0.37 once all of them run contextually.

text-embedding-3-large leads the baseline at k=5 and does not lead contextually; voyage-2 does. The ordering of the embedders is not preserved through contextualization, so a baseline comparison between embedders does not predict the contextual one.

That is the same finding as the one above, seen from the other side. Writing a line of context in front of each chunk recovers most of what separated these embedders, which means the choice of embedder buys less after contextualizing than the baseline spread suggests it would.

### What this does not establish

One corpus of source code, one model writing the context, 3 embedders, 248 queries. The direction holds at three values of k and across 3 embedders, which is not the same as being general. Nothing here says the pattern holds on prose, on a larger corpus, with a different model writing the context, or on an embedder whose baseline starts above text-embedding-3-large.

The 3 baselines span 76.78 to 83.89 at k=5. A claim about embedders stronger than that range is a claim about data nobody here has.

Anthropic published this technique on voyage-2 alone. What this table adds is 2 further embedding models, measured through a harness whose voyage-2 baseline reproduced their published figures to the hundredth.

## Not yet run

- titan-embed-v2, baseline — `python reproduce.py baseline --embedder=titan`
- titan-embed-v2, contextual — `python reproduce.py contextual --embedder=titan`

