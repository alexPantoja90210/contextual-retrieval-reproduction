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

## What contextual retrieval adds, per embedder

| Embedder | @5 | @10 | @20 |
| --- | --- | --- | --- |
| Anthropic, published | +7.20 | +5.19 | +4.23 |
| voyage-2 | +6.53 | +4.95 | +4.93 |
| arctic-embed-l-v2 | +9.33 | +8.36 | +5.84 |

## The same gains, adjusted for headroom

A model that starts lower has more room to gain, so the raw differences above are partly arithmetic. Each figure here is the share of the error still available that contextual retrieval removed: (contextual - baseline) / (100 - baseline).

| Embedder | @5 | @10 | @20 |
| --- | --- | --- | --- |
| Anthropic, published | 37.7% | 40.4% | 42.6% |
| voyage-2 | 34.2% | 38.6% | 49.6% |
| arctic-embed-l-v2 | 40.2% | 48.4% | 52.1% |

## Reading

**The gain survives the change of embedding model.** Contextual retrieval raises Pass@20 by +4.93 points on voyage-2 and +5.84 on arctic-embed-l-v2, against the same 248 queries and the same contextualized text. Only the embedder differs.

**It is worth more to the weaker model, and not only because that model had further to go.** arctic-embed-l-v2 starts below voyage-2 at every k, so some of its larger raw gain is headroom. After dividing that out it still closes more of the remaining error at every k (40.2% against 34.2% at k=5, 52.1% against 49.6% at k=20). The margin is wide where the numbers are low and narrow where both are already high.

**The two models converge.** The gap between them narrows from -4.14 / -4.41 / -1.28 at baseline to -1.34 / -1.01 / -0.37 once both run contextually. Writing a line of context in front of each chunk recovers most of what separated the two embedders.

### What this does not establish

One corpus of source code, one model writing the context, two embedders, 248 queries. The direction is consistent across three values of k, which is not the same as being general. Nothing here says the pattern holds on prose, on a larger corpus, or on a third embedder.

Anthropic published this technique on voyage-2 alone. What this table adds is a second embedding model, measured through a harness whose baseline reproduced their figures to the hundredth.

## Not yet run

- titan-embed-v2, baseline — `python reproduce.py baseline --embedder=titan`
- titan-embed-v2, contextual — `python reproduce.py contextual --embedder=titan`

