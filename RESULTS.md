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

## What contextual retrieval adds, per embedder

| Embedder | @5 | @10 | @20 |
| --- | --- | --- | --- |
| Anthropic, published | +7.20 | +5.19 | +4.23 |
| voyage-2 | +6.53 | +4.95 | +4.93 |

## Not yet run

- titan-embed-v2, baseline — `python reproduce.py baseline --embedder=titan`
- titan-embed-v2, contextual — `python reproduce.py contextual --embedder=titan`

