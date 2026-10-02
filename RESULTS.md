# Results

Pass@k on 737 chunks from 9 codebases, 248 queries, each with a
known golden chunk. Retrieval is dot-product over normalized
1024-dimension vectors, top-k. The scoring is the guide's own.

## Every run

| Embedder | Chunking | Pass@5 | Pass@10 | Pass@20 |
| --- | --- | --- | --- | --- |
| Anthropic, published | baseline | 80.92 | 87.15 | 90.06 |
| Anthropic, published | contextual | 88.12 | 92.34 | 94.29 |
| Anthropic, published | hybrid | 86.43 | 93.21 | 94.99 |
| voyage-2 | baseline | 80.92 | 87.15 | 90.06 |
| voyage-2 | contextual | 87.45 | 92.10 | 94.99 |
| voyage-2 | hybrid | 88.19 | 92.11 | 95.43 |
| arctic-embed-l-v2 | baseline | 76.78 | 82.74 | 88.79 |
| arctic-embed-l-v2 | contextual | 86.10 | 91.10 | 94.62 |
| arctic-embed-l-v2 | hybrid | 88.22 | 91.91 | 95.83 |
| text-embedding-3-large | baseline | 83.89 | 88.49 | 91.85 |
| text-embedding-3-large | contextual | 86.66 | 91.11 | 94.76 |
| text-embedding-3-large | hybrid | 87.77 | 91.45 | 95.87 |

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

## A known deviation from the guide, measured

The guide embeds the generated context BEFORE the chunk. This harness has always put it after. Same words, different order, which an embedding model reads and a keyword index does not. Rather than leave that for a reader to find, probe_order.py re-embeds the same 737 chunks with the same context text in the guide's order, reusing the cached query vectors, and scores it with the same evaluate(). Only the word order differs.

| arctic-embed-l-v2, contextual | @5 | @10 | @20 |
| --- | --- | --- | --- |
| chunk first (this harness) | 86.10 | 91.10 | 94.62 |
| context first (the guide) | 86.91 | 91.70 | 94.52 |
| difference | +0.81 | +0.60 | -0.10 |

The guide's order is worth +0.81 at k=5 and +0.60 at k=10, and -0.10 at k=20 -- it costs a little at the widest k, where both orders are already above 94%. The sign is not the same at every k, so this is a small effect with structure, not a uniform improvement.

That magnitude matters for one specific claim. This harness's voyage-2 contextual run came in 0.67 below the published figure at k=5 while its baseline reproduced to the hundredth, and +0.81 on a different embedder is the same order of magnitude as that gap. Plausible explanation, not a demonstrated one: the probe ran on arctic-embed-l-v2, and only a voyage-2 run could settle it.

The order is left as written. Every column in the tables above shares it, so the comparisons between embedders are unaffected: all three read the same text, byte for byte. What it affects is the comparison against Anthropic's published contextual row, which is why it is stated here instead of being corrected quietly after the numbers were already published.

## What the keyword fusion adds, on top of contextual

The keyword side reads chunk text only, so its ranking is the same for every embedder. Whatever separates these three columns arrived from the dense side.

| Embedder | @5 | @10 | @20 |
| --- | --- | --- | --- |
| voyage-2 | +0.74 | +0.01 | +0.44 |
| arctic-embed-l-v2 | +2.12 | +0.82 | +1.21 |
| text-embedding-3-large | +1.11 | +0.35 | +1.11 |

The same gains as a share of the error that was still left after contextual chunking -- the headroom correction, applied a second time so the two interventions can be read on one scale:

| Embedder | @5 | @10 | @20 |
| --- | --- | --- | --- |
| voyage-2 | 5.9% | 0.1% | 8.7% |
| arctic-embed-l-v2 | 15.2% | 9.2% | 22.5% |
| text-embedding-3-large | 8.3% | 3.9% | 21.2% |

### The same ordering, from a different mechanism

Ranked by their contextual score, arctic-embed-l-v2 then text-embedding-3-large then voyage-2 runs weakest to strongest, and the share of remaining error that the keyword fusion closes falls in that same order at every k: 15.2% / 8.3% / 5.9% at k=5, 9.2% / 3.9% / 0.1% at k=10, 22.5% / 21.2% / 8.7% at k=20.

That is the second time this ordering appears, and the two interventions have nothing in common. One writes a line of generated prose in front of a chunk and re-embeds it. The other leaves the vectors untouched and merges in a ranking built from word counts. Both pay off in inverse proportion to how good the dense retriever already was.

A prediction written down before these three runs: that the fusion would LOSE at k=5, because the published table drops from 88.12 to 86.43 there. It gained on all three embedders. Whatever produces that drop in the published row is particular to that setup, and the local keyword engine is one candidate among several. The prediction is left here because it was wrong, which is the only reason it is worth anything.

### Where the three end up

| Spread between the embedders | @5 | @10 | @20 |
| --- | --- | --- | --- |
| after baseline | 7.11 | 5.76 | 3.07 |
| after contextual | 1.34 | 1.01 | 0.37 |
| after hybrid | 0.45 | 0.66 | 0.44 |

At k=5 the ordering is exactly reversed. text-embedding-3-large has the best baseline and the worst hybrid score; arctic-embed-l-v2 has the worst baseline and the best hybrid score. The three span 7.11 points before either technique and 0.45 after both.

At k=20 the spread stops narrowing: 0.37 after contextual and 0.44 after fusion. Convergence is not a law here, it is what these two techniques happen to do at the values of k where there is still room to move.

## Not yet run

- titan-embed-v2, baseline — `python reproduce.py baseline --embedder=titan`
- titan-embed-v2, contextual — `python reproduce.py contextual --embedder=titan`
- titan-embed-v2, hybrid — `python reproduce.py hybrid --embedder=titan`

