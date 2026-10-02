#!/usr/bin/env python3
"""Why the reranked columns come out identical: recall at the pool size.

The rerank stage hands a cross-encoder 200 candidates per query. If every
embedder's 200 already contain the golden chunk, the cross-encoder sees the same
relevant documents whichever retriever chose them, and the choice of retriever
stops mattering. That is an explanation, and RESULTS.md states it as one, so it
has to be measurable rather than asserted.

This measures two things and writes them where consolidate.py can read them:

  recall@200   the share of golden chunks inside each embedder's pool
  overlap      how much the pools have in common, pairwise

The second is what keeps the first from being trivial. If the three pools were
the same 200 chunks, identical results would say nothing. They are not.

Offline, no key, no model: reads the contextual stores already on disk.

    python probe_pool.py
"""
import json
import os
import sys
from itertools import combinations

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT = os.path.join(HERE, "result_pool_recall.json")
POOL = 200

RUNS = [("voyage-2", ""), ("arctic-embed-l-v2", "_arctic"),
        ("titan-embed-v2", "_titan"), ("text-embedding-3-large", "_azure")]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def golden_texts(item):
    out = []
    for doc_uuid, index in item["golden_chunk_uuids"]:
        doc = next((d for d in item["golden_documents"] if d["uuid"] == doc_uuid), None)
        if not doc:
            continue
        chunk = next((c for c in doc["chunks"] if c["index"] == index), None)
        if chunk:
            out.append(chunk["content"].strip())
    return out


def main():
    import pickle
    queries = load_jsonl(os.path.join(HERE, "data", "evaluation_set.jsonl"))
    recall, pools = {}, {}

    for label, suffix in RUNS:
        path = os.path.join(HERE, "data", f"contextual{suffix}_vector_db.pkl")
        if not os.path.exists(path):
            continue
        with open(path, "rb") as f:
            d = pickle.load(f)
        emb = np.asarray(d["embeddings"])
        meta, cache = d["metadata"], d["query_cache"]
        text_of = [m.get("original_content", m["content"]).strip() for m in meta]

        found, picked = 0.0, []
        for item in queries:
            wanted = golden_texts(item)
            top = np.argsort(np.dot(emb, cache[item["query"]]))[::-1][:POOL]
            picked.append(set(top.tolist()))
            if not wanted:
                continue
            have = {text_of[i] for i in top}
            found += sum(1 for w in wanted if w in have) / len(wanted)
        # Divided by every query, like the Pass@k scoring, so the two are
        # on the same scale and a reader can compare them directly.
        recall[label] = 100 * found / len(queries)
        pools[label] = picked

    if len(recall) < 2:
        sys.exit("needs at least two contextual stores on disk")

    overlap = {}
    for a, b in combinations(pools, 2):
        shared = float(np.mean([len(x & y) for x, y in zip(pools[a], pools[b])]))
        overlap[f"{a} vs {b}"] = {"chunks": shared, "percent": 100 * shared / POOL}

    width = 62
    print(f"\n{'=' * width}")
    print(f"  CANDIDATE POOL  —  {POOL} chunks per query, from {len(queries)} queries")
    print(f"{'=' * width}")
    for label, value in recall.items():
        print(f"  recall@{POOL}  {label:30} {value:6.2f}%")
    print(f"{'-' * width}")
    for pair, d in overlap.items():
        print(f"  {pair:44} {d['chunks']:5.1f} of {POOL} shared")
    print(f"{'-' * width}")
    print("  High recall with low overlap is the point: the pools are")
    print("  different sets that agree on the few chunks that matter.")
    print(f"{'=' * width}\n")

    with open(RESULT, "w", encoding="utf-8") as f:
        json.dump({"pool": POOL, "queries": len(queries),
                   "recall": recall, "overlap": overlap}, f, indent=2)
    print(f"  written: {os.path.basename(RESULT)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
