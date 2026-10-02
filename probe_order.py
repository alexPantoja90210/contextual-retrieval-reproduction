#!/usr/bin/env python3
"""Does the concatenation order of chunk and context change the result?

The published guide embeds the generated context BEFORE the chunk:

    text_to_embed = f"{contextualized_text}\n\n{chunk['content']}"     # guide

This repository has been embedding it AFTER:

    content = f"{chunk['content']}\n\n{text}"              # reproduce.py:432

Both orders carry the same words, so a bag-of-words retriever would not notice.
An embedding model reads a sequence, so it can. Which of those is true here is
measurable, and this script measures it rather than arguing it.

It is a probe, not a stage. It writes its own cache and its own result file and
touches nothing that exists:

    contextual_arctic_guideorder_vector_db.pkl
    result_contextual_arctic_guideorder.json

Arctic Embed runs locally, so this costs nothing and needs no key. The query
vectors are reused from the existing arctic run -- query text is unchanged by
chunk order, so re-embedding them would only add a way to differ by accident.
Claude is never called: the context text was written once and is read off disk.

    python probe_order.py

Pass@k here is produced by the same evaluate() the stages use, so the number
is comparable to result_contextual_arctic.json line for line.
"""
import json
import os
import pickle
import sys
import time

from embedders import DIMENSIONS
from reproduce import (HERE, QUERIES, ContextualVectorDB, K_VALUES, evaluate,
                       load_jsonl, make_embedder)

# The stages keep their caches under data/, so the probe writes its own there
# too. Pointing it anywhere else would make it look like a stage that had never
# been run rather than a probe.
DATA = os.path.join(HERE, "data")
DONOR = os.path.join(DATA, "contextual_vector_db.pkl")
EXISTING = os.path.join(DATA, "contextual_arctic_vector_db.pkl")
CACHE = os.path.join(DATA, "contextual_arctic_guideorder_vector_db.pkl")
RESULT = os.path.join(HERE, "result_contextual_arctic_guideorder.json")
SEPARATOR = "\n\n"


def split_context(meta):
    """Recover the generated context from the stored combined text.

    The stage stored `original_content` and `content`, where content is
    original + separator + context. The context is what is left after removing
    the two known parts. Asserted rather than assumed, per chunk: if a single
    chunk does not have this shape the probe stops instead of silently
    embedding a truncated string.
    """
    original, combined = meta["original_content"], meta["content"]
    head = original + SEPARATOR
    if not combined.startswith(head):
        sys.exit(f"chunk {meta['chunk_id']} is not original + separator + context; "
                 f"the probe's assumption about the stored layout is wrong")
    return combined[len(head):]


def main():
    for path, what in ((DONOR, "the contextualized text"),
                       (EXISTING, "the existing arctic run")):
        if not os.path.exists(path):
            sys.exit(f"missing {os.path.basename(path)}, which holds {what}")

    with open(DONOR, "rb") as f:
        metadata = pickle.load(f)["metadata"]
    with open(EXISTING, "rb") as f:
        existing = pickle.load(f)
    query_cache = existing.get("query_cache", {})

    queries = load_jsonl(QUERIES)
    wanted = {q["query"] for q in queries}
    missing = wanted - set(query_cache)
    if missing:
        sys.exit(f"{len(missing)} query vectors are not cached; run\n"
                 "  python reproduce.py contextual --embedder=arctic\nfirst")

    embedder = make_embedder("arctic")
    db = ContextualVectorDB(embedder)
    db.db_path = CACHE          # never write over the stage's cache
    db.metadata = metadata
    db.query_cache = query_cache

    print(f"\n{len(metadata)} chunks, {len(queries)} queries")
    print(f"embedder: {embedder.label} at {DIMENSIONS} dimensions, locally")
    print("  context moved in FRONT of the chunk, as the guide writes it")
    print("  everything else held fixed: same chunks, same context text,")
    print("  same query vectors, same scoring\n")

    started = time.time()
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            db.embeddings = pickle.load(f)["embeddings"]
        print(f"  loaded {len(db.embeddings)} embeddings from a previous probe")
    else:
        texts = [split_context(m) + SEPARATOR + m["original_content"] for m in metadata]
        db.embeddings = db.embed_all(texts, "chunks")
        db.save()
        print(f"  cached to {os.path.basename(CACHE)}")

    mine = {k: evaluate(db, queries, k) for k in K_VALUES}

    with open(os.path.join(HERE, "result_contextual_arctic.json"), encoding="utf-8") as f:
        stage = {int(k): v for k, v in json.load(f)["mine"].items()}

    width = 62
    print(f"\n{'=' * width}")
    print("  CONCATENATION ORDER  —  arctic-embed-l-v2, contextual")
    print(f"{'=' * width}")
    print(f"  {'':9}{'chunk first':>14}{'context first':>16}{'diff':>9}")
    print(f"  {'':9}{'(this repo)':>14}{'(the guide)':>16}")
    for k in K_VALUES:
        print(f"  Pass@{k:<4}{stage[k]:>13.2f}%{mine[k]:>15.2f}%{mine[k] - stage[k]:>+9.2f}")
    print(f"{'-' * width}")
    print("  Same 737 chunks, same generated context, same 1024 dimensions,")
    print("  same query vectors, same scoring. Word order is the only change.")
    print(f"{'=' * width}\n")
    print(f"  elapsed: {time.time() - started:.0f}s")

    with open(RESULT, "w", encoding="utf-8") as f:
        json.dump({"probe": "concatenation order", "embedder": "arctic",
                   "order": "context_then_chunk", "mine": mine,
                   "repo_order_chunk_then_context": stage,
                   "chunks": len(metadata), "queries": len(queries),
                   "embed_model": embedder.label, "dimensions": DIMENSIONS},
                  f, indent=2)
    print(f"  written: {os.path.basename(RESULT)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
