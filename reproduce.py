#!/usr/bin/env python3
"""Reproduces Anthropic's Contextual Retrieval benchmark, and says whether it matched.

The published guide reports Pass@k on 737 chunks from 9 codebases against 248
queries, each with a known correct ("golden") chunk. The retrieval logic here is
the guide's own, unchanged: voyage-2 embeddings, dot-product similarity, top-k,
and the same scoring function. What this script adds is the comparison — your
numbers printed next to the published ones, with a verdict.

Run it in stages. Each stage caches its vector database to disk, so a second run
costs nothing and re-scores in seconds.

  stage 1  baseline      VOYAGE_API_KEY only        ~$0.05, a few minutes
  stage 2  contextual    + ANTHROPIC_API_KEY        ~$1-3, 15-30 minutes

Stages 3 and 4 in the guide (hybrid BM25, reranking) need Elasticsearch in
Docker and a Cohere key; they are not run here.

  python reproduce.py baseline
  python reproduce.py contextual
"""
import json
import os
import pickle
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import numpy as np
except ImportError:
    sys.exit("numpy is not installed.\n  python -m pip install numpy")

MODEL_NAME = "claude-haiku-4-5"
EMBED_MODEL = "voyage-2"
K_VALUES = [5, 10, 20]

# Voyage's free tier, before a payment method is added, allows 3 requests and
# 10,000 tokens per minute. Both are respected below. Adding a payment method
# raises these a lot and the free token allowance still applies, so pass
# --rpm/--tpm to go faster if you have done that.
FREE_RPM, FREE_TPM = 3, 10_000
# A single request must fit inside the per-minute token budget with room to
# spare, or it can never be sent.
MAX_TOKENS_PER_REQUEST = 8_000


def est_tokens(text):
    """Rough and deliberately generous: characters over four."""
    return max(1, len(text) // 4)


class RateLimiter:
    """Respects a requests-per-minute and a tokens-per-minute budget at once.

    The token budget is what actually binds here: the corpus is about 124,000
    estimated tokens, so on the free tier the floor is roughly 13 minutes no
    matter how the requests are arranged.
    """

    def __init__(self, rpm, tpm):
        self.rpm, self.tpm = rpm, tpm
        self.events = []          # (timestamp, tokens)

    def _prune(self, now):
        self.events = [e for e in self.events if now - e[0] < 60.0]

    def wait_for(self, tokens):
        while True:
            now = time.time()
            self._prune(now)
            used_req = len(self.events)
            used_tok = sum(t for _, t in self.events)
            if used_req < self.rpm and used_tok + tokens <= self.tpm:
                self.events.append((now, tokens))
                return
            oldest = min(e[0] for e in self.events)
            sleep = max(0.5, 60.0 - (now - oldest) + 0.25)
            print(f"  rate limit: waiting {sleep:.0f}s "
                  f"({used_req}/{self.rpm} req, {used_tok}/{self.tpm} tok in the last minute)",
                  end="\r")
            time.sleep(sleep)


def batch_by_tokens(texts, cap=MAX_TOKENS_PER_REQUEST):
    """Group texts so no request exceeds the per-request token cap."""
    batches, cur, cur_tokens = [], [], 0
    for t in texts:
        n = est_tokens(t)
        if cur and cur_tokens + n > cap:
            batches.append(cur)
            cur, cur_tokens = [], 0
        cur.append(t)
        cur_tokens += n
    if cur:
        batches.append(cur)
    return batches

# What the guide reports. The whole point of this script is the comparison.
PUBLISHED = {
    "baseline":   {5: 80.92, 10: 87.15, 20: 90.06},
    "contextual": {5: 88.12, 10: 92.34, 20: 94.29},
}
# Embeddings are deterministic, so a baseline that misses by more than this is a
# real difference. Stage 2 involves generation at temperature 0, which is close
# to deterministic but not guaranteed to be identical.
TOLERANCE = {"baseline": 0.5, "contextual": 2.0}

HERE = os.path.dirname(os.path.abspath(__file__))
CHUNKS = os.path.join(HERE, "data", "codebase_chunks.json")
QUERIES = os.path.join(HERE, "data", "evaluation_set.jsonl")


def require(module, package=None):
    """Import, or say plainly what to install. A traceback is not an answer."""
    try:
        return __import__(module)
    except ImportError:
        sys.exit(f"{module} is not installed.\n"
                 f"  python -m pip install {package or module}")


def need(var, where):
    v = os.getenv(var)
    if not v:
        sys.exit(f"{var} is not set. Get one at {where}, then:\n"
                 f'  PowerShell:  $env:{var} = "..."\n'
                 f"  bash:        export {var}=...")
    return v


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# --------------------------------------------------------------- vector stores
class VectorDB:
    """The guide's baseline store: embed each chunk as written."""

    name = "baseline"

    def __init__(self, rpm=FREE_RPM, tpm=FREE_TPM):
        voyageai = require("voyageai")
        self.client = voyageai.Client(api_key=need("VOYAGE_API_KEY", "voyageai.com"))
        self.embeddings, self.metadata, self.query_cache = [], [], {}
        self.limiter = RateLimiter(rpm, tpm)
        self.db_path = os.path.join(HERE, "data", f"{self.name}_vector_db.pkl")

    def embed_batch(self, texts):
        """One request, throttled, retried on a rate-limit answer."""
        tokens = sum(est_tokens(t) for t in texts)
        for attempt in range(6):
            self.limiter.wait_for(tokens)
            try:
                return self.client.embed(texts, model=EMBED_MODEL).embeddings
            except Exception as exc:
                if "rate limit" not in str(exc).lower() or attempt == 5:
                    raise
                back = 20 * (attempt + 1)
                print(f"  rate limited by the server, backing off {back}s     ")
                time.sleep(back)
        raise RuntimeError("unreachable")

    def embed_all(self, texts, label):
        batches = batch_by_tokens(texts)
        total_tokens = sum(est_tokens(t) for t in texts)
        minutes = total_tokens / max(1, self.limiter.tpm)
        print(f"  {label}: {len(texts)} texts, ~{total_tokens:,} tokens, "
              f"{len(batches)} requests")
        print(f"  at {self.limiter.tpm:,} tokens/min this takes about "
              f"{minutes:.0f} min")
        out, done = [], 0
        for b in batches:
            out.extend(self.embed_batch(b))
            done += len(b)
            print(f"  {label}: {done}/{len(texts)}                              ", end="\r")
        print(f"  {label}: {len(texts)}/{len(texts)} done                       ")
        return out

    def embed_queries(self, queries):
        """All query embeddings up front, in batches.

        Embedding them one at a time is 248 requests. On the free tier that is
        over an hour of waiting for 3,600 tokens of text.
        """
        unique = sorted({q["query"] for q in queries})
        missing = [q for q in unique if q not in self.query_cache]
        if not missing:
            return
        vecs = self.embed_all(missing, "queries")
        for q, v in zip(missing, vecs):
            self.query_cache[q] = v
        self.save()

    def texts_and_metadata(self, dataset):
        texts, meta = [], []
        for doc in dataset:
            for chunk in doc["chunks"]:
                texts.append(chunk["content"])
                meta.append({"doc_id": doc["doc_id"], "chunk_id": chunk["chunk_id"],
                             "original_index": chunk["original_index"],
                             "content": chunk["content"]})
        return texts, meta

    def load_data(self, dataset):
        if os.path.exists(self.db_path):
            with open(self.db_path, "rb") as f:
                d = pickle.load(f)
            self.embeddings, self.metadata = d["embeddings"], d["metadata"]
            self.query_cache = d.get("query_cache", {})
            print(f"  loaded {len(self.embeddings)} embeddings and "
                  f"{len(self.query_cache)} query vectors from cache")
            return
        texts, meta = self.texts_and_metadata(dataset)
        self.embed_and_store(texts, meta)

    def embed_and_store(self, texts, meta):
        self.embeddings = self.embed_all(texts, "chunks")
        self.metadata = meta
        self.save()
        print(f"  cached to {os.path.basename(self.db_path)}")

    def save(self):
        with open(self.db_path, "wb") as f:
            pickle.dump({"embeddings": self.embeddings, "metadata": self.metadata,
                         "query_cache": self.query_cache}, f)

    def search(self, query, k=20):
        if query not in self.query_cache:
            self.query_cache[query] = self.embed_batch([query])[0]
        sims = np.dot(self.embeddings, self.query_cache[query])
        return [{"metadata": self.metadata[i]} for i in np.argsort(sims)[::-1][:k]]


DOCUMENT_CONTEXT_PROMPT = """
<document>
{doc_content}
</document>
"""
CHUNK_CONTEXT_PROMPT = """
Here is the chunk we want to situate within the whole document
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else.
"""


class ContextualVectorDB(VectorDB):
    """The guide's contextual store: Claude writes a situating line per chunk,
    which is prepended to the chunk before embedding. The whole document is sent
    with each request and cached, so the document is billed once per document
    rather than once per chunk."""

    name = "contextual"

    def __init__(self, rpm=FREE_RPM, tpm=FREE_TPM):
        super().__init__(rpm=rpm, tpm=tpm)
        anthropic = require("anthropic")
        self.anthropic = anthropic.Anthropic(
            api_key=need("ANTHROPIC_API_KEY", "console.anthropic.com"))
        self.db_path = os.path.join(HERE, "data", f"{self.name}_vector_db.pkl")
        self.tokens = {"input": 0, "output": 0, "cache_read": 0, "cache_creation": 0}
        self.lock = threading.Lock()

    def situate(self, doc, chunk):
        r = self.anthropic.messages.create(
            model=MODEL_NAME, max_tokens=1000, temperature=0.0,
            messages=[{"role": "user", "content": [
                {"type": "text",
                 "text": DOCUMENT_CONTEXT_PROMPT.format(doc_content=doc),
                 "cache_control": {"type": "ephemeral"}},
                {"type": "text",
                 "text": CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk)},
            ]}],
        )
        return r.content[0].text, r.usage

    def load_data(self, dataset, threads=6):
        if os.path.exists(self.db_path):
            with open(self.db_path, "rb") as f:
                d = pickle.load(f)
            self.embeddings, self.metadata = d["embeddings"], d["metadata"]
            print(f"  loaded {len(self.embeddings)} embeddings from cache")
            return

        jobs = [(doc, ch) for doc in dataset for ch in doc["chunks"]]
        print(f"  situating {len(jobs)} chunks with {MODEL_NAME} ({threads} threads)")
        done = [0]

        def work(doc, chunk):
            text, usage = self.situate(doc["content"], chunk["content"])
            with self.lock:
                self.tokens["input"] += usage.input_tokens
                self.tokens["output"] += usage.output_tokens
                self.tokens["cache_read"] += getattr(usage, "cache_read_input_tokens", 0) or 0
                self.tokens["cache_creation"] += getattr(usage, "cache_creation_input_tokens", 0) or 0
                done[0] += 1
                print(f"  {done[0]}/{len(jobs)}", end="\r")
            return {"doc_id": doc["doc_id"], "chunk_id": chunk["chunk_id"],
                    "original_index": chunk["original_index"],
                    "original_content": chunk["content"],
                    "content": f"{chunk['content']}\n\n{text}"}

        results = []
        with ThreadPoolExecutor(max_workers=threads) as ex:
            futures = [ex.submit(work, d, c) for d, c in jobs]
            for f in as_completed(futures):
                results.append(f.result())
        print()
        t = self.tokens
        print(f"  tokens: {t['input']} in, {t['output']} out, "
              f"{t['cache_read']} cache read, {t['cache_creation']} cache written")
        if t["cache_read"] + t["cache_creation"]:
            saved = t["cache_read"] / max(1, t["cache_read"] + t["cache_creation"])
            print(f"  {saved:.0%} of document tokens came from cache rather than being re-billed")
        self.embed_and_store([r["content"] for r in results], results)


# ------------------------------------------------------------------- scoring
def evaluate(db, queries, k):
    """The guide's scoring, unchanged: for each query, what share of its golden
    chunks appear in the top k."""
    total = 0.0
    for i, item in enumerate(queries, 1):
        golden = []
        for doc_uuid, idx in item["golden_chunk_uuids"]:
            doc = next((d for d in item["golden_documents"] if d["uuid"] == doc_uuid), None)
            if not doc:
                continue
            ch = next((c for c in doc["chunks"] if c["index"] == idx), None)
            if ch:
                golden.append(ch["content"].strip())
        if not golden:
            continue
        got = db.search(item["query"], k=k)
        found = 0
        for g in golden:
            for doc in got[:k]:
                m = doc["metadata"]
                if m.get("original_content", m.get("content", "")).strip() == g:
                    found += 1
                    break
        total += found / len(golden)
        print(f"  Pass@{k}: {i}/{len(queries)}", end="\r")
    print(" " * 40, end="\r")
    return 100 * total / len(queries)


def report(stage, mine):
    pub = PUBLISHED[stage]
    tol = TOLERANCE[stage]
    print(f"\n{'=' * 64}")
    print(f"  {stage.upper()}  —  yours against the published guide")
    print(f"{'=' * 64}")
    print(f"  {'':8}{'published':>12}{'yours':>10}{'diff':>9}")
    ok = True
    for k in K_VALUES:
        d = mine[k] - pub[k]
        within = abs(d) <= tol
        ok = ok and within
        print(f"  Pass@{k:<4}{pub[k]:>11.2f}%{mine[k]:>9.2f}%{d:>+8.2f}"
              f"   {'' if within else '  <-- outside tolerance'}")
    print(f"{'-' * 64}")
    print(f"  tolerance: {tol:.1f} points")
    if ok:
        print(f"\n  REPRODUCED. Your run matches the published benchmark.")
    else:
        print(f"\n  DID NOT MATCH. Something differs — a different embedding model,")
        print(f"  a partial dataset, or a changed scoring rule. Worth finding before")
        print(f"  claiming either number.")
    print(f"{'=' * 64}\n")
    return ok


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a.split("=")[0]: a.split("=")[1]
             for a in sys.argv[1:] if a.startswith("--") and "=" in a}
    stage = args[0] if args else "baseline"
    if stage not in PUBLISHED:
        sys.exit(f"stage must be one of: {', '.join(PUBLISHED)}\n"
                 f"  python reproduce.py baseline [--rpm=3] [--tpm=10000]")
    rpm = int(flags.get("--rpm", FREE_RPM))
    tpm = int(flags.get("--tpm", FREE_TPM))

    dataset = json.load(open(CHUNKS, encoding="utf-8"))
    queries = load_jsonl(QUERIES)
    chunks = sum(len(d["chunks"]) for d in dataset)
    print(f"\n{chunks} chunks from {len(dataset)} documents, {len(queries)} queries")
    if (rpm, tpm) == (FREE_RPM, FREE_TPM):
        print("Voyage free tier: 3 requests and 10,000 tokens per minute.")
        print("With a payment method on file the limits rise and the free token")
        print("allowance still applies — then pass --rpm=300 --tpm=1000000.\n")
    else:
        print(f"Rate limits: {rpm} requests/min, {tpm:,} tokens/min\n")

    started = time.time()
    db = (VectorDB if stage == "baseline" else ContextualVectorDB)(rpm=rpm, tpm=tpm)
    db.load_data(dataset)
    db.embed_queries(queries)

    mine = {}
    for k in K_VALUES:
        mine[k] = evaluate(db, queries, k)
        print(f"  Pass@{k}: {mine[k]:.2f}%")

    matched = report(stage, mine)
    print(f"  elapsed: {time.time() - started:.0f}s")
    with open(os.path.join(HERE, f"result_{stage}.json"), "w", encoding="utf-8") as f:
        json.dump({"stage": stage, "mine": mine, "published": PUBLISHED[stage],
                   "matched": matched, "chunks": chunks, "queries": len(queries),
                   "embed_model": EMBED_MODEL,
                   "model": MODEL_NAME if stage == "contextual" else None},
                  f, indent=2)
    print(f"  written: result_{stage}.json")
    return 0 if matched else 1


if __name__ == "__main__":
    raise SystemExit(main())
