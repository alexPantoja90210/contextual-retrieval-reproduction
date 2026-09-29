#!/usr/bin/env python3
"""Reproduces Anthropic's Contextual Retrieval benchmark, and says whether it matched.

The published guide reports Pass@k on 737 chunks from 9 codebases against 248
queries, each with a known correct ("golden") chunk. The retrieval logic here is
the guide's own, unchanged: voyage-2 embeddings, dot-product similarity, top-k,
and the same scoring function. What this script adds is the comparison — your
numbers printed next to the published ones, with a verdict.

Run it in stages. Each caches its vector database to disk, so a second run costs
nothing and re-scores in seconds.

  stage 1  baseline      VOYAGE_API_KEY only
  stage 2  contextual    + ANTHROPIC_API_KEY

  python reproduce.py baseline
  python reproduce.py contextual [--rpm=3] [--tpm=10000]
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
# 10,000 tokens per minute. Both are respected. A payment method raises these
# a lot and the free token allowance still applies, so pass --rpm/--tpm then.
FREE_RPM, FREE_TPM = 3, 10_000
# A request is sized well under the per-minute budget, and the minute is not
# filled to the brim. Token counts are never exact, and something sized at the
# limit has nowhere to be wrong.
REQUEST_FRACTION = 0.55
BUDGET_FRACTION = 0.85

# Claude Haiku 4.5, dollars per million tokens.
HAIKU_IN, HAIKU_OUT = 1.00, 5.00
HAIKU_CACHE_WRITE, HAIKU_CACHE_READ = 1.25, 0.10
# Stage 2 should cost well under a dollar on this corpus. The guard is here so
# a run that goes wrong stops early instead of spending an account's balance.
DEFAULT_BUDGET = 1.50

PUBLISHED = {
    "baseline":   {5: 80.92, 10: 87.15, 20: 90.06},
    "contextual": {5: 88.12, 10: 92.34, 20: 94.29},
}
# Embeddings are deterministic, so a baseline off by more than this is a real
# difference. Stage 2 generates text at temperature 0 — close to deterministic,
# not guaranteed identical.
TOLERANCE = {"baseline": 0.5, "contextual": 2.0}

QUIET = bool(os.getenv("REPRODUCE_QUIET"))
HERE = os.path.dirname(os.path.abspath(__file__))
CHUNKS = os.path.join(HERE, "data", "codebase_chunks.json")
QUERIES = os.path.join(HERE, "data", "evaluation_set.jsonl")


def require(module, package=None):
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


# ------------------------------------------------------------------- tokens
# The real tokenizer when it loads, a pessimistic estimate otherwise. An earlier
# version divided characters by four, which is about right for prose and badly
# wrong for source code: it undercounted this corpus by 60%, and the very first
# request went out over the per-minute budget.
_TOKENIZER, _TRIED = None, False


def _tokenizer():
    global _TOKENIZER, _TRIED
    if not _TRIED:
        _TRIED = True
        try:
            from tokenizers import Tokenizer
            _TOKENIZER = Tokenizer.from_pretrained("voyageai/voyage-2")
            print("  counting tokens with the voyage-2 tokenizer")
        except Exception:
            print("  voyage-2 tokenizer unavailable; estimating at 2.5 chars/token")
    return _TOKENIZER


def est_tokens(text):
    t = _tokenizer()
    if t is not None:
        try:
            return max(1, len(t.encode(text).ids))
        except Exception:
            pass
    return max(1, int(len(text) / 2.5))


class RateLimiter:
    """Holds to a requests-per-minute and a tokens-per-minute budget at once."""

    def __init__(self, rpm, tpm):
        self.rpm, self.tpm = rpm, tpm
        self.events = []

    def penalize(self, tokens):
        """A refused request still counted against the server's window.

        Charge what was attempted, not a punishment. An earlier version spent
        half the minute per rejection, so two rejections stopped everything for
        a full minute — a wrong token count turned into dead waiting.
        """
        self.events.append((time.time(), tokens))

    def wait_for(self, tokens):
        while True:
            now = time.time()
            self.events = [e for e in self.events if now - e[0] < 60.0]
            n_req = len(self.events)
            n_tok = sum(t for _, t in self.events)
            if n_req < self.rpm and n_tok + tokens <= self.tpm:
                self.events.append((now, tokens))
                return
            oldest = min(e[0] for e in self.events)
            sleep = max(0.5, 60.0 - (now - oldest) + 0.25)
            print(f"  waiting {sleep:>4.0f}s for the rate window "
                  f"({n_req}/{self.rpm} req, {n_tok:,}/{self.tpm:,} tokens)   ",
                  end="\r")
            time.sleep(sleep)


# ------------------------------------------------------------- vector stores
class VectorDB:
    """The guide's baseline store: embed each chunk as written."""

    name = "baseline"

    def __init__(self, rpm=FREE_RPM, tpm=FREE_TPM):
        voyageai = require("voyageai")
        self.client = voyageai.Client(api_key=need("VOYAGE_API_KEY", "voyageai.com"))
        self.embeddings, self.metadata, self.query_cache = [], [], {}
        self.limiter = RateLimiter(rpm, int(tpm * BUDGET_FRACTION))
        self.request_cap = max(400, int(tpm * REQUEST_FRACTION))
        self._tok = {}
        self.db_path = os.path.join(HERE, "data", f"{self.name}_vector_db.pkl")

    # -- token accounting ----------------------------------------------------
    def tokens(self, text):
        if text not in self._tok:
            self._tok[text] = est_tokens(text)
        return self._tok[text]

    # -- embedding -----------------------------------------------------------
    def embed_all(self, texts, label):
        """Embed everything, learning the request size the server will accept.

        One loop, no recursion. A refused request shrinks the cap and the batch
        is rebuilt smaller from the same position, so the run does not depend on
        the token count being right — only on the error being fixable.
        """
        total = sum(self.tokens(t) for t in texts)
        print(f"  {label}: {len(texts)} texts, {total:,} tokens")
        print(f"  at {self.limiter.tpm:,} tokens/min that is about "
              f"{total / max(1, self.limiter.tpm):.0f} min")

        out, i, refused, stuck = [], 0, 0, 0
        while i < len(texts):
            batch, used, j = [], 0, i
            while j < len(texts):
                n = self.tokens(texts[j])
                if batch and used + n > self.request_cap:
                    break
                batch.append(texts[j])
                used += n
                j += 1

            self.limiter.wait_for(used)
            try:
                out.extend(self.client.embed(batch, model=EMBED_MODEL).embeddings)
            except Exception as exc:
                if "rate limit" not in str(exc).lower():
                    raise
                refused += 1
                self.limiter.penalize(used)
                if len(batch) > 1:
                    self.request_cap = max(400, used // 2)
                    print(f"  refused at {len(batch)} texts ({used:,} tokens); "
                          f"request cap is now {self.request_cap:,}           ")
                    continue
                stuck += 1
                if stuck > 2:
                    sys.exit(
                        f"\n  A single text of {used:,} tokens keeps being refused.\n"
                        f"  It cannot be split further, so no retry will help: the\n"
                        f"  account's per-minute token limit is below the size of one\n"
                        f"  chunk. Raise the limit on the provider, or lower --tpm to\n"
                        f"  match what the account really allows.")
                print(f"  one text of {used:,} tokens refused; waiting the window out"
                      f"          ")
                time.sleep(1 if QUIET else 61)
                continue
            i = j
            stuck = 0
            if not QUIET:
                print(f"  {label}: {i}/{len(texts)}  cap {self.request_cap:,}"
                      f"  {refused} refused            ", end="\r")
        print(f"  {label}: {len(texts)}/{len(texts)} done"
              f"                                   ")
        return out

    # -- data ----------------------------------------------------------------
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
            self.embeddings = d["embeddings"]
            self.metadata = d["metadata"]
            self.query_cache = d.get("query_cache", {})
            print(f"  loaded {len(self.embeddings)} embeddings and "
                  f"{len(self.query_cache)} query vectors from cache")
            return
        texts, meta = self.texts_and_metadata(dataset)
        self.embeddings = self.embed_all(texts, "chunks")
        self.metadata = meta
        self.save()
        print(f"  cached to {os.path.basename(self.db_path)}")

    def save(self):
        with open(self.db_path, "wb") as f:
            pickle.dump({"embeddings": self.embeddings, "metadata": self.metadata,
                         "query_cache": self.query_cache}, f)

    def embed_queries(self, queries):
        """All query vectors up front, in batches.

        One at a time is 248 requests. On the free tier that is over an hour of
        waiting for a few thousand tokens of text.
        """
        unique = sorted({q["query"] for q in queries})
        missing = [q for q in unique if q not in self.query_cache]
        if not missing:
            return
        for q, v in zip(missing, self.embed_all(missing, "queries")):
            self.query_cache[q] = v
        self.save()

    def search(self, query, k=20):
        if query not in self.query_cache:
            self.query_cache[query] = self.embed_all([query], "query")[0]
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
    """The guide's contextual store: Claude writes a line situating each chunk in
    its document, which is prepended before embedding. The document is sent with
    every request and cached, so it is billed once per document rather than once
    per chunk."""

    name = "contextual"

    def __init__(self, rpm=FREE_RPM, tpm=FREE_TPM, budget=DEFAULT_BUDGET):
        super().__init__(rpm=rpm, tpm=tpm)
        self.budget = budget
        anthropic = require("anthropic")
        self.anthropic = anthropic.Anthropic(
            api_key=need("ANTHROPIC_API_KEY", "console.anthropic.com"))
        self.db_path = os.path.join(HERE, "data", f"{self.name}_vector_db.pkl")
        self.usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
        self.lock = threading.Lock()
        self.temperature_ok = True
        self._temp_kwargs = self._temperature_kwargs()

    def _temperature_kwargs(self):
        """Pass temperature in whichever way this SDK version accepts.

        The published guide calls messages.create(temperature=0.0). SDK 1.9
        dropped temperature from that signature, so the guide as written raises
        TypeError on a current install. The API still takes the field, so it
        goes through extra_body instead — and if the server refuses it too, the
        run continues without it and says so, because a default temperature is
        a small loss of determinism, not a reason to stop.
        """
        import inspect
        try:
            params = inspect.signature(type(self.anthropic.messages).create).parameters
            if "temperature" in params:
                return {"temperature": 0.0}
        except (TypeError, ValueError):
            pass
        return {"extra_body": {"temperature": 0.0}}

    def situate(self, doc, chunk):
        body = dict(
            model=MODEL_NAME, max_tokens=1000,
            messages=[{"role": "user", "content": [
                {"type": "text",
                 "text": DOCUMENT_CONTEXT_PROMPT.format(doc_content=doc),
                 "cache_control": {"type": "ephemeral"}},
                {"type": "text",
                 "text": CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk)},
            ]}],
        )
        if self.temperature_ok:
            body.update(self._temp_kwargs)
        try:
            r = self.anthropic.messages.create(**body)
        except Exception as exc:
            if self.temperature_ok and "temperature" in str(exc).lower():
                with self.lock:
                    if self.temperature_ok:
                        self.temperature_ok = False
                        print("  this model does not take temperature; continuing "
                              "at the default                    ")
                r = self.anthropic.messages.create(
                    **{k: v for k, v in body.items()
                       if k not in ("temperature", "extra_body")})
            else:
                raise
        return r.content[0].text, r.usage

    def spend(self):
        """Dollars so far, from the usage the API actually reported."""
        u = self.usage
        return (u["input"] * HAIKU_IN + u["output"] * HAIKU_OUT
                + u["cache_write"] * HAIKU_CACHE_WRITE
                + u["cache_read"] * HAIKU_CACHE_READ) / 1_000_000

    def record(self, usage):
        self.usage["input"] += usage.input_tokens
        self.usage["output"] += usage.output_tokens
        self.usage["cache_read"] += getattr(usage, "cache_read_input_tokens", 0) or 0
        self.usage["cache_write"] += getattr(usage, "cache_creation_input_tokens", 0) or 0

    def load_data(self, dataset, threads=4):
        if os.path.exists(self.db_path):
            return VectorDB.load_data(self, dataset)

        total = sum(len(d["chunks"]) for d in dataset)
        print(f"  situating {total} chunks with {MODEL_NAME}, document by document")
        print(f"  budget: ${self.budget:.2f} — the run stops if it is exceeded")
        done, results = [0], []

        def entry(doc, chunk, text):
            return {"doc_id": doc["doc_id"], "chunk_id": chunk["chunk_id"],
                    "original_index": chunk["original_index"],
                    "original_content": chunk["content"],
                    "content": f"{chunk['content']}\n\n{text}"}

        def work(doc, chunk):
            text, usage = self.situate(doc["content"], chunk["content"])
            with self.lock:
                self.record(usage)
                done[0] += 1
                print(f"  situating: {done[0]}/{total}   ${self.spend():.2f} spent"
                      f"          ", end="\r")
            return entry(doc, chunk, text)

        for doc in dataset:
            chunks = doc["chunks"]
            if not chunks:
                continue
            # The first chunk of a document goes alone, so the document is
            # written to the cache once. Sending several at once races: each
            # request can arrive before the entry exists and pay the write
            # price, which is 12x the read price and would roughly double the
            # bill for this stage.
            results.append(work(doc, chunks[0]))
            if len(chunks) > 1:
                with ThreadPoolExecutor(max_workers=threads) as ex:
                    for f in as_completed([ex.submit(work, doc, c) for c in chunks[1:]]):
                        results.append(f.result())

            if self.spend() > self.budget:
                sys.exit(f"\n\n  Stopped at ${self.spend():.2f}, over the ${self.budget:.2f} "
                         f"budget, with {done[0]} of {total} chunks done.\n"
                         f"  Nothing was cached, so nothing is half-written. Raise it with\n"
                         f"  --budget=2.00 if that is the cost you expect.")

        print(f"  situating: {total}/{total} done                              ")
        u = self.usage
        print(f"  tokens: {u['input']:,} in, {u['output']:,} out, "
              f"{u['cache_read']:,} read from cache, {u['cache_write']:,} written")
        if u["cache_read"] + u["cache_write"]:
            share = u["cache_read"] / (u["cache_read"] + u["cache_write"])
            print(f"  {share:.0%} of document tokens came from cache rather than "
                  f"being re-billed")
            if share < 0.5:
                print("  That share is low. Without caching the document is billed once")
                print("  per chunk at full input price, which is what makes this stage")
                print("  expensive. The figures below are still valid; the bill is not")
                print("  what it should be.")
        print(f"  cost of this stage: ${self.spend():.2f}")

        self.embeddings = self.embed_all([r["content"] for r in results], "chunks")
        self.metadata = results
        self.save()
        print(f"  cached to {os.path.basename(self.db_path)}")


# ------------------------------------------------------------------- scoring
def evaluate(db, queries, k):
    """The guide's scoring, unchanged: per query, what share of its golden chunks
    appear in the top k."""
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
        print(f"  Pass@{k}: {i}/{len(queries)}   ", end="\r")
    print(" " * 44, end="\r")
    return 100 * total / len(queries)


def report(stage, mine):
    pub, tol = PUBLISHED[stage], TOLERANCE[stage]
    print(f"\n{'=' * 62}")
    print(f"  {stage.upper()}  —  yours against the published guide")
    print(f"{'=' * 62}")
    print(f"  {'':9}{'published':>12}{'yours':>10}{'diff':>9}")
    ok = True
    for k in K_VALUES:
        d = mine[k] - pub[k]
        within = abs(d) <= tol
        ok = ok and within
        print(f"  Pass@{k:<5}{pub[k]:>11.2f}%{mine[k]:>9.2f}%{d:>+8.2f}"
              f"{'' if within else '   <-- outside tolerance'}")
    print(f"{'-' * 62}")
    print(f"  tolerance: {tol:.1f} points")
    if ok:
        print("\n  REPRODUCED. Your run matches the published benchmark.")
    else:
        print("\n  DID NOT MATCH. Something differs — a different embedding model,")
        print("  a partial dataset, or a changed scoring rule.")
    print(f"{'=' * 62}\n")
    return ok


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = dict(a[2:].split("=", 1) for a in sys.argv[1:]
                 if a.startswith("--") and "=" in a)
    stage = args[0] if args else "baseline"
    if stage not in PUBLISHED:
        sys.exit(f"stage must be one of: {', '.join(PUBLISHED)}\n"
                 f"  python reproduce.py baseline [--rpm=3] [--tpm=10000]")
    rpm, tpm = int(flags.get("rpm", FREE_RPM)), int(flags.get("tpm", FREE_TPM))
    budget = float(flags.get("budget", DEFAULT_BUDGET))

    dataset = json.load(open(CHUNKS, encoding="utf-8"))
    queries = load_jsonl(QUERIES)
    chunks = sum(len(d["chunks"]) for d in dataset)
    print(f"\n{chunks} chunks from {len(dataset)} documents, {len(queries)} queries")
    if (rpm, tpm) == (FREE_RPM, FREE_TPM):
        print("Voyage free tier: 3 requests and 10,000 tokens per minute.")
        print("A payment method raises these and the free token allowance still")
        print("applies — then pass --rpm=300 --tpm=1000000.\n")
    else:
        print(f"Rate limits: {rpm} requests/min, {tpm:,} tokens/min\n")

    started = time.time()
    db = (VectorDB(rpm=rpm, tpm=tpm) if stage == "baseline"
          else ContextualVectorDB(rpm=rpm, tpm=tpm, budget=budget))
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
