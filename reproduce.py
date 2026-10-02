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

The embedder can be swapped. With --embedder=titan the same chunks, the same
queries and the same scoring run against Amazon Titan Text Embeddings V2 on
Bedrock instead of voyage-2. Nothing else changes, which is the point: the
gain contextual retrieval produces was published for one embedding model, and
whether it survives another is an open question.

  python reproduce.py baseline   --embedder=titan
  python reproduce.py contextual --embedder=titan

The second reuses the contextualized chunks the voyage-2 run already wrote to
disk, so Claude does not situate them a second time and no Anthropic key is
needed. The text is byte-identical across the two runs; only the embedder
differs.
"""
import json
import os
import pickle
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from embedders import DIMENSIONS, make_embedder

try:
    import numpy as np
except ImportError:
    sys.exit("numpy is not installed.\n  python -m pip install numpy")

MODEL_NAME = "claude-haiku-4-5"
K_VALUES = [5, 10, 20]

# Each provider states its own per-minute limits in embedders.py; --rpm and
# --tpm override them. A request is sized well under the per-minute budget, and the minute is not
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

# The guide's own figures, for voyage-2. They are the published anchor: the
# baseline reproduced here landed on all three to the hundredth, which is what
# makes this harness a calibrated instrument. A run on a different embedder is
# not measured against these — it is measured against the voyage-2 run that
# matched them.
PUBLISHED = {
    "baseline":   {5: 80.92, 10: 87.15, 20: 90.06},
    "contextual": {5: 88.12, 10: 92.34, 20: 94.29},
    # The guide's third row. Printed for orientation, never as a target: its
    # keyword side is Elasticsearch and this one is not, so the two are not
    # expected to meet. See bm25.py for what was matched and what was not.
    # Worth noticing before running anything -- the published Pass@5 FALLS here,
    # 88.12 to 86.43, which the guide's own prose does not mention.
    "hybrid":     {5: 86.43, 10: 93.21, 20: 94.99},
    # The guide's fourth row. Note what its own text says and its table does not:
    # this row is contextual plus reranking, NOT the hybrid row plus reranking.
    # The "+" column headings read as a cumulative pipeline and are not one.
    "rerank":     {5: 92.15, 10: 95.26, 20: 97.45},
}
# Embeddings are deterministic, so a baseline off by more than this is a real
# difference. Stage 2 generates text at temperature 0 — close to deterministic,
# not guaranteed identical.
TOLERANCE = {"baseline": 0.5, "contextual": 2.0}

# One embedder's worth of reranking, plus a little slack. Deliberately below
# the 1,000-call monthly trial allowance so a first run cannot spend it all.
DEFAULT_CALLS = 260

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


# Token counting moved to embedders.py, which owns it per provider: the real
# voyage-2 tokenizer where it loads, a pessimistic estimate otherwise. An
# earlier version divided characters by four, which is about right for prose
# and badly wrong for source code: it undercounted this corpus by 60%, and the
# very first request went out over the per-minute budget.


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

    def __init__(self, embedder, rpm=None, tpm=None):
        self.embedder = embedder
        rpm = rpm or embedder.default_rpm
        tpm = tpm or embedder.default_tpm
        self.embeddings, self.metadata, self.query_cache = [], [], {}
        self.limiter = RateLimiter(rpm, int(tpm * BUDGET_FRACTION))
        self.request_cap = max(400, int(tpm * REQUEST_FRACTION))
        self._tok = {}
        # The voyage-2 caches keep their original names, so an existing run is
        # not invalidated by this file gaining a second provider.
        self.db_path = os.path.join(
            HERE, "data", f"{self.name}{embedder.suffix}_vector_db.pkl")

    # -- token accounting ----------------------------------------------------
    def tokens(self, text):
        if text not in self._tok:
            self._tok[text] = self.embedder.count(text)
        return self._tok[text]

    # -- embedding -----------------------------------------------------------
    def embed_all(self, texts, label, kind="passage"):
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
        # Some providers take one text per call. The loop below still thinks in
        # batches; the provider fans them out and reassembles them in order.
        cap_texts = self.embedder.max_texts
        while i < len(texts):
            batch, used, j = [], 0, i
            while j < len(texts):
                n = self.tokens(texts[j])
                if batch and used + n > self.request_cap:
                    break
                if cap_texts and len(batch) >= cap_texts:
                    break
                batch.append(texts[j])
                used += n
                j += 1

            self.limiter.wait_for(used)
            try:
                out.extend(self.embedder.embed(batch, kind=kind))
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
        for q, v in zip(missing, self.embed_all(missing, "queries", kind="query")):
            self.query_cache[q] = v
        self.save()

    def search(self, query, k=20):
        if query not in self.query_cache:
            self.query_cache[query] = self.embed_all([query], "query",
                                                      kind="query")[0]
        sims = np.dot(self.embeddings, self.query_cache[query])
        return [{"metadata": self.metadata[i]} for i in np.argsort(sims)[::-1][:k]]


# Byte-identical to the guide, indentation included. The guide defines these
# inside ContextualVectorDB, so each line carries eight leading spaces. That
# whitespace is part of what reaches the model, so a reproduction keeps it
# instead of tidying it away.
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

    def __init__(self, embedder, rpm=None, tpm=None, budget=DEFAULT_BUDGET):
        super().__init__(embedder, rpm=rpm, tpm=tpm)
        self.budget = budget
        self.anthropic = None
        self._temp_kwargs = None
        self.usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
        self.lock = threading.Lock()
        self.temperature_ok = True

    def claude(self):
        """Built on first use, not in the constructor.

        Re-embedding chunks that Claude already situated needs no Anthropic
        key. Demanding one for a run that will never call the API is how a key
        ends up set in a shell that did not need it.
        """
        if self.anthropic is None:
            anthropic = require("anthropic")
            self.anthropic = anthropic.Anthropic(
                api_key=need("ANTHROPIC_API_KEY", "console.anthropic.com"))
            self._temp_kwargs = self._temperature_kwargs()
        return self.anthropic

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
        client = self.claude()
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
            r = client.messages.create(**body)
        except Exception as exc:
            if self.temperature_ok and "temperature" in str(exc).lower():
                with self.lock:
                    if self.temperature_ok:
                        self.temperature_ok = False
                        print("  this model does not take temperature; continuing "
                              "at the default                    ")
                r = client.messages.create(
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

        # A second embedder does not re-run the situating pass. The
        # contextualized text is already on disk from the voyage-2 run, and
        # reusing it byte for byte is what makes the comparison a comparison:
        # regenerating it would change two things at once.
        donor = os.path.join(HERE, "data", "contextual_vector_db.pkl")
        if self.embedder.suffix and os.path.exists(donor):
            with open(donor, "rb") as f:
                self.metadata = pickle.load(f)["metadata"]
            print(f"  reusing {len(self.metadata)} contextualized chunks written by")
            print("  the voyage-2 run; Claude is not called again. The text is")
            print("  identical, so the embedder is the only difference between them.")
            self.embeddings = self.embed_all(
                [m["content"] for m in self.metadata], "chunks")
            self.save()
            print(f"  cached to {os.path.basename(self.db_path)}")
            return

        total = sum(len(d["chunks"]) for d in dataset)
        print(f"  situating {total} chunks with {MODEL_NAME}, document by document")
        print(f"  budget: ${self.budget:.2f} — the run stops if it is exceeded")
        done, results = [0], []

        # KNOWN DEVIATION FROM THE GUIDE, kept deliberately.
        #
        # The guide embeds the generated context BEFORE the chunk:
        #     f"{contextualized_text}\n\n{chunk['content']}"
        # This line has always put it after. Both orders carry the same words,
        # so the difference is word order alone, which an embedding model can
        # read and a bag-of-words retriever cannot.
        #
        # probe_order.py measures it rather than guessing. On arctic-embed-l-v2:
        # Pass@5 +0.81, Pass@10 +0.60, Pass@20 -0.10 in the guide's favour at
        # the first two values of k and against it at the third. That is the
        # same order of magnitude as the gap between this harness's voyage-2
        # contextual run and the published figure (0.67 at k=5), so the order
        # is a plausible explanation for that gap -- plausible, not shown: the
        # probe ran on a different embedder.
        #
        # It stays as written so the three embedder columns remain comparable
        # to each other and to the numbers already published from them.
        # Changing it is a one-character edit and two re-runs; what it costs is
        # every contextual figure in RESULTS.md.
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
CONTEXT_SEPARATOR = "\n\n"


def contextualized_text(meta):
    """The generated context alone, recovered from the stored combined text.

    The contextual stage stores `content` as original + separator + context (see
    the deviation note above). The guide's keyword index treats the generated
    context as a field of its own, so it has to be recovered: indexing the
    combined string instead would count every word of the chunk twice and change
    the document frequencies of the whole corpus.

    Checked per chunk rather than assumed. A silently truncated field would
    still produce a ranking, and the ranking would still produce a Pass@k.
    """
    original, combined = meta["original_content"], meta["content"]
    head = original + CONTEXT_SEPARATOR
    if not combined.startswith(head):
        sys.exit(f"chunk {meta['chunk_id']} is not original + separator + context; "
                 "the assumption about the stored layout is wrong")
    return combined[len(head):]


class RerankDB:
    """Contextual retrieval with a Cohere reranker over a fixed candidate pool.

    One ranking per query, read at three values of k. The guide reranks a pool
    sized to each k and so pays three calls per query; this pays one, which is
    what fits the free trial allowance. rerank.py states what that costs in
    comparability.
    """

    def __init__(self, db, reranker, contexts):
        self.db = db
        self.reranker = reranker
        self.contexts = contexts        # chunk key -> the generated context
        self.metadata = db.metadata
        self._pool = {}

    def candidates(self, query):
        if query not in self._pool:
            self._pool[query] = self.db.search(query, k=rerank_lib.CANDIDATES)
        return self._pool[query]

    def work_item(self, query):
        """Everything one call needs, without making it.

        Returned before any call so the stage can count what a run would spend
        and refuse to begin rather than stopping halfway through a quota.
        """
        pool = self.candidates(query)
        ids = [bm25_chunk_key(r["metadata"]) for r in pool]
        key = rerank_lib.candidate_key(query, ids)
        docs = [rerank_lib.document_text(r["metadata"], self.contexts[i])
                for r, i in zip(pool, ids)]
        return key, query, docs

    def search(self, query, k=20):
        key, q, docs = self.work_item(query)
        order = self.reranker.rank(key, q, docs)
        pool = self.candidates(query)
        return [{"metadata": pool[i]["metadata"]} for i in order[:k]]


class HybridDB:
    """The guide's hybrid retrieval: the dense ranking fused with a keyword one.

    Wraps a contextual store instead of extending it, because nothing about the
    embedding path changes here -- the vectors are the ones already on disk and
    no model is called. Both recall lists are computed once per query and reused
    across the three values of k; retrieval is deterministic, so that is the same
    work the guide repeats three times.

    The keyword side reads only chunk text, so it is identical for every
    embedder. That is what makes three hybrid columns comparable: whatever
    separates them came from the dense side alone.
    """

    RECALL = 150            # the guide's num_chunks_to_recall
    SEMANTIC_WEIGHT = 0.8
    BM25_WEIGHT = 0.2

    def __init__(self, db, index):
        self.db = db
        self.index = index
        self.metadata = db.metadata
        self._recall = {}
        self.reset_counts()

    def reset_counts(self):
        self.from_semantic = self.from_keyword = self.returned = 0.0

    def search(self, query, k=20):
        if query not in self._recall:
            self._recall[query] = (self.db.search(query, k=self.RECALL),
                                   self.index.search(query, k=self.RECALL))
        semantic, keyword = self._recall[query]
        out, s, b = bm25_fuse(semantic, keyword, k,
                              self.SEMANTIC_WEIGHT, self.BM25_WEIGHT)
        self.from_semantic += s
        self.from_keyword += b
        self.returned += len(out)
        return out

    def split(self):
        """Share of returned chunks each side put there, as the guide reports it."""
        if not self.returned:
            return 0.0, 0.0
        return (100 * self.from_semantic / self.returned,
                100 * self.from_keyword / self.returned)


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
        if sys.stdout.isatty():
            # \r only overwrites on a terminal. Redirected, it would print
            # one line per query and bury the result it is counting towards.
            print(f"  Pass@{k}: {i}/{len(queries)}   ", end="\r")
    print(" " * 44, end="\r")
    return 100 * total / len(queries)


def load_reference(stage):
    """The voyage-2 result for this stage, if it has been run."""
    path = os.path.join(HERE, f"result_{stage}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return {int(k): v for k, v in json.load(f)["mine"].items()}


def report_published(stage, mine):
    """voyage-2 against the guide. A reproduction either matched or it did not."""
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


def report_swap(stage, mine, embedder, ref):
    """A different embedder against the voyage-2 run on this machine.

    There is no published figure for this, so there is nothing to pass or
    fail. What the run produces is a measurement and a difference, and the
    difference is the point: the corpus, the queries and the scoring are held
    fixed, so the columns differ by the embedding model and nothing else.
    """
    print(f"\n{'=' * 62}")
    print(f"  {stage.upper()}  —  {embedder.label} against your voyage-2 run")
    print(f"{'=' * 62}")
    if ref is None:
        print(f"  No result_{stage}.json on disk, so there is nothing to place")
        print(f"  these next to yet. Run the voyage-2 {stage} stage first.")
        print(f"{'-' * 62}")
        for k in K_VALUES:
            print(f"  Pass@{k:<5}{mine[k]:>9.2f}%")
        print(f"{'=' * 62}\n")
        return True
    # Sized to the label: "arctic-embed-l-v2" overflowed a fixed 16 and ran
    # into the column beside it.
    w = max(12, len(embedder.label) + 2)
    print(f"  {'':9}{'voyage-2':>12}{embedder.label:>{w}}{'diff':>9}")
    for k in K_VALUES:
        d = mine[k] - ref[k]
        print(f"  Pass@{k:<5}{ref[k]:>11.2f}%{mine[k]:>{w - 1}.2f}%{d:>+8.2f}")
    print(f"{'-' * 62}")
    print(f"  Same {DIMENSIONS} dimensions, same chunks, same queries, same scoring.")
    print("  The embedding model is the only difference between these columns.")
    print("  Run both stages on both embedders, then: python consolidate.py")
    print(f"{'=' * 62}\n")
    return True


def report_rerank(mine, embedder, contextual, reranker):
    """Places the rerank run next to the same embedder's contextual run."""
    width = 62
    print(f"\n{'=' * width}")
    print(f"  RERANK  —  {embedder.label}, contextual reranked by {rerank_lib.MODEL}")
    print(f"{'=' * width}")
    print(f"  {'':9}{'contextual':>13}{'reranked':>12}{'diff':>9}")
    for k in K_VALUES:
        if contextual and k in contextual:
            print(f"  Pass@{k:<4}{contextual[k]:>12.2f}%{mine[k]:>11.2f}%"
                  f"{mine[k] - contextual[k]:>+9.2f}")
        else:
            print(f"  Pass@{k:<4}{'—':>13}{mine[k]:>11.2f}%{'':>9}")
    print(f"{'-' * width}")
    print(reranker.report())
    print(f"  one ranking per query over {rerank_lib.CANDIDATES} candidates, read at")
    print("  all three values of k. The guide ranks a pool sized to each k.")
    print()
    print("  The guide published 92.15 / 95.26 / 97.45 here, on the same model")
    print("  but its own candidate pools, and on its contextual store rather")
    print("  than this one.")
    print(f"{'=' * width}")
    return True


def report_hybrid(mine, embedder, contextual, split):
    """Places the hybrid run next to the same embedder's contextual run.

    No pass or fail. The keyword engine here is not Elasticsearch, so agreement
    with the published row would be luck and disagreement would mean nothing.
    What the column answers is the question one embedder can answer about
    itself: does fusing a keyword ranking in move its own number, and where.
    """
    width = 62
    print(f"\n{'=' * width}")
    print(f"  HYBRID  —  {embedder.label}, contextual with BM25 fused in")
    print(f"{'=' * width}")
    print(f"  {'':9}{'contextual':>13}{'hybrid':>12}{'diff':>9}")
    for k in K_VALUES:
        if contextual and k in contextual:
            print(f"  Pass@{k:<4}{contextual[k]:>12.2f}%{mine[k]:>11.2f}%"
                  f"{mine[k] - contextual[k]:>+9.2f}")
        else:
            print(f"  Pass@{k:<4}{'—':>13}{mine[k]:>11.2f}%{'':>9}")
    print(f"{'-' * width}")
    sem, kw = split
    print(f"  Of the chunks returned, {sem:.1f}% came from the dense ranking")
    print(f"  and {kw:.1f}% from BM25, counting a chunk found by both as half")
    print("  to each.")
    print()
    print("  The guide published 86.43 / 93.21 / 94.99 here, on Elasticsearch.")
    print("  This keyword side is a local reimplementation, so that row is")
    print("  context for reading this one, not a target it should meet.")
    print(f"{'=' * width}")
    return True


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = dict(a[2:].split("=", 1) for a in sys.argv[1:]
                 if a.startswith("--") and "=" in a)
    stage = args[0] if args else "baseline"
    if stage not in PUBLISHED:
        sys.exit(f"stage must be one of: {', '.join(PUBLISHED)}\n"
                 f"  python reproduce.py baseline [--embedder=voyage|titan]")
    # The hybrid stage reads vectors off disk and never embeds, so it does
    # not build a provider: no key is demanded and no model is loaded.
    # Neither the hybrid nor the rerank stage embeds anything: both read vectors
    # the contextual stage already wrote. Building a provider would demand a key
    # or load a model for work that never happens.
    embedder = make_embedder(flags.get("embedder", "voyage"),
                             offline=(stage in ("hybrid", "rerank")))
    rpm = int(flags["rpm"]) if "rpm" in flags else embedder.default_rpm
    tpm = int(flags["tpm"]) if "tpm" in flags else embedder.default_tpm
    budget = float(flags.get("budget", DEFAULT_BUDGET))

    dataset = json.load(open(CHUNKS, encoding="utf-8"))
    queries = load_jsonl(QUERIES)
    chunks = sum(len(d["chunks"]) for d in dataset)
    print(f"\n{chunks} chunks from {len(dataset)} documents, {len(queries)} queries")
    print(f"embedder: {embedder.label} at {DIMENSIONS} dimensions")
    if embedder.key == "voyage" and (rpm, tpm) == (3, 10_000):
        print("Voyage free tier: 3 requests and 10,000 tokens per minute.")
        print("A payment method raises these and the free token allowance still")
        print("applies — then pass --rpm=300 --tpm=1000000.\n")
    else:
        print(f"Rate limits: {rpm} requests/min, {tpm:,} tokens/min\n")

    started = time.time()
    db = (VectorDB(embedder, rpm=rpm, tpm=tpm) if stage == "baseline"
          else ContextualVectorDB(embedder, rpm=rpm, tpm=tpm, budget=budget))
    db.load_data(dataset)

    target, contextual, split, reranker = db, None, None, None
    if stage in ("hybrid", "rerank"):
        # Neither stage may reach an embedding model. Everything they need was
        # written by the contextual stage. A missing query vector would make the
        # run start embedding, which is the surprise worth refusing.
        absent = {q["query"] for q in queries} - set(db.query_cache)
        if absent:
            sys.exit(
                f"{len(absent)} query vectors are not cached. Run\n"
                f"  python reproduce.py contextual --embedder={embedder.key}\n"
                "first; this stage never calls an embedding model.")
        path = os.path.join(HERE, f"result_contextual{embedder.suffix}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                contextual = {int(k): v for k, v in json.load(f)["mine"].items()}

    if stage == "rerank":
        import rerank as _rerank
        globals()["rerank_lib"] = _rerank
        from bm25 import chunk_key as _ck
        globals()["bm25_chunk_key"] = _ck
        contexts = {_ck(m): contextualized_text(m) for m in db.metadata}
        budget = int(flags.get("calls", DEFAULT_CALLS))
        cache = os.path.join(HERE, "data",
                             f"rerank{embedder.suffix}_cache.json")
        reranker = _rerank.Reranker(cache, budget)
        target = RerankDB(db, reranker, contexts)

        # Price the run before starting it. A trial key allows 1,000 calls a
        # month; stopping halfway through one is how the rest of the month is
        # lost, so the count is printed and the budget checked up front.
        work = [target.work_item(q["query"]) for q in queries]
        need = reranker.missing(work)
        mins = need * _rerank.SECONDS_PER_CALL / 60
        print(f"  {len(work)} queries, {len(work) - need} already cached")
        print(f"  this run would make {need} new Cohere calls "
              f"to {_rerank.MODEL}")
        print(f"  at {_rerank.TRIAL_RPM} requests/min that is about {mins:.0f} min")
        if need > budget:
            sys.exit(f"  budget is {budget}. Pass --calls={need} to allow it.")
        print(f"  budget: {budget}\n")
    elif stage == "hybrid":
        # This stage must not reach the network. Everything it needs was written
        # by the contextual stage: the chunk vectors, the query vectors and the
        # context text. If any query vector were absent the run would quietly
        # start embedding, which is exactly the surprise worth refusing.
        absent = {q["query"] for q in queries} - set(db.query_cache)
        if absent:
            sys.exit(
                f"{len(absent)} query vectors are not cached. Run\n"
                f"  python reproduce.py contextual --embedder={embedder.key}\n"
                "first; the hybrid stage never calls an embedding model.")
        require("snowballstemmer")
        from bm25 import Bm25Index, fuse as _fuse, chunk_key as _ck
        globals()["bm25_fuse"] = _fuse
        globals()["bm25_chunk_key"] = _ck
        meta = [dict(m, contextualized_content=contextualized_text(m))
                for m in db.metadata]
        print(f"  building the keyword index over {len(meta)} chunks, two fields")
        target = HybridDB(db, Bm25Index(meta))
        print("  no embedding model is called in this stage\n")
    else:
        db.embed_queries(queries)

    mine = {}
    for k in K_VALUES:
        if stage == "hybrid":
            target.reset_counts()
        mine[k] = evaluate(target, queries, k)
        print(f"  Pass@{k}: {mine[k]:.2f}%")
        if stage == "hybrid":
            split = target.split()

    ref = load_reference(stage) if embedder.suffix else None
    if stage == "rerank":
        ok = report_rerank(mine, embedder, contextual, reranker)
    elif stage == "hybrid":
        ok = report_hybrid(mine, embedder, contextual, split)
    elif embedder.key == "voyage":
        ok = report_published(stage, mine)
    else:
        ok = report_swap(stage, mine, embedder, ref)
    note = embedder.report()
    if note:
        print(note)
    print(f"  elapsed: {time.time() - started:.0f}s")
    name = f"result_{stage}{embedder.suffix}.json"
    with open(os.path.join(HERE, name), "w", encoding="utf-8") as f:
        json.dump({"stage": stage, "embedder": embedder.key, "mine": mine,
                   "published": PUBLISHED[stage] if embedder.key == "voyage" else None,
                   "reference": ref,
                   "matched": ok, "chunks": chunks, "queries": len(queries),
                   "embed_model": embedder.label, "dimensions": DIMENSIONS,
                   "model": MODEL_NAME if stage == "contextual" else None},
                  f, indent=2)
    print(f"  written: {name}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
