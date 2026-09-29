#!/usr/bin/env python3
"""Embedding providers for the reproduction, behind one interface.

The reproduction answered a published question: does contextual retrieval
improve Pass@k on this corpus? It did, and the baseline landed on the
published figures to the hundredth, which is what makes this harness a
calibrated instrument rather than a script.

The open question is a different one, and nobody has published an answer:
does that gain survive a change of embedding model? Anthropic measured it
with voyage-2. This module exists so that question can be asked by changing
exactly one thing.

That constraint shapes the interface. A provider exposes a token count, a
batch size the caller may fill, and an embed call that returns vectors in
the order it was given them. Nothing else. The chunking, the scoring, the
queries and the golden chunks are identical across providers, so a
difference in the result is a difference in the embedder.

Both providers are configured to 1024 dimensions. A comparison between
vectors of different widths would measure the width as much as the model.
"""
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

# Both providers, so the two runs are comparable. voyage-2 is 1024 natively;
# Titan v2 offers 256, 512 and 1024 and defaults to 1024, but it is set
# explicitly here rather than assumed.
DIMENSIONS = 1024


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


class Embedder:
    """What the benchmark needs from an embedding provider, and no more."""

    key = None            # what --embedder takes
    model = None          # the provider's own model identifier
    label = None          # what appears in the results table
    max_texts = None      # texts per call; None means only the token budget caps it
    default_rpm = None
    default_tpm = None
    suffix = ""           # appended to cache and result filenames

    def count(self, text):
        """Tokens in a text, for the rate limiter. An estimate is acceptable;
        the batching loop recovers from an undercount by shrinking."""
        raise NotImplementedError

    def embed(self, texts, kind="passage"):
        """Vectors for texts, in the same order. Raises on unrecoverable error.

        `kind` is "passage" or "query". Some models want a prefix on queries
        and not on passages; providers that do not distinguish ignore it.
        """
        raise NotImplementedError

    def report(self):
        """One line about what this provider actually did, or None."""
        return None


# --------------------------------------------------------------------- voyage
class VoyageEmbedder(Embedder):
    """The guide's own provider, unchanged. This is the calibrated path."""

    key = "voyage"
    model = "voyage-2"
    label = "voyage-2"
    max_texts = None
    # Voyage's free tier, before a payment method is added, allows 3 requests
    # and 10,000 tokens per minute.
    default_rpm, default_tpm = 3, 10_000
    suffix = ""           # keeps the existing cache and result filenames

    def __init__(self):
        voyageai = require("voyageai")
        self.client = voyageai.Client(api_key=need("VOYAGE_API_KEY", "voyageai.com"))
        self._tokenizer, self._tried = None, False

    def tokenizer(self):
        if not self._tried:
            self._tried = True
            try:
                from tokenizers import Tokenizer
                self._tokenizer = Tokenizer.from_pretrained("voyageai/voyage-2")
                print("  counting tokens with the voyage-2 tokenizer")
            except Exception:
                print("  voyage-2 tokenizer unavailable; estimating at 2.5 chars/token")
        return self._tokenizer

    def count(self, text):
        t = self.tokenizer()
        if t is not None:
            try:
                return max(1, len(t.encode(text).ids))
            except Exception:
                pass
        return max(1, int(len(text) / 2.5))

    def embed(self, texts, kind="passage"):
        # Voyage takes an input_type that distinguishes queries from documents.
        # The guide does not pass it, and this column has to stay the
        # reproduction, so it is not passed here either. Using it would
        # probably score better and would no longer be the published method.
        return self.client.embed(texts, model=self.model).embeddings


# -------------------------------------------------------------------- bedrock
# Retried: the service is busy or the model is warming. Waiting helps.
RETRYABLE = {"ThrottlingException", "TooManyRequestsException",
             "ServiceUnavailableException", "ModelNotReadyException",
             "InternalServerException"}
MAX_TRIES = 6


class BedrockEmbedder(Embedder):
    """Amazon Titan Text Embeddings V2, through Bedrock's InvokeModel.

    Titan takes one text per call — there is no batch endpoint — so a batch
    handed down by the caller is fanned out across a few threads here and
    reassembled in order. The caller's rate limiter still sees one batch, so
    the accounting above this class is unchanged.
    """

    key = "titan"
    model = "amazon.titan-embed-text-v2:0"
    label = "titan-embed-v2"
    max_texts = 32
    # Bedrock's on-demand quotas for Titan embeddings are far above the free
    # Voyage tier. These are deliberately conservative: the run is a few
    # minutes either way, and a throttle costs more than a wait.
    default_rpm, default_tpm = 600, 300_000
    suffix = "_titan"

    def __init__(self, region=None, threads=4):
        boto3 = require("boto3")
        self.region = region or os.getenv("AWS_REGION") or "us-east-1"
        session = boto3.Session(region_name=self.region)
        if session.get_credentials() is None:
            sys.exit(
                "No AWS credentials found.\n"
                "  Configure a scoped IAM user (see iam-policy-titan-embed.json):\n"
                "    aws configure --profile titan-embed\n"
                '    $env:AWS_PROFILE = "titan-embed"\n'
                "  Never root-account access keys.")
        self.client = session.client("bedrock-runtime")
        self.threads = threads
        self.lock = threading.Lock()
        self.real_tokens = 0     # what Titan reported, summed
        self.est_tokens = 0      # what we guessed, summed
        self.retries = 0

    def count(self, text):
        # Titan's tokenizer is not published. 2.5 chars/token is the same
        # pessimistic estimate the Voyage path falls back to, and the run
        # compares it against Titan's own reported counts at the end.
        n = max(1, int(len(text) / 2.5))
        with self.lock:
            self.est_tokens += n
        return n

    def _one(self, text):
        body = json.dumps({"inputText": text,
                           "dimensions": DIMENSIONS,
                           "normalize": True})
        for attempt in range(MAX_TRIES):
            try:
                r = self.client.invoke_model(modelId=self.model, body=body)
                payload = json.loads(r["body"].read())
            except Exception as exc:
                code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
                if code == "AccessDeniedException":
                    sys.exit(
                        f"\n  Bedrock refused the call: {code}.\n"
                        f"  Bedrock's Model access page has been retired — serverless\n"
                        f"  foundation models enable themselves on first invocation, so\n"
                        f"  there is no activation step to have missed. What is left:\n"
                        f"    1. The IAM identity lacks bedrock:InvokeModel on\n"
                        f"       {self.model}.\n"
                        f"    2. The policy names a different region than {self.region}.\n"
                        f"    3. A Service Control Policy or permissions boundary on the\n"
                        f"       account overrides the grant.\n"
                        f"  See BEDROCK-SETUP.md.")
                if code == "ValidationException" and "too long" in str(exc).lower():
                    sys.exit(
                        f"\n  A chunk of about {self.count(text):,} estimated tokens "
                        f"exceeds Titan's\n  input limit. It cannot be split without "
                        f"changing the corpus, which\n  would make the comparison "
                        f"against the voyage-2 run meaningless.\n  Stopping rather "
                        f"than silently truncating.")
                if code not in RETRYABLE or attempt == MAX_TRIES - 1:
                    raise
                wait = min(30.0, 1.5 ** attempt)
                with self.lock:
                    self.retries += 1
                time.sleep(wait)
                continue
            with self.lock:
                self.real_tokens += payload.get("inputTextTokenCount", 0)
            return payload["embedding"]
        raise RuntimeError("unreachable")

    def embed(self, texts, kind="passage"):
        # Titan v2 has no query/passage distinction in its request body.
        if len(texts) == 1:
            return [self._one(texts[0])]
        with ThreadPoolExecutor(max_workers=self.threads) as ex:
            # map preserves input order, which the scoring depends on: a
            # vector reassembled against the wrong chunk would still score,
            # just wrongly, and nothing downstream would notice.
            return list(ex.map(self._one, texts))

    def report(self):
        if not self.real_tokens:
            return None
        drift = 100 * (self.est_tokens - self.real_tokens) / self.real_tokens
        line = (f"  Titan reported {self.real_tokens:,} input tokens; "
                f"the estimate was {self.est_tokens:,} ({drift:+.0f}%)")
        if self.retries:
            line += f"\n  {self.retries} calls were throttled and retried"
        return line




# ----------------------------------------------------------------------- local
# Chosen over bge-large-en-v1.5, which is the obvious open-source model at this
# width, because bge caps at 512 tokens. On this corpus that cap truncates 1 of
# 737 baseline chunks and 67 of 737 contextual ones: the situating line Claude
# prepends lifts the median chunk from 261 to 399 tokens, so a 512-token model
# would cut the contextual run 67 times more often than the baseline. A small
# measured gain would then be unreadable — technique failing to transfer, or
# model never seeing the context? The bias runs against the technique, which is
# the worst direction for it to run. 8192 tokens removes the question.
#
# gte-large-en-v1.5 is the other 1024-by-8192 option and needs
# trust_remote_code=True. This one does not, so no code from a model repository
# is executed here.
ARCTIC_MAX_TOKENS = 8192


class LocalEmbedder(Embedder):
    """Snowflake Arctic Embed L v2.0, on this machine.

    No API, no key, no account, no per-token cost. Anyone who clones the
    repository can run this column of the table, which is not true of any
    hosted provider.
    """

    key = "arctic"
    model = "Snowflake/snowflake-arctic-embed-l-v2.0"
    label = "arctic-embed-l-v2"
    max_texts = 16
    # Nothing is rate limited: the model runs here. The limiter stays in the
    # path rather than being special-cased out, so one code path serves every
    # provider; these numbers simply never bind.
    default_rpm, default_tpm = 10 ** 6, 10 ** 9
    suffix = "_arctic"

    def __init__(self):
        st = require("sentence_transformers", "sentence-transformers")
        print(f"  loading {self.model} (first run downloads about 2.3 GB)")
        self.st = st.SentenceTransformer(self.model)
        self.truncated = 0
        self.longest = 0

        # sentence-transformers reports the length it will actually enforce,
        # which can sit below what the architecture supports. Asking for the
        # full window and then reading the value back is the only way to know
        # what this install will do, rather than what the model card says.
        try:
            self.st.max_seq_length = ARCTIC_MAX_TOKENS
        except Exception:
            pass
        self.limit = int(getattr(self.st, "max_seq_length", 0) or 0)
        print(f"  maximum sequence length in effect: {self.limit or 'unknown'} tokens")

        # sentence-transformers renamed this; support both rather than warn.
        get_dim = (getattr(self.st, "get_embedding_dimension", None)
                   or self.st.get_sentence_embedding_dimension)
        dim = get_dim()
        if dim != DIMENSIONS:
            sys.exit(f"\n  {self.model} returns {dim} dimensions, not {DIMENSIONS}.\n"
                     f"  The comparison holds width fixed, so a different width\n"
                     f"  would measure the width alongside the model.")

    def count(self, text):
        # The model's own tokenizer, so this is exact rather than estimated.
        n = len(self.st.tokenizer.encode(text, add_special_tokens=False))
        self.longest = max(self.longest, n)
        if self.limit and n > self.limit:
            self.truncated += 1
        return max(1, n)

    def embed(self, texts, kind="passage"):
        # Arctic asks for a prefix on queries only. Applying it to passages, or
        # omitting it from queries, quietly costs retrieval accuracy and looks
        # exactly like the model being worse.
        kwargs = {"prompt_name": "query"} if kind == "query" else {}
        out = self.st.encode(list(texts), normalize_embeddings=True,
                             show_progress_bar=False, **kwargs)
        return [v.tolist() for v in out]

    def report(self):
        line = f"  longest text seen: {self.longest:,} tokens"
        if self.limit:
            line += f", limit {self.limit:,}"
        if self.truncated:
            line += (f"\n  {self.truncated} texts exceeded the limit and were "
                     f"truncated.\n"
                     f"  Truncation is not symmetric across stages — contextual "
                     f"chunks are longer\n"
                     f"  than baseline ones — so a gain measured under it "
                     f"understates the technique.")
        else:
            line += "\n  nothing was truncated"
        return line


PROVIDERS = {c.key: c for c in (VoyageEmbedder, BedrockEmbedder, LocalEmbedder)}


def make_embedder(key):
    if key not in PROVIDERS:
        sys.exit(f"--embedder must be one of: {', '.join(PROVIDERS)}")
    return PROVIDERS[key]()
