# Running the benchmark on Amazon Bedrock

This adds a second embedding model to the reproduction. The corpus, the
queries, the contextualized text and the scoring are unchanged; only the
embedder differs. Everything here is done once.

## What this costs

Titan Text Embeddings V2 is $0.02 per million input tokens. The two stages
embed 737 chunks and 248 queries twice over — well under a million tokens in
total, so the whole thing is a few cents.

The contextual stage does **not** call Claude again. It reuses the
contextualized chunks the voyage-2 run already wrote to
`data/contextual_vector_db.pkl`, which is both cheaper and more correct: the
text being embedded is byte-identical across the two runs, so the embedding
model is the only variable.

## Model access: nothing to do

Bedrock's **Model access** page has been retired. Serverless foundation models
are enabled automatically across commercial regions the first time an account
invokes them, so Titan Embed v2 needs no activation step — IAM permission is
now the only gate.

Two exceptions, neither of which applies here: Anthropic models may ask a
first-time user for use-case details, and Marketplace-served models still need
one invocation by a user with Marketplace permissions. Titan is a first-party
serverless model and is covered by neither.

If an `AccessDeniedException` survives a correct IAM policy, the remaining
causes are an account-level Service Control Policy or a permissions boundary,
not a missing activation.

## 1. Create a scoped IAM user

Never root-account access keys. The policy in `iam-policy-titan-embed.json`
allows one action on one model in one region and nothing else.

1. IAM console > **Policies** > **Create policy** > JSON tab.
2. Paste `iam-policy-titan-embed.json`. Name it `titan-embed-invoke`.
3. IAM console > **Users** > **Create user**, name it `titan-embed`.
   Do not give it console access.
4. Attach `titan-embed-invoke` directly to the user.
5. On the user's **Security credentials** tab, **Create access key** >
   *Application running outside AWS*.

Copy the key and secret into the CLI in the next step. They are shown once.

## 2. Configure the CLI

```powershell
aws configure --profile titan-embed
# AWS Access Key ID:     [paste]
# AWS Secret Access Key: [paste]
# Default region name:   us-east-1
# Default output format: json

$env:AWS_PROFILE = "titan-embed"
python -m pip install boto3
```

Confirm it works before running anything long:

```powershell
aws bedrock-runtime invoke-model `
  --model-id amazon.titan-embed-text-v2:0 `
  --body (echo '{"inputText":"hello","dimensions":1024,"normalize":true}' | base64) `
  --cli-binary-format raw-in-base64-out `
  out.json
```

A `ValidationException` here is a formatting problem with the command. An
`AccessDeniedException` means the IAM policy is not attached, names a different
region, or is overridden by an account-level policy.

## 3. Run

```powershell
python test_embedders.py                            # offline, no AWS needed
python reproduce.py baseline   --embedder=titan
python reproduce.py contextual --embedder=titan
python consolidate.py
```

Each stage caches its vectors to `data/`, so a second run re-scores in
seconds and costs nothing.

## What the result means

There is no published Titan figure for this benchmark to match, so these runs
are not a reproduction and the script does not grade them. What they produce
is a measurement: contextual retrieval's gain on one embedding model, placed
next to the same gain on another, with every other input held fixed.

The reproduction itself — the voyage-2 baseline landing on Anthropic's
published Pass@5/10/20 — is what makes that measurement worth reading. It is
the evidence the harness reads true.

## Error 002, and what was eliminated to call it account-level

```
ValidationException: Error 002: Access to Bedrock models is not allowed
for this account
```

This is not an IAM problem and no policy will fix it. It is an account-level
restriction: the request was signed and authorized, and the service refused it
afterwards. A missing permission produces `AccessDeniedException` instead, and
the two strings are distinct for that reason.

### What was tried here

This section is written from a real account that hit it, rather than from the
documentation, because the documentation does not cover this code. Everything
below was eliminated before concluding anything.

| Checked | Result |
| --- | --- |
| Scoped IAM user with `bedrock:InvokeModel` on the Titan embedding model | Error 002 |
| The same call from an identity with AdministratorAccess | Error 002 |
| `us-east-1` | Error 002 |
| `us-west-2` | Error 002 |
| From the Bedrock console playground, not the SDK | Error 002 |
| Models other than Titan — Nova Lite, Nova 2 Lite | Error 002 |
| The **Model access** page, to request or enable a model | Retired; there is no longer anything to enable |

So it is not the policy, not the key, not the region, not the SDK, not the
model, and not an unrequested model grant. Listing and selecting models works.
Only invocation is refused, identically, everywhere.

That combination leaves the account itself, which is what the message says and
what nothing configurable can change.

### What the error is not

It is **not in Bedrock's published list of API error codes**. Searching for it
returns forum posts rather than documentation, and the causes reported there —
accounts with no payment history, organization-level Service Control Policies —
are plausible explanations that this account could neither confirm nor rule out
from the outside. They are listed as leads, not as the answer.

### The route, and its honest odds

A support case under **Account and billing**, which needs no paid support plan.
Case `179074483400988` was opened for this account and was still unassigned and
unanswered when this file was written (2026-10-02). That is the whole route;
there is no API call, console toggle or policy that routes around it.

### It does not block the benchmark

`--embedder=arctic` runs every stage locally with no account anywhere, and
`--embedder=azure` runs on an Azure AI Foundry deployment. The Titan path in
`embedders.py` is written and covered by `test_embedders.py`, so if the account
is ever unblocked the column is four commands away and nothing needs changing:

```powershell
python reproduce.py baseline   --embedder=titan
python reproduce.py contextual --embedder=titan
python reproduce.py hybrid     --embedder=titan
python reproduce.py rerank     --embedder=titan
```

Worth knowing before spending effort on that: the three embedders already
measured land on the same Pass@k to every decimal once reranking is applied
(see RESULTS.md). A fourth is unlikely to change the conclusion. It would
confirm it on one more model, which is worth something and is not worth
waiting for.

## Teardown

Nothing here runs when you are not running it. There is no infrastructure to
delete. When you are finished with the experiment, deactivate the access key
on the `titan-embed` user, or delete the user.
