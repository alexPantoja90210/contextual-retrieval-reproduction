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

## 1. Grant model access

Model access is per account and per region, and it is separate from IAM. An
identity with full Bedrock permissions still gets `AccessDeniedException`
until this is done.

1. Bedrock console, region **us-east-1**.
2. **Model access** in the left navigation.
3. Enable **Titan Text Embeddings V2**. It is granted immediately.

## 2. Create a scoped IAM user

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

## 3. Configure the CLI

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
`AccessDeniedException` means step 1 or step 2 is incomplete.

## 4. Run

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

## Teardown

Nothing here runs when you are not running it. There is no infrastructure to
delete. When you are finished with the experiment, deactivate the access key
on the `titan-embed` user, or delete the user.
