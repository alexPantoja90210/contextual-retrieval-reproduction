# Running the benchmark on Azure AI Foundry

This adds a fourth embedding model to the comparison: OpenAI's
`text-embedding-3-large`, served from an Azure AI Foundry deployment. The
corpus, the queries, the contextualized text and the scoring are unchanged.

## What this costs

Under a Standard deployment, Azure bills per token, not per hour. The two
stages embed 737 chunks and 248 queries twice over — under a million tokens in
total, which is cents.

The contextual stage does **not** call Claude again. It reuses the
contextualized chunks already on disk from the voyage-2 run, so the text being
embedded is byte-identical across every provider and the embedder is the only
variable.

## 1. Deploy the model

Azure AI Foundry no longer asks for project settings; opening it creates a
project with defaults. Note the region it picked — model availability varies by
region, and the catalogue's "available in my project" filter is the honest test.

1. Foundry > Model catalog > `text-embedding-3-large` (publisher **OpenAI**).
2. **Deploy**, deployment name `text-embedding-3-large`.
3. Deployment type: **Standard** or **Global Standard**. Both bill per API
   call. **Not** Provisioned or PTU, which reserve capacity and bill by the
   hour whether or not anything runs.
4. Leave the tokens-per-minute slider at its default rather than taking the
   region's whole quota. Quota is shared, and a later chat or reranker
   deployment needs some.

Two notes on what else is in that catalogue. The Hugging Face entries deploy to
managed compute, billed hourly — the opposite of what this run wants. And
Global Standard may process data outside the resource's Azure geography, while
Standard does not; for a public code corpus that is immaterial, but it is the
screen where data residency is actually decided.

## 2. Configure the environment

The endpoint is on the Foundry project overview page. Paste it as the portal
shows it; the code appends the API path.

```powershell
$env:AZURE_OPENAI_ENDPOINT = "https://<resource>.services.ai.azure.com"
$env:AZURE_OPENAI_API_KEY  = "..."
python -m pip install openai tiktoken
```

`tiktoken` is optional. With it, token counts are exact, which matters because
the deployment enforces a real tokens-per-minute limit; without it the run falls
back to a pessimistic estimate and the batching loop shrinks requests on refusal
as it does for every other provider.

Confirm the deployment answers before running anything long:

```powershell
python -c "import os,openai; c=openai.OpenAI(api_key=os.environ['AZURE_OPENAI_API_KEY'], base_url=os.environ['AZURE_OPENAI_ENDPOINT'].rstrip('/')+'/openai/v1/'); r=c.embeddings.create(model='text-embedding-3-large', input='hello', dimensions=1024); print('dims', len(r.data[0].embedding))"
```

It should print `dims 1024`.

## 3. Run

```powershell
python test_embedders.py                          # offline, no Azure needed
python reproduce.py baseline   --embedder=azure
python reproduce.py contextual --embedder=azure
python consolidate.py
```

Each stage caches its vectors to `data/`, so a second run re-scores in seconds
and costs nothing.

## Why the width is pinned

`text-embedding-3-large` is 3072 dimensions natively. Every other column in the
table is 1024, so the run requests 1024 through the API's `dimensions`
parameter and then **verifies the first vector**.

If a deployment ignores that parameter and returns the native width, the code
truncates and renormalizes locally and says so. Truncation is valid for this
model family, whose vectors remain usable as prefixes; renormalization is not
optional, because the scoring is a dot product and an unnormalized vector would
let length count as similarity. A vector narrower than 1024 stops the run, since
it cannot be widened.

## Cost guardrail

Set a budget before the first run. Cost Management > Budgets, with alerts on
actual spend and, more usefully, one on **forecast** — actual-spend alerts
arrive after the money is gone.

A budget notifies; it does not stop anything. Nothing here runs when it is not
being run, so there is no idle cost, but a provisioned deployment created by
mistake would bill continuously. Deleting the deployment when the experiment is
finished is the actual control.
