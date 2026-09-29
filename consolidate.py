#!/usr/bin/env python3
"""Puts every run in one table and writes RESULTS.md.

Two things are worth separating and are easy to conflate.

The reproduction is a claim about this harness: the voyage-2 baseline landed
on Anthropic's published figures, so the instrument reads true. That question
is settled and does not reopen.

The embedder swap is a claim about the technique: contextual retrieval's gain
was published for one embedding model, and this table shows whether it
survives another. The comparison that answers it is not titan against
Anthropic's numbers — it is titan-contextual against titan-baseline, placed
next to voyage-contextual against voyage-baseline. Comparing across both axes
at once would move two variables and measure neither.

  python consolidate.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
K_VALUES = [5, 10, 20]

PUBLISHED = {
    "baseline":   {5: 80.92, 10: 87.15, 20: 90.06},
    "contextual": {5: 88.12, 10: 92.34, 20: 94.29},
}

# label, filename suffix, the command that produces it
RUNS = [
    ("voyage-2", "", "python reproduce.py {stage}"),
    ("arctic-embed-l-v2", "_arctic", "python reproduce.py {stage} --embedder=arctic"),
    ("titan-embed-v2", "_titan", "python reproduce.py {stage} --embedder=titan"),
]
STAGES = ["baseline", "contextual"]


def load(stage, suffix):
    path = os.path.join(HERE, f"result_{stage}{suffix}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return {int(k): v for k, v in d["mine"].items()}


def row(label, stage, scores):
    cells = "".join(f" {scores[k]:.2f} |" for k in K_VALUES)
    return f"| {label} | {stage} |{cells}"


def gain_row(label, base, ctx):
    cells = "".join(f" {ctx[k] - base[k]:+.2f} |" for k in K_VALUES)
    return f"| {label} |{cells}"


def main():
    found, missing = {}, []
    for label, suffix, cmd in RUNS:
        for stage in STAGES:
            scores = load(stage, suffix)
            if scores:
                found[(label, stage)] = scores
            else:
                missing.append((label, stage, cmd.format(stage=stage)))

    if not found:
        sys.exit("No result files on disk. Run a stage first:\n"
                 "  python reproduce.py baseline")

    out = []
    out.append("# Results")
    out.append("")
    out.append("Pass@k on 737 chunks from 9 codebases, 248 queries, each with a")
    out.append("known golden chunk. Retrieval is dot-product over normalized")
    out.append("1024-dimension vectors, top-k. The scoring is the guide's own.")
    out.append("")
    out.append("## Every run")
    out.append("")
    out.append("| Embedder | Chunking | Pass@5 | Pass@10 | Pass@20 |")
    out.append("| --- | --- | --- | --- | --- |")
    for stage in STAGES:
        out.append(row("Anthropic, published", stage, PUBLISHED[stage]))
    for label, suffix, _ in RUNS:
        for stage in STAGES:
            if (label, stage) in found:
                out.append(row(label, stage, found[(label, stage)]))
    out.append("")

    complete = [label for label, _, _ in RUNS
                if (label, "baseline") in found and (label, "contextual") in found]
    if complete:
        out.append("## What contextual retrieval adds, per embedder")
        out.append("")
        out.append("| Embedder | @5 | @10 | @20 |")
        out.append("| --- | --- | --- | --- |")
        out.append(gain_row("Anthropic, published",
                            PUBLISHED["baseline"], PUBLISHED["contextual"]))
        for label in complete:
            out.append(gain_row(label, found[(label, "baseline")],
                                found[(label, "contextual")]))
        out.append("")

    if len(complete) >= 2:
        a, b = complete[0], complete[1]   # voyage-2 first, then the next finished
        ga = {k: found[(a, "contextual")][k] - found[(a, "baseline")][k]
              for k in K_VALUES}
        gb = {k: found[(b, "contextual")][k] - found[(b, "baseline")][k]
              for k in K_VALUES}
        out.append("## Reading")
        out.append("")
        out.append(f"Contextual retrieval raises Pass@20 by {ga[20]:+.2f} points on "
                   f"{a} and {gb[20]:+.2f} on {b}, on the same 248 queries against "
                   f"the same contextualized text. The two runs share every input "
                   f"except the embedding model.")
        out.append("")
        spread = max(abs(ga[k] - gb[k]) for k in K_VALUES)
        out.append(f"The largest difference between the two gains, across the three "
                   f"values of k, is {spread:.2f} points.")
        out.append("")
        out.append("Anthropic published this technique measured on voyage-2 alone. "
                   "What the table above adds is a second embedding model, run "
                   "through a harness whose baseline reproduced their figures.")
        out.append("")

    if missing:
        out.append("## Not yet run")
        out.append("")
        for label, stage, cmd in missing:
            out.append(f"- {label}, {stage} — `{cmd}`")
        out.append("")

    text = "\n".join(out)
    print()
    print(text)
    path = os.path.join(HERE, "RESULTS.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"written: {os.path.basename(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
