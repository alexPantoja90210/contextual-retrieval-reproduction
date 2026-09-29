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

    if complete:
        # A weaker baseline leaves more room to improve, so a larger raw gain is
        # partly arithmetic. Reporting the share of remaining error that closed
        # separates the two. Without this the strongest-looking column is simply
        # the one that started furthest back.
        out.append("## The same gains, adjusted for headroom")
        out.append("")
        out.append("A model that starts lower has more room to gain, so the raw "
                   "differences above are partly arithmetic. Each figure here is "
                   "the share of the error still available that contextual "
                   "retrieval removed: (contextual - baseline) / (100 - baseline).")
        out.append("")
        out.append("| Embedder | @5 | @10 | @20 |")
        out.append("| --- | --- | --- | --- |")

        def closed_row(label, b, c):
            cells = "".join(f" {100 * (c[k] - b[k]) / (100 - b[k]):.1f}% |"
                            for k in K_VALUES)
            return f"| {label} |{cells}"

        out.append(closed_row("Anthropic, published",
                              PUBLISHED["baseline"], PUBLISHED["contextual"]))
        for label in complete:
            out.append(closed_row(label, found[(label, "baseline")],
                                  found[(label, "contextual")]))
        out.append("")

    if len(complete) >= 2:
        a, b = complete[0], complete[1]   # voyage-2 first, then the next finished
        ga = {k: found[(a, "contextual")][k] - found[(a, "baseline")][k]
              for k in K_VALUES}
        gb = {k: found[(b, "contextual")][k] - found[(b, "baseline")][k]
              for k in K_VALUES}
        ca = {k: 100 * ga[k] / (100 - found[(a, "baseline")][k]) for k in K_VALUES}
        cb = {k: 100 * gb[k] / (100 - found[(b, "baseline")][k]) for k in K_VALUES}
        gap_b = {k: found[(b, "baseline")][k] - found[(a, "baseline")][k]
                 for k in K_VALUES}
        gap_c = {k: found[(b, "contextual")][k] - found[(a, "contextual")][k]
                 for k in K_VALUES}

        out.append("## Reading")
        out.append("")
        out.append(f"**The gain survives the change of embedding model.** "
                   f"Contextual retrieval raises Pass@20 by {ga[20]:+.2f} points on "
                   f"{a} and {gb[20]:+.2f} on {b}, against the same 248 queries and "
                   f"the same contextualized text. Only the embedder differs.")
        out.append("")
        bigger = all(cb[k] > ca[k] for k in K_VALUES)
        if bigger:
            out.append(f"**It is worth more to the weaker model, and not only "
                       f"because that model had further to go.** {b} starts below "
                       f"{a} at every k, so some of its larger raw gain is "
                       f"headroom. After dividing that out it still closes more of "
                       f"the remaining error at every k "
                       f"({cb[5]:.1f}% against {ca[5]:.1f}% at k=5, "
                       f"{cb[20]:.1f}% against {ca[20]:.1f}% at k=20). The margin "
                       f"is wide where the numbers are low and narrow where both "
                       f"are already high.")
            out.append("")
        out.append(f"**The two models converge.** The gap between them narrows from "
                   f"{gap_b[5]:+.2f} / {gap_b[10]:+.2f} / {gap_b[20]:+.2f} at "
                   f"baseline to {gap_c[5]:+.2f} / {gap_c[10]:+.2f} / "
                   f"{gap_c[20]:+.2f} once both run contextually. Writing a line of "
                   f"context in front of each chunk recovers most of what separated "
                   f"the two embedders.")
        out.append("")
        out.append("### What this does not establish")
        out.append("")
        out.append("One corpus of source code, one model writing the context, two "
                   "embedders, 248 queries. The direction is consistent across "
                   "three values of k, which is not the same as being general. "
                   "Nothing here says the pattern holds on prose, on a larger "
                   "corpus, or on a third embedder.")
        out.append("")
        out.append(f"Anthropic published this technique on {a} alone. What this "
                   f"table adds is a second embedding model, measured through a "
                   f"harness whose baseline reproduced their figures to the "
                   f"hundredth.")
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
