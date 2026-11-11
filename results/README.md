# Results

Generated experiment output belongs here locally and is ignored by Git.

This directory contains two different classes of evidence.

## Structural and smoke runs

Structural runs prove that the configured workload executes and help explain where time is spent. They may use synthetic input and/or random weights, so they are **not performance baselines**.

Typical artifacts:

- `centerpoint.json` — environment, input/voxel counts, latency, stage timing, sparse-backbone timing, and memory usage
- `operators.txt` — operator summary sorted by CUDA time
- `trace.json` — execution timeline for Chrome/Perfetto or similar tools

A structural run does not need `prediction.pt` when the model uses random weights and the predictions have no correctness meaning.

## Reference baseline and optimization runs

A reference baseline uses representative input and a compatible trained model. Optimization runs must preserve enough evidence to compare performance **and** verify that behavior remains acceptable.

Preserve:

- `timing.json` — latency distribution and measurement metadata
- `operators.txt` — operator-level profile
- `trace.json` — execution timeline
- `prediction.pt` — reference or candidate output used for correctness comparison

For optimization experiments, keep each run in a separate directory so the before/after evidence remains reproducible.

Example:

```text
results/
    smoke/
        centerpoint.json
        operators.txt
        trace.json

    baseline/
        timing.json
        operators.txt
        trace.json
        prediction.pt

    optimization-001/
        timing.json
        operators.txt
        trace.json
        prediction.pt
```

## What belongs in Git

Raw result artifacts are working evidence and should normally remain local.

Promote only small, reviewed summaries into version control after checking that they contain no restricted dataset content and nothing that cannot be shared publicly. Durable conclusions belong in places such as:

- `CHANGELOG.md`
- milestone documents under `docs/`
- other small reviewed summaries that describe the experiment without embedding raw restricted data

The intent is to keep **raw experimental evidence replaceable** while making **important conclusions and reference points durable and versioned**.
