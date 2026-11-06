# evalbench

A concurrent, cached, statistically-rigorous LLM evaluation and
regression-testing harness. The engineering problem this solves isn't "call
an LLM judge and average the scores" — it's doing that *reliably at scale*
(bounded concurrency, retry-worthy judge flakiness, not re-paying for
unchanged test cases) and *honestly* (an LLM judge is noisy, and a golden
set is a finite sample — a 3-point score improvement on 30 cases can easily
be noise, and this harness is built specifically to say so instead of
reporting it as a finding).

```
golden cases ──▶ Runner (bounded concurrency) ──▶ Target.Answer(question)
                                                          │
                                        Judge.Score() × K independent samples
                                        (cached by content hash, per sample)
                                                          │
                                              per-case mean + spread
                                                          │
                                   bootstrap CI over CASES (run-level)
                                                          │
                     PairedBootstrapCompare(baseline, candidate) ──▶ verdict:
                     improvement | regression | no_significant_difference
```

## Why this is a different project from "another RAG demo"

The author already has two other RAG-flavored portfolio projects. This one
ships a deliberately minimal example RAG target (`pkg/ragexample`) *only*
as something real to evaluate — the deliverable and the actual engineering
effort are entirely in the harness: concurrency, caching, and the
statistics. It's also deliberately distinct from this portfolio's
AI-Engineer "eval-gate CI tool" project: that one is about developer
experience and CI/PR integration; this one is about the distributed-systems
and statistics problem underneath — running potentially thousands of judge
calls reliably and cheaply, and comparing two runs without fooling yourself.

## The statistical rigor, concretely

`pkg/stats.PairedBootstrapCompare` resamples **case indices** (not scores
independently) with replacement, computes `mean(candidate - baseline)` per
resample, and takes a percentile confidence interval of that distribution.
Pairing matters: it cancels out per-case difficulty variance that an
unpaired comparison would conflate with a real system difference. The
verdict is `no_significant_difference` whenever that interval contains
zero — regardless of whether the point estimate happens to be positive or
negative. `TestPairedBootstrapCompareDoesNotOverreactToNoise` is the test
that matters most here: feed it two runs with only independent noise
separating them on a realistically small (15-case) golden set, and it
must NOT report a winner.

## Quickstart (no API keys needed)

```bash
make build
make demo
```

This runs the bundled golden set (`testdata/golden.json`) against both the
good example target and a deliberately degraded one (simulates a common
real regression: truncated retrieved context), then compares them — and
`evalbench compare` exits non-zero on a detected regression, so it's usable
directly as a CI gate.

```bash
./bin/evalbench run -cases=testdata/golden.json -target=good -out=baseline.json
./bin/evalbench run -cases=testdata/golden.json -target=degraded -out=candidate.json
./bin/evalbench compare -baseline=baseline.json -candidate=candidate.json
```

## Wiring in a real judge or target

- **Real judge**: `judge.NewHTTPJudge(name, baseURL, apiKey, model)` talks
  to any OpenAI-compatible `/chat/completions` endpoint with a fixed
  grading rubric prompt (see `pkg/judge/httpjudge.go`). The bundled
  `judge.Fake` (word-overlap + configurable injected noise/latency/failure)
  is what every test in this repo runs against instead — fast, free,
  deterministic.
- **Real target**: implement `eval.Target`'s one method,
  `Answer(ctx, question) (string, error)` — wrap your actual RAG pipeline,
  agent, or chat system.

## Caching

Every judge call is keyed by `(judgeID, question, answer, reference,
sampleIndex)` — see `pkg/cache`. The sample index is part of the key
deliberately: within one run, each of a case's K independent judge samples
gets a genuinely independent call (so run-to-run judge variance is real,
not deduplicated away), but re-running the exact same eval afterward is
free, since every one of those K calls now has a cache hit. Change
`judgeID` (e.g. bump a prompt version) and the cache is correctly bypassed
for the new judge/prompt, not silently reused.

## Testing

```bash
make test        # all unit + end-to-end tests, no external services
make test-race
```

No external services are required for any test — `pkg/judge.Fake` is the
only judge the test suite uses. The two most important tests are at the
full-pipeline level (`pkg/eval`'s `TestEndToEndGoodVsGoodIsNoSignificantDifference`
and `TestEndToEndGoodVsDegradedIsRegression`): a real (if toy) target, real
concurrency, real caching, feeding into the real statistical comparison —
not synthetic numbers handed directly to the stats package.

## What's deliberately out of scope (v1)

- **Dynamic, multi-rubric judging** (e.g. separate factuality/tone/safety
  scores per case). One scalar score + pass/fail per judge call is the
  scope here; the statistics layer is agnostic to how that score was
  produced and would work unchanged with a richer rubric.
- **Automatic golden-set generation.** Golden cases are supplied, not
  synthesized.
- **A production RAG system.** See `pkg/ragexample`'s own doc comment —
  it's intentionally minimal, by design, not an oversight.

## Project layout

```
pkg/judge/       Judge interface + Fake (deterministic, for tests) + HTTPJudge (real)
pkg/cache/       content-addressable cache for judge calls
pkg/stats/       bootstrap confidence intervals, paired regression comparison
pkg/eval/        orchestration: concurrency, judge sampling, caching wiring
pkg/ragexample/  minimal example target (good + deliberately degraded variants)
cmd/evalbench/   CLI: run, compare
docs/ARCHITECTURE.md  design rationale
```
