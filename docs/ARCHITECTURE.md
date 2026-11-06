# Architecture

## What problem this solves

Every team shipping an LLM-backed system eventually builds some version of
"run a golden set through the system, have an LLM judge grade the
answers, see if a change made things better or worse." The hard, often
skipped part isn't wiring that up — it's doing it in a way that's
*statistically honest*. LLM judges are noisy: the same answer can score
differently across repeated calls to the same judge. Golden sets are
finite: thirty hand-picked test cases are a sample of the input space, not
the whole thing. A dashboard that reports "72% → 75%, nice, ship it"
without accounting for either source of variance is reporting noise as a
finding, and this project exists specifically to not do that.

## Why a reframed project, not a third RAG system

This is the sixth of six researched SDE portfolio projects; the original
brief was "a RAG system with an LLM-judge eval harness." The author
already has two other RAG-flavored projects (a hybrid retrieval system,
and a lead-follow-up agent with RAG elements) — building a third retrieval
architecture would be redundant effort that doesn't add a new signal. The
actual differentiator the research identified — "eval-harness discipline
beats another RAG demo" — is the harness itself, so that's what this
project's effort went into. `pkg/ragexample` exists only to be a real
subject for the harness to evaluate: deliberately the simplest possible
retriever (word-overlap over an in-memory corpus), with a "good" and a
"degraded" variant built specifically so the harness's own tests can prove
the statistical machinery works on a realistic case, not just synthetic
numbers.

## Two independent sources of uncertainty, handled at two different levels

- **Judge noise, within one case.** `pkg/eval.Runner` samples the judge
  `JudgeSamples` independent times per case (default 3), not once. A
  case's `Scores` slice keeps every sample, not just the mean, so a caller
  can see directly when the judge itself disagreed with itself on a given
  case — a signal worth surfacing separately from whatever the run-level
  comparison concludes.
- **Finite golden-set size, across the whole run.** `RunResult.MeanCI` is
  a bootstrap confidence interval computed by resampling **cases** (each
  case's already-averaged score), answering "given only this many golden
  cases, how much could the true population mean plausibly differ from
  what was observed?" — a question a single point estimate can't answer at
  all.

Conflating these two would be a mistake in either direction: averaging
away judge noise without ever reporting it hides a judge-reliability
problem; failing to account for golden-set-size uncertainty at the
run level makes small samples look more conclusive than they are.

## The comparison: paired bootstrap, and why pairing specifically matters

`stats.PairedBootstrapCompare(baseline, candidate, ...)` requires the two
score slices to be the same length and in the same case order — i.e.
*paired* by case — and resamples **indices**, not the two slices
independently. The reason pairing matters: some golden cases are
inherently harder than others (ambiguous phrasing, genuinely obscure
facts). If you resampled baseline and candidate scores independently, that
per-case difficulty variance would get mixed into the apparent
baseline-vs-candidate difference, inflating the uncertainty estimate (or,
worse, occasionally producing a spurious signal from case-difficulty
variation that has nothing to do with which system actually produced
better answers for the SAME case). Resampling matched pairs and looking at
the *differences* cancels that out — the comparison is only ever measuring
"for the same question, did this system do better," aggregated with
honest uncertainty over which subset of questions you happened to test
with.

The verdict (`improvement` / `regression` / `no_significant_difference`)
is decided purely by whether the resulting confidence interval contains
zero — not by the sign of the point estimate alone. This is deliberate:
a positive-looking mean difference that the data can't actually
distinguish from zero at the chosen confidence level is, correctly,
"no significant difference," not "a small improvement." The test suite's
most important single test enforces exactly this:
`TestPairedBootstrapCompareDoesNotOverreactToNoise` constructs two runs
over a realistically small (15-case) golden set where the only difference
between baseline and candidate is independent per-call noise around an
identical true mean, and asserts the verdict is `no_significant_difference`
— the thing a naive comparison would get wrong.

## Caching: why the sample index is part of the cache key

A case's `JudgeSamples` independent judge calls exist specifically to
measure judge variance — caching them against each other (e.g. keying only
on question+answer+reference) would silently collapse K independent
samples down to one cached value repeated K times, destroying the exact
signal those samples exist to capture. `cache.Key` includes the sample
index precisely so that doesn't happen: within a single first run, all K
samples are genuinely independent judge calls; a *second, identical* run
(same question, same answer, same reference, same judge ID) hits the
cache for all K of them, since the key is otherwise unchanged — which is
the actual goal (don't re-pay for an unchanged eval), achieved without
compromising the first run's variance measurement. Changing `JudgeID`
(e.g. bumping a prompt or model version) changes every cache key for that
judge, so a deliberate prompt change is correctly never served stale
scores from before the change.

## Testing strategy

- `pkg/stats`: percentile-bootstrap correctness against known
  distributions, interval narrowing with more samples, and — the test
  that matters most — that a noise-only difference on a realistically
  small sample is correctly classified as "no significant difference."
- `pkg/judge`: the Fake judge's score-ordering (closer answers score
  higher), context-cancellation during simulated latency, bounded-noise
  clamping, and configurable failure injection.
- `pkg/cache`: round-trip correctness and that every component of the key
  (judge ID, question, answer, reference, sample index) actually changes
  the key.
- `pkg/ragexample`: the example target's retrieval and truncation behave
  as documented.
- `pkg/eval`: concurrency (every case gets evaluated), caching (a rerun
  makes zero additional judge calls; a changed JudgeID makes fresh ones),
  and — the two tests that tie everything together —
  `TestEndToEndGoodVsGoodIsNoSignificantDifference` and
  `TestEndToEndGoodVsDegradedIsRegression`, which run the full real
  pipeline (toy target → Fake judge → concurrent sampling → caching →
  paired bootstrap comparison) and check that the final verdict is
  correct, not just that intermediate numbers look plausible.

## What's deliberately out of scope (v1)

See the README — multi-rubric judging, automatic golden-set generation,
and a production-grade RAG system are all reasonable extensions this
project doesn't attempt, consistent with keeping the engineering effort on
the evaluation harness itself.
