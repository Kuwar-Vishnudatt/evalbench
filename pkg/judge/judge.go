// Package judge defines how a candidate answer gets scored against a
// reference, and provides two implementations: a fully deterministic Fake
// (so the rest of the harness — concurrency, caching, statistics — can be
// tested without live LLM calls or API costs) and a real HTTP-based judge
// for actual use. This mirrors the same Backend/Fake split used in the
// llmgate project: the thing being judged (or routed, there) needs to be
// swappable and the swappable-for-testing variant needs to be genuinely
// deterministic, not just "usually consistent."
package judge

import "context"

// Verdict is one judge's assessment of one answer.
type Verdict struct {
	// Score in [0,1]: 1 means the answer fully satisfies the reference: 1
	// means fully correct, 0 means fully wrong/irrelevant.
	Score float64
	// Pass is the judge's own binary call — not simply Score >= 0.5,
	// because a judge may apply its own threshold or weigh factors (e.g.
	// factual correctness) more strictly than a continuous score implies.
	Pass      bool
	Rationale string
}

// Judge scores a candidate Answer against a Reference for Question. A
// single Judge implementation is expected to be noisy across repeated
// calls for the same input — that's real LLM-judge behavior, not a bug —
// which is exactly why the harness samples a judge multiple times per case
// rather than trusting one call (see pkg/eval).
type Judge interface {
	Name() string
	Score(ctx context.Context, question, answer, reference string) (Verdict, error)
}
