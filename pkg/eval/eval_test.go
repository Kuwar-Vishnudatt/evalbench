package eval

import (
	"context"
	"math/rand"
	"path/filepath"
	"sync/atomic"
	"testing"

	"github.com/Kuwar-Vishnudatt/evalbench/pkg/cache"
	"github.com/Kuwar-Vishnudatt/evalbench/pkg/judge"
	"github.com/Kuwar-Vishnudatt/evalbench/pkg/ragexample"
	"github.com/Kuwar-Vishnudatt/evalbench/pkg/stats"
)

func testCorpus() ragexample.Corpus {
	return ragexample.Corpus{
		{ID: "d1", Text: "The capital of France is Paris, a city known for the Eiffel Tower and the Louvre museum."},
		{ID: "d2", Text: "Mount Everest is the tallest mountain above sea level, located in the Himalayas on the border of Nepal and Tibet."},
		{ID: "d3", Text: "Photosynthesis is the process by which plants convert sunlight into chemical energy stored in glucose."},
	}
}

func goldenCases() []GoldenCase {
	return []GoldenCase{
		{ID: "c1", Question: "What is the capital of France", Reference: "Paris is the capital of France, famous for the Eiffel Tower."},
		{ID: "c2", Question: "What is the tallest mountain", Reference: "Mount Everest is the tallest mountain, located in the Himalayas."},
		{ID: "c3", Question: "What is photosynthesis", Reference: "Photosynthesis converts sunlight into chemical energy in plants."},
	}
}

func TestRunEvaluatesEveryCaseConcurrently(t *testing.T) {
	target := ragexample.GoodTarget{Corpus: testCorpus()}
	j := judge.NewFake("fake", 1)
	runner := NewRunner(target, j, nil, Config{Concurrency: 2, JudgeSamples: 3}, rand.New(rand.NewSource(1)))

	result, err := runner.Run(context.Background(), goldenCases())
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Cases) != 3 {
		t.Fatalf("expected 3 case results, got %d", len(result.Cases))
	}
	for _, c := range result.Cases {
		if c.Err != nil {
			t.Fatalf("case %s: unexpected error: %v", c.CaseID, c.Err)
		}
		if len(c.Scores) != 3 {
			t.Fatalf("case %s: expected 3 judge samples, got %d", c.CaseID, len(c.Scores))
		}
		if c.MeanScore < 0.3 {
			t.Fatalf("case %s: expected a reasonably good score for GoodTarget, got %v", c.CaseID, c.MeanScore)
		}
	}
}

// countingJudge wraps a Judge to count real (non-cached) calls, so tests
// can assert on cache hit/miss behavior directly.
type countingJudge struct {
	inner judge.Judge
	calls atomic.Int64
}

func (c *countingJudge) Name() string { return c.inner.Name() }
func (c *countingJudge) Score(ctx context.Context, q, a, ref string) (judge.Verdict, error) {
	c.calls.Add(1)
	return c.inner.Score(ctx, q, a, ref)
}

func TestCachingAvoidsRepeatedJudgeCallsOnRerun(t *testing.T) {
	cachePath := filepath.Join(t.TempDir(), "cache.db")
	c, err := cache.Open(cachePath)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	target := ragexample.GoodTarget{Corpus: testCorpus()}
	cj := &countingJudge{inner: judge.NewFake("fake", 1)}
	cfg := Config{Concurrency: 2, JudgeSamples: 3, JudgeID: "fake-v1"}

	runner1 := NewRunner(target, cj, c, cfg, rand.New(rand.NewSource(1)))
	if _, err := runner1.Run(context.Background(), goldenCases()); err != nil {
		t.Fatal(err)
	}
	firstRunCalls := cj.calls.Load()
	if firstRunCalls == 0 {
		t.Fatal("expected real judge calls on the first run")
	}

	// A second run with the IDENTICAL config/target/cases must be free —
	// everything should come from cache.
	runner2 := NewRunner(target, cj, c, cfg, rand.New(rand.NewSource(2)))
	if _, err := runner2.Run(context.Background(), goldenCases()); err != nil {
		t.Fatal(err)
	}
	if got := cj.calls.Load(); got != firstRunCalls {
		t.Fatalf("expected zero additional judge calls on an identical rerun (cache should cover it), calls went from %d to %d", firstRunCalls, got)
	}
}

func TestCacheMissesOnDifferentJudgeID(t *testing.T) {
	cachePath := filepath.Join(t.TempDir(), "cache.db")
	c, err := cache.Open(cachePath)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	target := ragexample.GoodTarget{Corpus: testCorpus()}
	cj := &countingJudge{inner: judge.NewFake("fake", 1)}

	r1 := NewRunner(target, cj, c, Config{JudgeID: "v1"}, rand.New(rand.NewSource(1)))
	r1.Run(context.Background(), goldenCases())
	afterV1 := cj.calls.Load()

	// A different JudgeID (e.g. a changed rubric/prompt version) must NOT
	// reuse v1's cached scores — that would silently hide the effect of
	// the prompt change.
	r2 := NewRunner(target, cj, c, Config{JudgeID: "v2"}, rand.New(rand.NewSource(1)))
	r2.Run(context.Background(), goldenCases())
	if cj.calls.Load() <= afterV1 {
		t.Fatalf("expected a different JudgeID to bypass the cache and make real calls, calls stayed at %d", cj.calls.Load())
	}
}

// TestEndToEndGoodVsGoodIsNoSignificantDifference and
// TestEndToEndGoodVsDegradedIsRegression are the full-pipeline proof: a
// real (if toy) target, a real (if fake) judge, real concurrency and
// caching, feeding into the real statistical comparison — not synthetic
// numbers fed directly to the stats package.
func TestEndToEndGoodVsGoodIsNoSignificantDifference(t *testing.T) {
	corpus := testCorpus()
	j := judge.NewFake("fake", 42)
	cfg := Config{JudgeSamples: 5, BootstrapIters: 3000}

	r1 := NewRunner(ragexample.GoodTarget{Corpus: corpus}, j, nil, cfg, rand.New(rand.NewSource(10)))
	run1, err := r1.Run(context.Background(), goldenCases())
	if err != nil {
		t.Fatal(err)
	}
	r2 := NewRunner(ragexample.GoodTarget{Corpus: corpus}, j, nil, cfg, rand.New(rand.NewSource(11)))
	run2, err := r2.Run(context.Background(), goldenCases())
	if err != nil {
		t.Fatal(err)
	}

	cmp, err := stats.PairedBootstrapCompare(run1.MeanScores(), run2.MeanScores(), 3000, 0.95, rand.New(rand.NewSource(12)))
	if err != nil {
		t.Fatal(err)
	}
	if cmp.Verdict != stats.VerdictNoSignificantDifference {
		t.Fatalf("expected comparing the same target against itself to show no significant difference, got verdict=%s diff=%v ci=%+v",
			cmp.Verdict, cmp.MeanDiff, cmp.DiffCI)
	}
}

func TestEndToEndGoodVsDegradedIsRegression(t *testing.T) {
	corpus := testCorpus()
	j := judge.NewFake("fake", 42)
	cfg := Config{JudgeSamples: 5, BootstrapIters: 3000}

	baseline := NewRunner(ragexample.GoodTarget{Corpus: corpus}, j, nil, cfg, rand.New(rand.NewSource(20)))
	baselineRun, err := baseline.Run(context.Background(), goldenCases())
	if err != nil {
		t.Fatal(err)
	}
	candidate := NewRunner(ragexample.DegradedTarget{Corpus: corpus, TruncateToN: 3}, j, nil, cfg, rand.New(rand.NewSource(21)))
	candidateRun, err := candidate.Run(context.Background(), goldenCases())
	if err != nil {
		t.Fatal(err)
	}

	cmp, err := stats.PairedBootstrapCompare(baselineRun.MeanScores(), candidateRun.MeanScores(), 3000, 0.95, rand.New(rand.NewSource(22)))
	if err != nil {
		t.Fatal(err)
	}
	if cmp.Verdict != stats.VerdictRegression {
		t.Fatalf("expected the truncated-context target to be flagged as a regression against the full-context baseline, got verdict=%s diff=%v ci=%+v",
			cmp.Verdict, cmp.MeanDiff, cmp.DiffCI)
	}
}

func TestRunPropagatesTargetErrors(t *testing.T) {
	target := erroringTarget{}
	j := judge.NewFake("fake", 1)
	runner := NewRunner(target, j, nil, Config{}, rand.New(rand.NewSource(1)))

	result, err := runner.Run(context.Background(), goldenCases())
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range result.Cases {
		if c.Err == nil {
			t.Fatalf("expected every case to report the target's error, case %s had none", c.CaseID)
		}
	}
}

type erroringTarget struct{}

func (erroringTarget) Answer(ctx context.Context, question string) (string, error) {
	return "", errAlwaysFails
}

var errAlwaysFails = errAlwaysFailsType{}

type errAlwaysFailsType struct{}

func (errAlwaysFailsType) Error() string { return "target always fails (test)" }
