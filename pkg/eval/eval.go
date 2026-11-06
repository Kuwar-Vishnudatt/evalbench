// Package eval orchestrates a concurrent, cached, statistically-sampled
// evaluation run: for each golden case, get the target system's answer,
// score it with the judge multiple independent times (because a single
// LLM-judge call is noisy — see pkg/judge), and aggregate.
package eval

import (
	"context"
	"fmt"
	"math/rand"
	"sync"

	"github.com/Kuwar-Vishnudatt/evalbench/pkg/cache"
	"github.com/Kuwar-Vishnudatt/evalbench/pkg/judge"
	"github.com/Kuwar-Vishnudatt/evalbench/pkg/stats"
)

// GoldenCase is one test case: a question with a known-good reference
// answer to grade against.
type GoldenCase struct {
	ID        string
	Question  string
	Reference string
}

// Target is the system under evaluation (a RAG pipeline, a chat agent,
// anything that answers a question).
type Target interface {
	Answer(ctx context.Context, question string) (string, error)
}

type CaseResult struct {
	CaseID string
	Answer string
	// Scores holds every independent judge sample's score for this case —
	// kept, not just averaged, so a caller can see judge disagreement
	// directly (a high spread here means the judge itself isn't confident,
	// independent of whatever the golden-set-level bootstrap CI says about
	// cross-case uncertainty).
	Scores    []float64
	MeanScore float64
	PassRate  float64 // fraction of judge samples that passed
	Err       error
}

type RunResult struct {
	JudgeID string
	Cases   []CaseResult
	// MeanCI is the bootstrap confidence interval (over CASES, not judge
	// samples) for the run's overall mean score.
	MeanCI stats.CI
}

// MeanScores returns each case's mean score, in the same order as Cases —
// the shape PairedBootstrapCompare needs for comparing two runs over the
// identical golden set.
func (r RunResult) MeanScores() []float64 {
	out := make([]float64, len(r.Cases))
	for i, c := range r.Cases {
		out[i] = c.MeanScore
	}
	return out
}

type Config struct {
	Concurrency    int
	JudgeSamples   int // independent judge calls per case
	BootstrapIters int
	Confidence     float64
	// JudgeID identifies the judge+prompt-version for cache keys (e.g.
	// "fake-v1" or "gpt-4o-mini@rubric-v2"). Changing it invalidates old
	// cached scores, deliberately — see pkg/cache.
	JudgeID string
}

func (c Config) withDefaults() Config {
	if c.Concurrency <= 0 {
		c.Concurrency = 4
	}
	if c.JudgeSamples <= 0 {
		c.JudgeSamples = 3
	}
	if c.BootstrapIters <= 0 {
		c.BootstrapIters = 2000
	}
	if c.Confidence <= 0 {
		c.Confidence = 0.95
	}
	if c.JudgeID == "" {
		c.JudgeID = "default"
	}
	return c
}

type Runner struct {
	Target Target
	Judge  judge.Judge
	Cache  *cache.Cache // may be nil (no caching)
	Config Config
	rng    *rand.Rand
}

// NewRunner builds a Runner. rng drives bootstrap resampling for the
// run's confidence interval; pass a seeded one for reproducible tests.
func NewRunner(target Target, j judge.Judge, c *cache.Cache, cfg Config, rng *rand.Rand) *Runner {
	if rng == nil {
		rng = rand.New(rand.NewSource(1))
	}
	return &Runner{Target: target, Judge: j, Cache: c, Config: cfg.withDefaults(), rng: rng}
}

// Run evaluates every case with up to Config.Concurrency cases in flight
// at once.
func (r *Runner) Run(ctx context.Context, cases []GoldenCase) (RunResult, error) {
	results := make([]CaseResult, len(cases))
	sem := make(chan struct{}, r.Config.Concurrency)
	var wg sync.WaitGroup

	for i, c := range cases {
		wg.Add(1)
		go func(i int, c GoldenCase) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			results[i] = r.runCase(ctx, c)
		}(i, c)
	}
	wg.Wait()

	var meanScores []float64
	for _, c := range results {
		if c.Err == nil {
			meanScores = append(meanScores, c.MeanScore)
		}
	}

	ci := stats.BootstrapMeanCI(meanScores, r.Config.BootstrapIters, r.Config.Confidence, r.rng)
	return RunResult{JudgeID: r.Config.JudgeID, Cases: results, MeanCI: ci}, nil
}

func (r *Runner) runCase(ctx context.Context, c GoldenCase) CaseResult {
	answer, err := r.Target.Answer(ctx, c.Question)
	if err != nil {
		return CaseResult{CaseID: c.ID, Err: fmt.Errorf("target answer: %w", err)}
	}

	scores := make([]float64, 0, r.Config.JudgeSamples)
	passes := 0
	for sample := 0; sample < r.Config.JudgeSamples; sample++ {
		v, err := r.scoreOnce(ctx, c, answer, sample)
		if err != nil {
			return CaseResult{CaseID: c.ID, Answer: answer, Err: fmt.Errorf("judge sample %d: %w", sample, err)}
		}
		scores = append(scores, v.Score)
		if v.Pass {
			passes++
		}
	}

	return CaseResult{
		CaseID:    c.ID,
		Answer:    answer,
		Scores:    scores,
		MeanScore: stats.Mean(scores),
		PassRate:  float64(passes) / float64(len(scores)),
	}
}

func (r *Runner) scoreOnce(ctx context.Context, c GoldenCase, answer string, sample int) (judge.Verdict, error) {
	if r.Cache != nil {
		key := cache.Key(r.Config.JudgeID, c.Question, answer, c.Reference, sample)
		if entry, ok, err := r.Cache.Get(key); err == nil && ok {
			return judge.Verdict{Score: entry.Score, Pass: entry.Pass, Rationale: entry.Rationale}, nil
		}
		v, err := r.Judge.Score(ctx, c.Question, answer, c.Reference)
		if err != nil {
			return judge.Verdict{}, err
		}
		_ = r.Cache.Set(key, cache.Entry{Score: v.Score, Pass: v.Pass, Rationale: v.Rationale})
		return v, nil
	}
	return r.Judge.Score(ctx, c.Question, answer, c.Reference)
}
