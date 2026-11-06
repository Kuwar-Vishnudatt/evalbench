package judge

import (
	"context"
	"errors"
	"math/rand"
	"strings"
	"sync/atomic"
	"time"
)

// Fake is a fully deterministic (given a fixed seed), configurable judge:
// its base score is word-overlap similarity between answer and reference
// (a real, if crude, measure — not a random number), with optional
// injected noise (to simulate a real LLM judge's run-to-run variance),
// latency, and failure injection. Driving the rest of the harness's
// concurrency/caching/statistics logic against this instead of a live LLM
// API is what makes that logic's own tests fast, free, and deterministic.
type Fake struct {
	name      string
	latency   time.Duration
	noiseStd  float64 // stddev of Gaussian noise added to the base score
	passAt    float64 // Pass = (noisy score >= passAt)
	failEvery int     // if > 0, every Nth call errors instead of scoring
	rng       *rand.Rand

	calls atomic.Int64
}

func NewFake(name string, seed int64) *Fake {
	return &Fake{name: name, passAt: 0.5, rng: rand.New(rand.NewSource(seed))}
}

func (f *Fake) Name() string { return f.name }

func (f *Fake) SetLatency(d time.Duration) { f.latency = d }
func (f *Fake) SetNoiseStd(std float64)    { f.noiseStd = std }
func (f *Fake) SetPassThreshold(t float64) { f.passAt = t }
func (f *Fake) SetFailEveryNthCall(n int)  { f.failEvery = n }
func (f *Fake) Calls() int64               { return f.calls.Load() }

func (f *Fake) Score(ctx context.Context, question, answer, reference string) (Verdict, error) {
	n := f.calls.Add(1)
	if f.failEvery > 0 && n%int64(f.failEvery) == 0 {
		return Verdict{}, errors.New("fake judge: simulated transient failure")
	}

	if f.latency > 0 {
		select {
		case <-time.After(f.latency):
		case <-ctx.Done():
			return Verdict{}, ctx.Err()
		}
	}

	base := wordOverlap(answer, reference)
	score := base
	if f.noiseStd > 0 {
		score += f.rng.NormFloat64() * f.noiseStd
	}
	if score < 0 {
		score = 0
	}
	if score > 1 {
		score = 1
	}

	return Verdict{
		Score:     score,
		Pass:      score >= f.passAt,
		Rationale: "fake judge: word-overlap similarity (+ simulated noise)",
	}, nil
}

// wordOverlap is a crude but genuine similarity measure: the Jaccard
// index of the two strings' lowercased word sets. Good enough to make
// "the degraded target produces worse answers" actually show up as a
// lower score in tests, without needing a real LLM to notice it.
func wordOverlap(a, b string) float64 {
	setA := wordSet(a)
	setB := wordSet(b)
	if len(setA) == 0 && len(setB) == 0 {
		return 1
	}
	if len(setA) == 0 || len(setB) == 0 {
		return 0
	}
	intersection := 0
	for w := range setA {
		if setB[w] {
			intersection++
		}
	}
	union := len(setA) + len(setB) - intersection
	if union == 0 {
		return 0
	}
	return float64(intersection) / float64(union)
}

func wordSet(s string) map[string]bool {
	words := strings.Fields(strings.ToLower(s))
	set := make(map[string]bool, len(words))
	for _, w := range words {
		w = strings.Trim(w, ".,!?;:\"'()")
		if w != "" {
			set[w] = true
		}
	}
	return set
}
