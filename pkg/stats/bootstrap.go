// Package stats applies actual statistical rigor to comparing two eval
// runs — the thing a naive "72% vs 75%, so it's better" eval dashboard
// skips. An eval score's mean over a finite golden set has sampling
// uncertainty from two independent sources: the judge itself is noisy
// (the same answer can score differently across repeated judge calls —
// handled in pkg/eval by sampling a judge multiple times per case), and
// the golden set is itself a finite sample of the space of possible
// inputs (handled here, by bootstrap resampling over CASES). A 3-point
// improvement on 30 golden cases can easily be noise; this package is
// what lets the harness say so instead of reporting it as a finding.
package stats

import (
	"errors"
	"math"
	"math/rand"
	"sort"
)

var (
	errMismatchedLength = errors.New("stats: baseline and candidate must be the same length (paired by case)")
	errEmptyInput       = errors.New("stats: empty input")
)

func Mean(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	var sum float64
	for _, x := range xs {
		sum += x
	}
	return sum / float64(len(xs))
}

func StdDev(xs []float64) float64 {
	if len(xs) < 2 {
		return 0
	}
	m := Mean(xs)
	var sumSq float64
	for _, x := range xs {
		d := x - m
		sumSq += d * d
	}
	return math.Sqrt(sumSq / float64(len(xs)-1))
}

// CI is a confidence interval: Lower and Upper bound the quantity of
// interest at the given Confidence level (e.g. 0.95 for a 95% CI).
type CI struct {
	Lower, Upper, Confidence float64
}

// Contains reports whether v falls within [Lower, Upper].
func (c CI) Contains(v float64) bool { return v >= c.Lower && v <= c.Upper }

// BootstrapMeanCI computes a percentile-bootstrap confidence interval for
// the mean of xs: resample len(xs) values WITH replacement, iterations
// times, compute the mean each time, and take the percentile bounds of
// that distribution. This directly answers "given only this many samples,
// how much could the true mean plausibly differ from what we observed?"
func BootstrapMeanCI(xs []float64, iterations int, confidence float64, rng *rand.Rand) CI {
	if len(xs) == 0 {
		return CI{Confidence: confidence}
	}
	means := make([]float64, iterations)
	resample := make([]float64, len(xs))
	for i := 0; i < iterations; i++ {
		for j := range resample {
			resample[j] = xs[rng.Intn(len(xs))]
		}
		means[i] = Mean(resample)
	}
	lo, hi := percentileBounds(means, confidence)
	return CI{Lower: lo, Upper: hi, Confidence: confidence}
}

// Comparison classifies a paired before/after eval result.
type Comparison struct {
	MeanDiff float64 // mean(candidate) - mean(baseline)
	DiffCI   CI
	Verdict  string // "improvement" | "regression" | "no_significant_difference"
}

const (
	VerdictImprovement             = "improvement"
	VerdictRegression              = "regression"
	VerdictNoSignificantDifference = "no_significant_difference"
)

// PairedBootstrapCompare compares two equal-length, ORDER-PAIRED score
// slices (baseline[i] and candidate[i] must be the same golden case) via
// paired bootstrap: resample case INDICES with replacement (not the two
// slices independently — pairing matters, since it cancels out per-case
// difficulty variance that an unpaired comparison would conflate with a
// real system difference), compute mean(candidate-baseline) per resample,
// and take the confidence interval of that distribution. The verdict is
// "no significant difference" whenever the interval contains zero — a
// positive OR negative point estimate that the data simply can't
// distinguish from noise at this sample size.
func PairedBootstrapCompare(baseline, candidate []float64, iterations int, confidence float64, rng *rand.Rand) (Comparison, error) {
	if len(baseline) != len(candidate) {
		return Comparison{}, errMismatchedLength
	}
	n := len(baseline)
	if n == 0 {
		return Comparison{}, errEmptyInput
	}

	diffs := make([]float64, n)
	for i := range diffs {
		diffs[i] = candidate[i] - baseline[i]
	}
	meanDiff := Mean(diffs)

	resampleDiffs := make([]float64, iterations)
	scratch := make([]float64, n)
	for i := 0; i < iterations; i++ {
		for j := range scratch {
			scratch[j] = diffs[rng.Intn(n)]
		}
		resampleDiffs[i] = Mean(scratch)
	}
	lo, hi := percentileBounds(resampleDiffs, confidence)
	ci := CI{Lower: lo, Upper: hi, Confidence: confidence}

	verdict := VerdictNoSignificantDifference
	if !ci.Contains(0) {
		if meanDiff > 0 {
			verdict = VerdictImprovement
		} else {
			verdict = VerdictRegression
		}
	}

	return Comparison{MeanDiff: meanDiff, DiffCI: ci, Verdict: verdict}, nil
}

func percentileBounds(xs []float64, confidence float64) (lo, hi float64) {
	sorted := append([]float64(nil), xs...)
	sort.Float64s(sorted)
	alpha := (1 - confidence) / 2
	loIdx := int(alpha * float64(len(sorted)))
	hiIdx := int((1 - alpha) * float64(len(sorted)))
	if hiIdx >= len(sorted) {
		hiIdx = len(sorted) - 1
	}
	return sorted[loIdx], sorted[hiIdx]
}
