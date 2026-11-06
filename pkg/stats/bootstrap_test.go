package stats

import (
	"math/rand"
	"testing"
)

func TestMeanAndStdDev(t *testing.T) {
	xs := []float64{1, 2, 3, 4, 5}
	if m := Mean(xs); m != 3 {
		t.Fatalf("expected mean 3, got %v", m)
	}
	// Known sample stddev of 1..5 is sqrt(2.5) ≈ 1.5811
	if sd := StdDev(xs); sd < 1.58 || sd > 1.59 {
		t.Fatalf("expected stddev ~1.581, got %v", sd)
	}
}

func TestBootstrapMeanCIContainsTrueMeanMostOfTheTime(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	// Simulate 100 "cases" drawn from a known distribution (mean 0.7).
	xs := make([]float64, 100)
	for i := range xs {
		xs[i] = 0.7 + rng.NormFloat64()*0.1
	}
	ci := BootstrapMeanCI(xs, 2000, 0.95, rng)
	if !ci.Contains(Mean(xs)) {
		t.Fatalf("CI should always contain the observed sample mean itself, got %+v mean=%v", ci, Mean(xs))
	}
	if ci.Lower > 0.75 || ci.Upper < 0.65 {
		t.Fatalf("expected a CI roughly bracketing the true mean 0.7, got [%v, %v]", ci.Lower, ci.Upper)
	}
}

func TestBootstrapMeanCINarrowsWithMoreSamples(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	small := make([]float64, 5)
	large := make([]float64, 500)
	for i := range small {
		small[i] = 0.7 + rng.NormFloat64()*0.1
	}
	for i := range large {
		large[i] = 0.7 + rng.NormFloat64()*0.1
	}
	ciSmall := BootstrapMeanCI(small, 2000, 0.95, rng)
	ciLarge := BootstrapMeanCI(large, 2000, 0.95, rng)
	widthSmall := ciSmall.Upper - ciSmall.Lower
	widthLarge := ciLarge.Upper - ciLarge.Lower
	if widthLarge >= widthSmall {
		t.Fatalf("expected a much wider CI with only 5 samples than with 500, got small=%v large=%v", widthSmall, widthLarge)
	}
}

func TestPairedBootstrapCompareDetectsClearRegression(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	n := 40
	baseline := make([]float64, n)
	candidate := make([]float64, n)
	for i := range baseline {
		baseline[i] = 0.85 + rng.NormFloat64()*0.05
		candidate[i] = 0.55 + rng.NormFloat64()*0.05 // clearly, consistently worse
	}
	cmp, err := PairedBootstrapCompare(baseline, candidate, 2000, 0.95, rng)
	if err != nil {
		t.Fatal(err)
	}
	if cmp.Verdict != VerdictRegression {
		t.Fatalf("expected a clear regression to be detected, got verdict=%s diff=%v ci=%+v", cmp.Verdict, cmp.MeanDiff, cmp.DiffCI)
	}
}

func TestPairedBootstrapCompareDetectsClearImprovement(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	n := 40
	baseline := make([]float64, n)
	candidate := make([]float64, n)
	for i := range baseline {
		baseline[i] = 0.5 + rng.NormFloat64()*0.05
		candidate[i] = 0.9 + rng.NormFloat64()*0.05
	}
	cmp, err := PairedBootstrapCompare(baseline, candidate, 2000, 0.95, rng)
	if err != nil {
		t.Fatal(err)
	}
	if cmp.Verdict != VerdictImprovement {
		t.Fatalf("expected a clear improvement to be detected, got verdict=%s diff=%v ci=%+v", cmp.Verdict, cmp.MeanDiff, cmp.DiffCI)
	}
}

// TestPairedBootstrapCompareDoesNotOverreactToNoise is the test that
// actually matters most: a tiny, noise-sized difference on a small sample
// must NOT be reported as a regression or improvement. A naive "is the
// mean lower" comparison would get this wrong every time; that's the
// entire reason this package exists.
func TestPairedBootstrapCompareDoesNotOverreactToNoise(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	n := 15 // deliberately small, realistic golden-set size
	baseline := make([]float64, n)
	candidate := make([]float64, n)
	for i := range baseline {
		shared := 0.75 + rng.NormFloat64()*0.15 // same underlying quality
		baseline[i] = clamp01(shared + rng.NormFloat64()*0.05)
		candidate[i] = clamp01(shared + rng.NormFloat64()*0.05) // independent noise only
	}
	cmp, err := PairedBootstrapCompare(baseline, candidate, 2000, 0.95, rng)
	if err != nil {
		t.Fatal(err)
	}
	if cmp.Verdict != VerdictNoSignificantDifference {
		t.Fatalf("expected noise-level differences on a small sample to be classified as no significant difference, got verdict=%s diff=%v ci=%+v",
			cmp.Verdict, cmp.MeanDiff, cmp.DiffCI)
	}
}

func TestPairedBootstrapCompareRejectsMismatchedLength(t *testing.T) {
	rng := rand.New(rand.NewSource(6))
	_, err := PairedBootstrapCompare([]float64{1, 2}, []float64{1, 2, 3}, 100, 0.95, rng)
	if err == nil {
		t.Fatal("expected an error for mismatched-length paired input")
	}
}

func clamp01(x float64) float64 {
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}
