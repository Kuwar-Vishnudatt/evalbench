package judge

import (
	"context"
	"testing"
	"time"
)

func TestFakeScoresHigherForCloserAnswers(t *testing.T) {
	f := NewFake("f", 1)
	ctx := context.Background()

	good, err := f.Score(ctx, "q", "Paris is the capital of France", "Paris is the capital of France")
	if err != nil {
		t.Fatal(err)
	}
	bad, err := f.Score(ctx, "q", "bananas are yellow", "Paris is the capital of France")
	if err != nil {
		t.Fatal(err)
	}
	if good.Score <= bad.Score {
		t.Fatalf("expected a near-identical answer to score higher than an unrelated one, good=%v bad=%v", good.Score, bad.Score)
	}
	if !good.Pass {
		t.Fatalf("expected a near-identical answer to pass, got %+v", good)
	}
}

func TestFakeFailEveryNthCall(t *testing.T) {
	f := NewFake("f", 1)
	f.SetFailEveryNthCall(3)
	ctx := context.Background()

	var errs int
	for i := 0; i < 9; i++ {
		if _, err := f.Score(ctx, "q", "a", "a"); err != nil {
			errs++
		}
	}
	if errs != 3 {
		t.Fatalf("expected exactly 3 failures out of 9 calls with failEvery=3, got %d", errs)
	}
}

func TestFakeRespectsContextCancellationDuringLatency(t *testing.T) {
	f := NewFake("f", 1)
	f.SetLatency(time.Hour)
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()

	_, err := f.Score(ctx, "q", "a", "a")
	if err == nil {
		t.Fatal("expected context deadline to interrupt a long simulated latency")
	}
}

func TestFakeNoiseStaysWithinScoreBounds(t *testing.T) {
	f := NewFake("f", 1)
	f.SetNoiseStd(2.0) // deliberately huge, to try to push score out of [0,1]
	ctx := context.Background()
	for i := 0; i < 50; i++ {
		v, err := f.Score(ctx, "q", "some answer text", "some answer text")
		if err != nil {
			t.Fatal(err)
		}
		if v.Score < 0 || v.Score > 1 {
			t.Fatalf("score escaped [0,1] bounds despite clamping: %v", v.Score)
		}
	}
}
