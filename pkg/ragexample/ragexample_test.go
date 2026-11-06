package ragexample

import (
	"context"
	"strings"
	"testing"
)

func sampleCorpus() Corpus {
	return Corpus{
		{ID: "d1", Text: "The capital of France is Paris, known for the Eiffel Tower."},
		{ID: "d2", Text: "Mount Everest is the tallest mountain in the Himalayas."},
	}
}

func TestGoodTargetRetrievesRelevantDocument(t *testing.T) {
	target := GoodTarget{Corpus: sampleCorpus()}
	answer, err := target.Answer(context.Background(), "What is the capital of France")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(answer, "Paris") {
		t.Fatalf("expected the France-related document to be retrieved, got %q", answer)
	}
}

func TestDegradedTargetTruncatesAnswer(t *testing.T) {
	target := DegradedTarget{Corpus: sampleCorpus(), TruncateToN: 3}
	answer, err := target.Answer(context.Background(), "What is the tallest mountain")
	if err != nil {
		t.Fatal(err)
	}
	words := strings.Fields(answer)
	if len(words) != 3 {
		t.Fatalf("expected exactly 3 words from truncation, got %d: %q", len(words), answer)
	}
	if !strings.Contains(answer, "Everest") {
		t.Fatalf("expected the truncated answer to still reference the right document, got %q", answer)
	}
}

func TestEmptyCorpusReturnsEmptyAnswer(t *testing.T) {
	target := GoodTarget{Corpus: Corpus{}}
	answer, err := target.Answer(context.Background(), "anything")
	if err != nil {
		t.Fatal(err)
	}
	if answer != "" {
		t.Fatalf("expected an empty answer from an empty corpus, got %q", answer)
	}
}
