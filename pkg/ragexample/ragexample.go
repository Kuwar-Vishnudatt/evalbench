// Package ragexample is a deliberately minimal retrieval-and-answer target
// — just enough to be a REAL subject for evalbench to evaluate, not a
// second RAG system as a deliverable in its own right (the author already
// has two other RAG-flavored portfolio projects; the point of THIS project
// is the evaluation harness, not another retrieval architecture). It
// ships two variants precisely so the harness's own tests can prove the
// statistical comparison logic works on a realistic case: GoodTarget vs.
// itself should report no significant difference; GoodTarget vs.
// DegradedTarget (which simulates a real, common RAG regression —
// truncating the retrieved context) should be caught as a regression.
package ragexample

import (
	"context"
	"sort"
	"strings"
)

type Document struct {
	ID   string
	Text string
}

type Corpus []Document

// retrieve returns the document with the highest word-overlap score
// against query — a toy retriever, intentionally: good enough to make
// "the degraded variant answers worse" a real, measurable effect, without
// building a second embeddings-based retrieval system redundant with the
// author's existing hybrid-rag project.
func (c Corpus) retrieve(query string) Document {
	type scored struct {
		doc   Document
		score float64
	}
	scoredDocs := make([]scored, len(c))
	for i, d := range c {
		scoredDocs[i] = scored{doc: d, score: overlap(query, d.Text)}
	}
	sort.Slice(scoredDocs, func(i, j int) bool { return scoredDocs[i].score > scoredDocs[j].score })
	if len(scoredDocs) == 0 {
		return Document{}
	}
	return scoredDocs[0].doc
}

func overlap(a, b string) float64 {
	setA := wordSet(a)
	setB := wordSet(b)
	if len(setA) == 0 {
		return 0
	}
	hits := 0
	for w := range setA {
		if setB[w] {
			hits++
		}
	}
	return float64(hits) / float64(len(setA))
}

func wordSet(s string) map[string]bool {
	out := make(map[string]bool)
	for _, w := range strings.Fields(strings.ToLower(s)) {
		w = strings.Trim(w, ".,!?;:\"'()")
		if w != "" {
			out[w] = true
		}
	}
	return out
}

// GoodTarget answers with the full text of the best-matching document.
type GoodTarget struct{ Corpus Corpus }

func (t GoodTarget) Answer(ctx context.Context, question string) (string, error) {
	return t.Corpus.retrieve(question).Text, nil
}

// DegradedTarget simulates a real, common regression class: the retrieved
// context gets truncated (e.g. a chunking bug, a context-window
// miscalculation) before being used to answer, so the "answer" only ever
// reflects the first few words of the right document — present, but
// substantially less complete/correct than GoodTarget's.
type DegradedTarget struct {
	Corpus      Corpus
	TruncateToN int // words
}

func (t DegradedTarget) Answer(ctx context.Context, question string) (string, error) {
	doc := t.Corpus.retrieve(question)
	n := t.TruncateToN
	if n <= 0 {
		n = 4
	}
	words := strings.Fields(doc.Text)
	if len(words) > n {
		words = words[:n]
	}
	return strings.Join(words, " "), nil
}
