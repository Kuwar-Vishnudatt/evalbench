// Command evalbench runs a golden-case eval against the bundled example
// RAG target (or compares two saved runs), demonstrating the full
// pipeline end to end from the command line. See pkg/eval to wire in a
// real target and pkg/judge.NewHTTPJudge for a real LLM judge.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"time"

	"github.com/Kuwar-Vishnudatt/evalbench/pkg/cache"
	"github.com/Kuwar-Vishnudatt/evalbench/pkg/eval"
	"github.com/Kuwar-Vishnudatt/evalbench/pkg/judge"
	"github.com/Kuwar-Vishnudatt/evalbench/pkg/ragexample"
	"github.com/Kuwar-Vishnudatt/evalbench/pkg/stats"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: evalbench run|compare [flags]")
		os.Exit(2)
	}
	switch os.Args[1] {
	case "run":
		runCmd(os.Args[2:])
	case "compare":
		compareCmd(os.Args[2:])
	default:
		fmt.Fprintf(os.Stderr, "unknown command %q (want: run, compare)\n", os.Args[1])
		os.Exit(2)
	}
}

func runCmd(args []string) {
	fs := flag.NewFlagSet("run", flag.ExitOnError)
	casesPath := fs.String("cases", "", "path to a JSON array of {id,question,reference} golden cases")
	target := fs.String("target", "good", "example target variant: good | degraded")
	truncateN := fs.Int("degraded-truncate-words", 3, "DegradedTarget: words to truncate the answer to")
	judgeSamples := fs.Int("judge-samples", 3, "independent judge calls per case")
	concurrency := fs.Int("concurrency", 4, "concurrent cases in flight")
	judgeID := fs.String("judge-id", "fake-v1", "cache key namespace for the judge+prompt version")
	cachePath := fs.String("cache", "evalbench-cache.db", "path to the judge-call cache (empty = no caching)")
	outPath := fs.String("out", "run.json", "where to write the run result JSON")
	seed := fs.Int64("seed", 1, "RNG seed, for reproducible bootstrap confidence intervals")
	fs.Parse(args)

	if *casesPath == "" {
		fmt.Fprintln(os.Stderr, "run: -cases is required")
		os.Exit(2)
	}
	cases, err := loadCases(*casesPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load cases: %v\n", err)
		os.Exit(1)
	}

	corpus := exampleCorpus()
	var t eval.Target
	switch *target {
	case "good":
		t = ragexample.GoodTarget{Corpus: corpus}
	case "degraded":
		t = ragexample.DegradedTarget{Corpus: corpus, TruncateToN: *truncateN}
	default:
		fmt.Fprintf(os.Stderr, "unknown -target %q (want good or degraded)\n", *target)
		os.Exit(2)
	}

	var c *cache.Cache
	if *cachePath != "" {
		c, err = cache.Open(*cachePath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "open cache: %v\n", err)
			os.Exit(1)
		}
		defer c.Close()
	}

	j := judge.NewFake("fake", 42)
	runner := eval.NewRunner(t, j, c, eval.Config{
		Concurrency:  *concurrency,
		JudgeSamples: *judgeSamples,
		JudgeID:      *judgeID,
	}, rand.New(rand.NewSource(*seed)))

	start := time.Now()
	result, err := runner.Run(context.Background(), cases)
	if err != nil {
		fmt.Fprintf(os.Stderr, "run: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("ran %d case(s) in %v: mean score %.3f (95%% CI [%.3f, %.3f])\n",
		len(result.Cases), time.Since(start), stats.Mean(result.MeanScores()), result.MeanCI.Lower, result.MeanCI.Upper)

	if err := saveResult(*outPath, result); err != nil {
		fmt.Fprintf(os.Stderr, "save result: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("wrote %s\n", *outPath)
}

func compareCmd(args []string) {
	fs := flag.NewFlagSet("compare", flag.ExitOnError)
	baselinePath := fs.String("baseline", "", "baseline run.json")
	candidatePath := fs.String("candidate", "", "candidate run.json")
	iters := fs.Int("iterations", 3000, "bootstrap resample iterations")
	confidence := fs.Float64("confidence", 0.95, "confidence level")
	seed := fs.Int64("seed", 1, "RNG seed")
	fs.Parse(args)

	if *baselinePath == "" || *candidatePath == "" {
		fmt.Fprintln(os.Stderr, "compare: -baseline and -candidate are required")
		os.Exit(2)
	}

	baseline, err := loadResult(*baselinePath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load baseline: %v\n", err)
		os.Exit(1)
	}
	candidate, err := loadResult(*candidatePath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load candidate: %v\n", err)
		os.Exit(1)
	}

	cmp, err := stats.PairedBootstrapCompare(baseline.MeanScores(), candidate.MeanScores(), *iters, *confidence, rand.New(rand.NewSource(*seed)))
	if err != nil {
		fmt.Fprintf(os.Stderr, "compare: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("verdict: %s\n", cmp.Verdict)
	fmt.Printf("mean difference (candidate - baseline): %+.4f\n", cmp.MeanDiff)
	fmt.Printf("%.0f%% CI on that difference: [%+.4f, %+.4f]\n", cmp.DiffCI.Confidence*100, cmp.DiffCI.Lower, cmp.DiffCI.Upper)
	if cmp.Verdict == stats.VerdictRegression {
		os.Exit(1) // non-zero exit so this is usable as a CI gate
	}
}

type caseFile struct {
	ID        string `json:"id"`
	Question  string `json:"question"`
	Reference string `json:"reference"`
}

func loadCases(path string) ([]eval.GoldenCase, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var raw []caseFile
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, err
	}
	cases := make([]eval.GoldenCase, len(raw))
	for i, r := range raw {
		cases[i] = eval.GoldenCase{ID: r.ID, Question: r.Question, Reference: r.Reference}
	}
	return cases, nil
}

// storedResult is RunResult's JSON-safe shape (CaseResult.Err is an error,
// not directly serializable).
type storedResult struct {
	JudgeID string             `json:"judge_id"`
	MeanCI  stats.CI           `json:"mean_ci"`
	Cases   []storedCaseResult `json:"cases"`
}
type storedCaseResult struct {
	CaseID    string    `json:"case_id"`
	Answer    string    `json:"answer"`
	Scores    []float64 `json:"scores"`
	MeanScore float64   `json:"mean_score"`
	PassRate  float64   `json:"pass_rate"`
	Error     string    `json:"error,omitempty"`
}

func saveResult(path string, r eval.RunResult) error {
	sr := storedResult{JudgeID: r.JudgeID, MeanCI: r.MeanCI, Cases: make([]storedCaseResult, len(r.Cases))}
	for i, c := range r.Cases {
		scr := storedCaseResult{CaseID: c.CaseID, Answer: c.Answer, Scores: c.Scores, MeanScore: c.MeanScore, PassRate: c.PassRate}
		if c.Err != nil {
			scr.Error = c.Err.Error()
		}
		sr.Cases[i] = scr
	}
	data, err := json.MarshalIndent(sr, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

func loadResult(path string) (eval.RunResult, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return eval.RunResult{}, err
	}
	var sr storedResult
	if err := json.Unmarshal(data, &sr); err != nil {
		return eval.RunResult{}, err
	}
	r := eval.RunResult{JudgeID: sr.JudgeID, MeanCI: sr.MeanCI, Cases: make([]eval.CaseResult, len(sr.Cases))}
	for i, c := range sr.Cases {
		r.Cases[i] = eval.CaseResult{CaseID: c.CaseID, Answer: c.Answer, Scores: c.Scores, MeanScore: c.MeanScore, PassRate: c.PassRate}
	}
	return r, nil
}

func exampleCorpus() ragexample.Corpus {
	return ragexample.Corpus{
		{ID: "d1", Text: "The capital of France is Paris, a city known for the Eiffel Tower and the Louvre museum."},
		{ID: "d2", Text: "Mount Everest is the tallest mountain above sea level, located in the Himalayas on the border of Nepal and Tibet."},
		{ID: "d3", Text: "Photosynthesis is the process by which plants convert sunlight into chemical energy stored in glucose."},
		{ID: "d4", Text: "The Great Wall of China is a series of fortifications built across the historical northern borders of China."},
		{ID: "d5", Text: "The speed of light in a vacuum is approximately 299,792 kilometers per second."},
	}
}
