package judge

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
)

// HTTPJudge calls a real LLM through any OpenAI-compatible chat-completions
// endpoint (OpenAI, Groq, Together, a local vLLM/Ollama server, ...) with a
// fixed rubric prompt, and parses a structured verdict back out of the
// model's response. This is the judge a real evaluation run uses; Fake
// (fake.go) is what the harness's own tests run against, so they're fast,
// free, and deterministic.
type HTTPJudge struct {
	name    string
	baseURL string
	apiKey  string
	model   string
	client  *http.Client
}

func NewHTTPJudge(name, baseURL, apiKey, model string) *HTTPJudge {
	return &HTTPJudge{name: name, baseURL: baseURL, apiKey: apiKey, model: model, client: &http.Client{}}
}

func (j *HTTPJudge) Name() string { return j.name }

const rubricTemplate = `You are a strict grader. Score the ANSWER against the REFERENCE for the given QUESTION.

QUESTION: %s

REFERENCE (ground truth): %s

ANSWER (to grade): %s

Respond with EXACTLY two lines, nothing else:
SCORE: <a number from 0.0 to 1.0, where 1.0 means the answer fully and correctly conveys the reference's content, 0.0 means it is wrong or irrelevant>
RATIONALE: <one sentence>`

type chatRequest struct {
	Model    string        `json:"model"`
	Messages []chatMessage `json:"messages"`
}
type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}
type chatResponse struct {
	Choices []struct {
		Message chatMessage `json:"message"`
	} `json:"choices"`
}

func (j *HTTPJudge) Score(ctx context.Context, question, answer, reference string) (Verdict, error) {
	prompt := fmt.Sprintf(rubricTemplate, question, reference, answer)
	body, err := json.Marshal(chatRequest{Model: j.model, Messages: []chatMessage{{Role: "user", Content: prompt}}})
	if err != nil {
		return Verdict{}, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, j.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return Verdict{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	if j.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+j.apiKey)
	}

	resp, err := j.client.Do(req)
	if err != nil {
		return Verdict{}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return Verdict{}, fmt.Errorf("judge API returned status %d", resp.StatusCode)
	}

	var parsed chatResponse
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return Verdict{}, fmt.Errorf("decode judge response: %w", err)
	}
	if len(parsed.Choices) == 0 {
		return Verdict{}, fmt.Errorf("judge response had no choices")
	}

	return parseVerdict(parsed.Choices[0].Message.Content)
}

func parseVerdict(text string) (Verdict, error) {
	var score float64
	var rationale string
	foundScore := false

	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimSpace(line)
		switch {
		case strings.HasPrefix(strings.ToUpper(line), "SCORE:"):
			raw := strings.TrimSpace(line[strings.Index(line, ":")+1:])
			v, err := strconv.ParseFloat(raw, 64)
			if err != nil {
				return Verdict{}, fmt.Errorf("parse score %q: %w", raw, err)
			}
			score = v
			foundScore = true
		case strings.HasPrefix(strings.ToUpper(line), "RATIONALE:"):
			rationale = strings.TrimSpace(line[strings.Index(line, ":")+1:])
		}
	}

	if !foundScore {
		return Verdict{}, fmt.Errorf("judge response did not contain a parseable SCORE line: %q", text)
	}
	if score < 0 || score > 1 {
		return Verdict{}, fmt.Errorf("judge returned out-of-range score %v", score)
	}

	return Verdict{Score: score, Pass: score >= 0.5, Rationale: rationale}, nil
}
