package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// fakeSender scripts model replies so the agent loop is testable without
// network access. All calls happen on the test goroutine. Review passes
// carry no tools; reviewErr fails them to test reviewer outages.
type fakeSender struct {
	t         *testing.T
	requests  []messageRequest
	script    func(call int, req messageRequest) messageResponse
	reviewErr error
}

func (f *fakeSender) createMessage(_ context.Context, req messageRequest) (*messageResponse, error) {
	f.requests = append(f.requests, req)
	if len(req.Tools) == 0 && f.reviewErr != nil {
		return nil, f.reviewErr
	}
	resp := f.script(len(f.requests), req)
	return &resp, nil
}

// reviewVerdictResponse scripts the reviewer pass: a bare text verdict with
// no tools, exactly like reviewToolCall sends.
func reviewVerdictResponse(text string) messageResponse {
	return messageResponse{
		ID:         "msg_review",
		Content:    []contentBlock{{Type: "text", Text: text}},
		StopReason: "end_turn",
	}
}

func advertisedTools(req messageRequest) string {
	var names []string
	for _, tool := range req.Tools {
		names = append(names, tool.Name)
	}
	return strings.Join(names, ",")
}

func requestJSON(t *testing.T, req messageRequest) string {
	t.Helper()
	raw, err := json.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}
	return string(raw)
}

// TestAgentLoopWithStubAPI drives the whole prompt → tool_use → tool_result
// → answer cycle, including real bash execution.
func TestAgentLoopWithStubAPI(t *testing.T) {
	sender := &fakeSender{t: t}
	sender.script = func(call int, req messageRequest) messageResponse {
		if len(req.Tools) == 0 {
			return reviewVerdictResponse("APPROVE: harmless echo")
		}
		if got := advertisedTools(req); got != "bash,read,edit,write" {
			t.Errorf("unexpected tools advertised: %s", got)
		}
		if req.Model != "test-model" {
			t.Errorf("unexpected model: %s", req.Model)
		}
		if call == 1 {
			return messageResponse{
				ID: "msg_1",
				Content: []contentBlock{
					{Type: "text", Text: "I'll run a command."},
					{Type: "tool_use", ID: "toolu_1", Name: "bash",
						Input: json.RawMessage(`{"command":"echo hello-harnais"}`)},
				},
				StopReason: "tool_use",
			}
		}
		// The executed tool result must have round-tripped into the follow-up.
		if blob := requestJSON(t, req); !strings.Contains(blob, "toolu_1") || !strings.Contains(blob, "hello-harnais") {
			t.Errorf("tool result did not round-trip: %s", blob)
		}
		return messageResponse{
			ID:         fmt.Sprintf("msg_%d", call),
			Content:    []contentBlock{{Type: "text", Text: "done"}},
			StopReason: "end_turn",
		}
	}

	cfg := config{model: "test-model", maxTokens: 128, maxIters: 5}
	answer, history, err := runPrompt(context.Background(), sender, cfg, nil, "hi",
		func(format string, args ...any) {})
	if err != nil {
		t.Fatalf("runPrompt: %v", err)
	}
	if answer != "done" {
		t.Fatalf("unexpected answer: %q", answer)
	}
	if len(history) != 4 {
		t.Fatalf("expected 4 messages (user, assistant, tools, answer), got %d", len(history))
	}
}

// proposeEcho scripts a model that proposes one bash call, then finishes.
func proposeEcho(t *testing.T, command, final string) func(call int, req messageRequest) messageResponse {
	t.Helper()
	return func(call int, req messageRequest) messageResponse {
		if len(req.Tools) == 0 {
			t.Errorf("review pass needs an explicit verdict script")
			return reviewVerdictResponse("DENY: test forgot the verdict")
		}
		if len(req.Messages) == 1 {
			return messageResponse{
				ID: "msg_1",
				Content: []contentBlock{
					{Type: "tool_use", ID: "toolu_9", Name: "bash",
						Input: json.RawMessage(`{"command":` + strconv.Quote(command) + `}`)},
				},
				StopReason: "tool_use",
			}
		}
		return messageResponse{
			ID:         "msg_2",
			Content:    []contentBlock{{Type: "text", Text: final}},
			StopReason: "end_turn",
		}
	}
}

// TestReviewerApprovesRunsTheCall verifies an approved mutating call
// executes with no human step: the reviewer said yes, so it runs.
func TestReviewerApprovesRunsTheCall(t *testing.T) {
	sender := &fakeSender{t: t}
	sender.script = func(call int, req messageRequest) messageResponse {
		if len(req.Tools) == 0 {
			return reviewVerdictResponse("APPROVE: harmless echo")
		}
		return proposeEcho(t, "echo reviewed-ok", "ran it")(call, req)
	}

	cfg := config{model: "test-model", maxTokens: 128, maxIters: 5}
	answer, _, err := runPrompt(context.Background(), sender, cfg, nil, "hi",
		func(format string, args ...any) {})
	if err != nil {
		t.Fatalf("runPrompt: %v", err)
	}
	if answer != "ran it" {
		t.Fatalf("unexpected answer: %q", answer)
	}
	if blob := requestJSON(t, sender.requests[len(sender.requests)-1]); !strings.Contains(blob, "reviewed-ok") {
		t.Errorf("approved result did not round-trip: %s", blob)
	}
}

// TestReviewerDeniesBlocksExecution verifies a denied call never runs and
// the rationale reaches the model with a safer-path instruction.
func TestReviewerDeniesBlocksExecution(t *testing.T) {
	marker := filepath.Join(t.TempDir(), "must-not-exist")
	sender := &fakeSender{t: t}
	sender.script = func(call int, req messageRequest) messageResponse {
		if len(req.Tools) == 0 {
			return reviewVerdictResponse("DENY: writes outside the workdir")
		}
		return proposeEcho(t, "touch "+marker, "stood down")(call, req)
	}

	cfg := config{model: "test-model", maxTokens: 128, maxIters: 5}
	answer, _, err := runPrompt(context.Background(), sender, cfg, nil, "hi",
		func(format string, args ...any) {})
	if err != nil {
		t.Fatalf("runPrompt: %v", err)
	}
	if answer != "stood down" {
		t.Fatalf("unexpected answer: %q", answer)
	}
	if _, err := os.Stat(marker); !os.IsNotExist(err) {
		t.Error("denied command must not execute")
	}
	blob := requestJSON(t, sender.requests[len(sender.requests)-1])
	if !strings.Contains(blob, "Reviewer denied this action") || !strings.Contains(blob, "writes outside the workdir") {
		t.Errorf("denial rationale did not reach the model: %s", blob)
	}
	if !strings.Contains(blob, "materially safer alternative") {
		t.Errorf("denial must instruct a safer path: %s", blob)
	}
}

// TestReviewerBreakerAbortsTurn verifies three consecutive denials abort
// the turn instead of looping on more escalation attempts.
func TestReviewerBreakerAbortsTurn(t *testing.T) {
	sender := &fakeSender{t: t}
	sender.script = func(call int, req messageRequest) messageResponse {
		if len(req.Tools) == 0 {
			return reviewVerdictResponse("DENY: no")
		}
		return messageResponse{
			ID: fmt.Sprintf("msg_%d", call),
			Content: []contentBlock{
				{Type: "tool_use", ID: fmt.Sprintf("toolu_%d", call), Name: "bash",
					Input: json.RawMessage(`{"command":"echo again"}`)},
			},
			StopReason: "tool_use",
		}
	}

	cfg := config{model: "test-model", maxTokens: 128, maxIters: 10}
	_, _, err := runPrompt(context.Background(), sender, cfg, nil, "hi",
		func(format string, args ...any) {})
	if err == nil || !strings.Contains(err.Error(), "3 actions in a row") {
		t.Fatalf("expected circuit-breaker error, got %v", err)
	}
}

// TestReviewerUnavailableAbortsTurn verifies a reviewer transport failure
// aborts distinctly from a denial.
func TestReviewerUnavailableAbortsTurn(t *testing.T) {
	sender := &fakeSender{t: t, reviewErr: fmt.Errorf("backend down")}
	sender.script = proposeEcho(t, "echo hi", "unused")

	cfg := config{model: "test-model", maxTokens: 128, maxIters: 5}
	_, _, err := runPrompt(context.Background(), sender, cfg, nil, "hi",
		func(format string, args ...any) {})
	if err == nil || !strings.Contains(err.Error(), "reviewer unavailable") {
		t.Fatalf("expected reviewer outage error, got %v", err)
	}
}

func TestReviewVerdictParse(t *testing.T) {
	for _, tc := range []struct {
		in       string
		approved bool
		want     string
	}{
		{"APPROVE: safe", true, "safe"},
		{"approve looks fine", true, "looks fine"},
		{"APPROVE", true, ""},
		{"DENY: destructive", false, "destructive"},
		{"deny - risky", false, "risky"},
		{"DENY", false, ""},
		{"APPROVEDLY", false, "no clear verdict"},
		{"maybe later", false, "no clear verdict"},
		{"", false, "no clear verdict"},
	} {
		approved, rationale := reviewVerdict(tc.in)
		if approved != tc.approved || !strings.Contains(rationale, tc.want) {
			t.Errorf("reviewVerdict(%q) = (%v, %q), want (%v, containing %q)",
				tc.in, approved, rationale, tc.approved, tc.want)
		}
	}
}

func TestRenderTranscript(t *testing.T) {
	history := []message{
		textMessage("user", "list files"),
		blocksMessage("assistant", []contentBlock{
			{Type: "text", Text: "on it"},
			{Type: "tool_use", ID: "c1", Name: "bash", Input: json.RawMessage(`{"command":"ls"}`)},
		}),
		blocksMessage("user", []contentBlock{
			{Type: "tool_result", ToolUseID: "c1", Content: "main.go"},
		}),
	}
	got := renderTranscript(history)
	for _, want := range []string{"user: list files", "assistant: on it", "called bash", `"ls"`, "result: main.go"} {
		if !strings.Contains(got, want) {
			t.Errorf("transcript missing %q:\n%s", want, got)
		}
	}
	long := []message{textMessage("user", strings.Repeat("x", reviewTranscriptTotal+100))}
	if got := renderTranscript(long); len(got) > reviewTranscriptTotal+reviewTranscriptChars {
		t.Errorf("transcript must stay compact: %d bytes", len(got))
	}
}
