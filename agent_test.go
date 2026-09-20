package main

import (
	"context"
	"encoding/json"
	"errors"
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

// TestReviewerEscalationLiftsSandbox verifies APPROVE UNSANDBOXED runs one
// bash call outside the OS sandbox. runBash marks sandboxed runs with
// HARNAIS_SANDBOX=1, so `printenv` distinguishes: sandboxed prints 1,
// escalated prints nothing. (On platforms without sandbox-exec every run
// is unsandboxed, so the output assertion is vacuous there and the
// progress-line assertion carries the wiring check.)
func TestReviewerEscalationLiftsSandbox(t *testing.T) {
	sender := &fakeSender{t: t}
	sender.script = func(call int, req messageRequest) messageResponse {
		if len(req.Tools) == 0 {
			return reviewVerdictResponse("APPROVE UNSANDBOXED: test needs the real environment")
		}
		return proposeEcho(t, "printenv HARNAIS_SANDBOX", "escalated it")(call, req)
	}

	var reports []string
	report := func(format string, args ...any) {
		reports = append(reports, fmt.Sprintf(format, args...))
	}
	cfg := config{model: "test-model", maxTokens: 128, maxIters: 5, sandbox: true}
	answer, history, err := runPrompt(context.Background(), sender, cfg, nil, "hi", report)
	if err != nil {
		t.Fatalf("runPrompt: %v", err)
	}
	if answer != "escalated it" {
		t.Fatalf("unexpected answer: %q", answer)
	}
	found := false
	for _, r := range reports {
		if strings.Contains(r, "escalated outside the sandbox") {
			found = true
		}
	}
	if !found {
		t.Errorf("escalation was not reported: %q", reports)
	}
	sawResult := false
	for _, msg := range history {
		var blocks []contentBlock
		if err := json.Unmarshal(msg.Content, &blocks); err != nil {
			continue
		}
		for _, b := range blocks {
			if b.Type != "tool_result" {
				continue
			}
			sawResult = true
			// Escalated, the marker var is absent so printenv finds
			// nothing; sandboxed, it prints 1 — or fails with the
			// sandbox hint where nested sandboxing is denied.
			if !strings.Contains(b.Content, "(no output)") || strings.Contains(b.Content, "sandboxed: denied") {
				t.Errorf("escalated call still ran sandboxed, output:\n%s", b.Content)
			}
		}
	}
	if !sawResult {
		t.Error("no tool result in history")
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

// TestIterationCapIsASoftStop pins the contract that matters for long
// tasks: running out of budget hands the turn's work back instead of
// throwing it away, and a follow-up prompt resumes from it.
func TestIterationCapIsASoftStop(t *testing.T) {
	readPath := filepath.Join(t.TempDir(), "notes.txt")
	if err := os.WriteFile(readPath, []byte("the answers are here\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	sender := &fakeSender{t: t}
	sender.script = func(call int, req messageRequest) messageResponse {
		if len(req.Tools) == 0 {
			return reviewVerdictResponse("APPROVE")
		}
		// Keep asking for a tool the reviewer never has to approve.
		return messageResponse{
			ID: fmt.Sprintf("msg_%d", call),
			Content: []contentBlock{
				{Type: "tool_use", ID: fmt.Sprintf("toolu_%d", call), Name: "read",
					Input: json.RawMessage(`{"path":` + strconv.Quote(readPath) + `}`)},
			},
			StopReason: "tool_use",
		}
	}

	cfg := config{model: "test-model", maxTokens: 128, maxIters: 3}
	answer, history, err := runPrompt(context.Background(), sender, cfg, nil, "keep reading",
		func(format string, args ...any) {})
	var capErr iterationCapError
	if !errors.As(err, &capErr) {
		t.Fatalf("expected the iteration cap, got %v", err)
	}
	if answer != "" {
		t.Errorf("a capped turn has no answer, got %q", answer)
	}
	if !strings.Contains(capErr.Error(), "carry on") || !strings.Contains(capErr.Error(), "-n") {
		t.Errorf("the cap must tell the user how to continue: %q", capErr.Error())
	}
	// The three tool results are in the history, so nothing was thrown away.
	if got := strings.Count(requestJSON(t, messageRequest{Messages: history}), "the answers are here"); got != 3 {
		t.Fatalf("capped turn must keep its tool results, found %d in history", got)
	}

	// The next prompt resumes: the model sees the earlier reads.
	sender.script = func(call int, req messageRequest) messageResponse {
		if len(req.Tools) == 0 {
			return reviewVerdictResponse("APPROVE")
		}
		if blob := requestJSON(t, req); !strings.Contains(blob, "the answers are here") {
			t.Errorf("resumed turn must carry the earlier tool results: %s", blob)
		}
		return messageResponse{
			ID:         "msg_done",
			Content:    []contentBlock{{Type: "text", Text: "picked up where we left off"}},
			StopReason: "end_turn",
		}
	}
	answer, _, err = runPrompt(context.Background(), sender, cfg, history, "carry on",
		func(format string, args ...any) {})
	if err != nil || answer != "picked up where we left off" {
		t.Fatalf("resumed turn failed: answer=%q err=%v", answer, err)
	}
}

func TestTruncatedAnswerIsReported(t *testing.T) {
	// Anthropic says stop_reason "max_tokens" and OpenAI-compatible backends
	// say finish_reason "length" when the model was cut off mid-answer. The
	// docs put it plainly for thinking models: "a long thinking pass can
	// consume the budget before the text response completes". The answer
	// still comes back, but it must never come back silently short.
	for _, tc := range []struct {
		name    string
		reason  string
		cap     int
		wantSub string
	}{
		{"anthropic at the user's cap", "max_tokens", 4096, "-max-tokens cap (4096)"},
		{"openai-style at the model's limit", "length", 0, "model's own output limit"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sender := &fakeSender{t: t}
			sender.script = func(int, messageRequest) messageResponse {
				return messageResponse{
					ID:         "msg_trunc",
					Content:    []contentBlock{{Type: "text", Text: "the answer so far"}},
					StopReason: tc.reason,
				}
			}
			var lines []string
			answer, _, err := runPrompt(context.Background(), sender,
				config{model: "m", maxTokens: tc.cap}, nil, "write something long",
				func(format string, args ...any) { lines = append(lines, fmt.Sprintf(format, args...)) })
			if err != nil {
				t.Fatalf("a truncated answer is not a failure: %v", err)
			}
			if answer != "the answer so far" {
				t.Errorf("partial answer must survive, got %q", answer)
			}
			if joined := strings.Join(lines, "\n"); !strings.Contains(joined, tc.wantSub) {
				t.Errorf("truncation must be reported (%q missing):\n%s", tc.wantSub, joined)
			}
		})
	}
}

// TestZeroMaxItersMeansNoCap: the loop's only stop with -n 0 is the model
// deciding it is done.
func TestZeroMaxItersMeansNoCap(t *testing.T) {
	readPath := filepath.Join(t.TempDir(), "notes.txt")
	if err := os.WriteFile(readPath, []byte("still going\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	const rounds = 40
	sender := &fakeSender{t: t}
	sender.script = func(call int, req messageRequest) messageResponse {
		if len(req.Tools) == 0 {
			return reviewVerdictResponse("APPROVE")
		}
		if call <= rounds {
			return messageResponse{
				ID: fmt.Sprintf("msg_%d", call),
				Content: []contentBlock{
					{Type: "tool_use", ID: fmt.Sprintf("toolu_%d", call), Name: "read",
						Input: json.RawMessage(`{"path":` + strconv.Quote(readPath) + `}`)},
				},
				StopReason: "tool_use",
			}
		}
		return messageResponse{
			ID:         "msg_done",
			Content:    []contentBlock{{Type: "text", Text: "finally done"}},
			StopReason: "end_turn",
		}
	}

	cfg := config{model: "test-model", maxTokens: 128} // maxIters 0 = no cap
	answer, _, err := runPrompt(context.Background(), sender, cfg, nil, "go",
		func(format string, args ...any) {})
	if err != nil || answer != "finally done" {
		t.Fatalf("an uncapped loop must run to the model's own stop: answer=%q err=%v", answer, err)
	}
	if len(sender.requests) != rounds+1 {
		t.Errorf("expected %d model calls, got %d", rounds+1, len(sender.requests))
	}
}

func TestReviewVerdictParse(t *testing.T) {
	for _, tc := range []struct {
		in        string
		approved  bool
		escalated bool
		want      string
	}{
		{"APPROVE: safe", true, false, "safe"},
		{"approve looks fine", true, false, "looks fine"},
		{"APPROVE", true, false, ""},
		{"APPROVE UNSANDBOXED: needs network", true, true, "needs network"},
		{"approve unsandboxed fetch a module", true, true, "fetch a module"},
		{"APPROVE-UNSANDBOXED: outside files", true, true, "outside files"},
		{"APPROVE UNSANDBOXED", true, true, ""},
		{"DENY: destructive", false, false, "destructive"},
		{"deny - risky", false, false, "risky"},
		{"DENY", false, false, ""},
		{"APPROVEDLY", false, false, "no clear verdict"},
		{"APPROVE UNSANDBOXEDLY: typo", false, false, "no clear verdict"},
		{"maybe later", false, false, "no clear verdict"},
		{"", false, false, "no clear verdict"},
	} {
		approved, escalated, rationale := reviewVerdict(tc.in)
		if approved != tc.approved || escalated != tc.escalated || !strings.Contains(rationale, tc.want) {
			t.Errorf("reviewVerdict(%q) = (%v, %v, %q), want (%v, %v, containing %q)",
				tc.in, approved, escalated, rationale, tc.approved, tc.escalated, tc.want)
		}
	}
}

func TestSandboxForCall(t *testing.T) {
	for _, tc := range []struct {
		sandbox, escalated bool
		tool               string
		want               bool
	}{
		{true, false, "bash", true},
		{true, true, "bash", false},
		{false, true, "bash", false},
		{false, false, "bash", false},
		// Escalation is meaningless off bash: read/edit/write never run
		// under sandbox-exec, so the flag is ignored, never widening.
		{true, true, "edit", true},
		{true, true, "write", true},
		{true, true, "read", true},
	} {
		if got := sandboxForCall(tc.sandbox, tc.tool, tc.escalated); got != tc.want {
			t.Errorf("sandboxForCall(%v, %q, %v) = %v, want %v",
				tc.sandbox, tc.tool, tc.escalated, got, tc.want)
		}
	}
}

func TestReviewGoal(t *testing.T) {
	history := []message{
		textMessage("user", "fix the typo on the homepage"),
		blocksMessage("assistant", []contentBlock{{Type: "text", Text: "on it"}}),
		blocksMessage("user", []contentBlock{
			{Type: "tool_result", ToolUseID: "c1", Content: "some output"},
		}),
		textMessage("user", "also update the footer"),
	}
	if got := reviewGoal(history); got != "fix the typo on the homepage" {
		t.Errorf("goal must be the first user text, got %q", got)
	}
	if got := reviewGoal(nil); got != "(none stated)" {
		t.Errorf("empty history needs a fallback, got %q", got)
	}
	long := []message{textMessage("user", strings.Repeat("y", reviewTranscriptChars+10))}
	if got := reviewGoal(long); len(got) > reviewTranscriptChars+10 {
		t.Errorf("goal must stay compact: %d bytes", len(got))
	}
}

// TestReviewSeesOriginalTask verifies the reviewer request pins the session
// goal even after tool traffic pushes it out of the transcript window.
func TestReviewSeesOriginalTask(t *testing.T) {
	history := []message{textMessage("user", "migrate the database")}
	for i := 0; i < reviewTranscriptMessages+2; i++ {
		history = append(history, textMessage("user", fmt.Sprintf("filler %d", i)))
	}
	sender := &fakeSender{t: t}
	sender.script = func(call int, req messageRequest) messageResponse {
		return reviewVerdictResponse("APPROVE: on task")
	}
	cfg := config{model: "test-model", maxTokens: 128}
	approved, _, _, err := reviewToolCall(context.Background(), sender, cfg, history,
		"bash", map[string]any{"command": "echo hi"})
	if err != nil || !approved {
		t.Fatalf("reviewToolCall = (%v, %v), want approval", approved, err)
	}
	blob := requestJSON(t, sender.requests[0])
	if !strings.Contains(blob, "Original task: migrate the database") {
		t.Errorf("review request must pin the session goal:\n%s", blob)
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
