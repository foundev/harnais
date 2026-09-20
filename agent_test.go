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

	cfg := config{model: "test-model"}
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

	cfg := config{model: "test-model"}
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
	cfg := config{model: "test-model", sandbox: true}
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

	cfg := config{model: "test-model"}
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

	cfg := config{model: "test-model"}
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

	cfg := config{model: "test-model"}
	_, _, err := runPrompt(context.Background(), sender, cfg, nil, "hi",
		func(format string, args ...any) {})
	if err == nil || !strings.Contains(err.Error(), "reviewer unavailable") {
		t.Fatalf("expected reviewer outage error, got %v", err)
	}
}

// TestTurnRunsUntilTheModelStops: there is no step budget any more, so the
// only thing that ends a turn is the model deciding it is done (or an
// error, or the reviewer's circuit breaker).
func TestTurnRunsUntilTheModelStops(t *testing.T) {
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

	cfg := config{model: "test-model"}
	answer, _, err := runPrompt(context.Background(), sender, cfg, nil, "go",
		func(format string, args ...any) {})
	if err != nil || answer != "finally done" {
		t.Fatalf("the loop must run to the model's own stop: answer=%q err=%v", answer, err)
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
	// All plain user prompts are retained for the reviewer, in order —
	// the goal-supersede notice is steering, not an instruction.
	history = append(history, textMessage("user", goalUpdatedNotice("also update the footer")))
	want := "user instructions, in order (later ones supersede earlier ones):\n" +
		"fix the typo on the homepage\nalso update the footer"
	if got := reviewGoal(history, ""); got != want {
		t.Errorf("reviewer must retain every user instruction in order, got %q", got)
	}
	if got := reviewGoal(nil, ""); got != "(none stated)" {
		t.Errorf("empty history needs a fallback, got %q", got)
	}
	if got := reviewGoal(history, "roll back the migration"); got != "roll back the migration" {
		t.Errorf("an explicit /goal must win over history, got %q", got)
	}
	long := []message{textMessage("user", strings.Repeat("y", reviewTranscriptChars+10))}
	if got := reviewGoal(long, ""); len(got) > reviewTranscriptChars+10 {
		t.Errorf("goal must stay compact: %d bytes", len(got))
	}
}

// TestReviewRetainsAllUserInstructions verifies the reviewer request keeps
// every plain user prompt — including the original, after tool traffic has
// pushed it out of the transcript window — in order, with the supersede
// rule. Tool-result messages are blocks, not instructions, and the
// goal-supersede notice is steering, so neither counts.
func TestReviewRetainsAllUserInstructions(t *testing.T) {
	history := []message{textMessage("user", "migrate the database")}
	for i := 0; i < reviewTranscriptMessages+2; i++ {
		history = append(history, blocksMessage("user", []contentBlock{
			{Type: "tool_result", ToolUseID: fmt.Sprintf("c%d", i), Content: fmt.Sprintf("filler %d", i)},
		}))
	}
	history = append(history, textMessage("user", "inventory the replicas"))
	history = append(history, textMessage("user", goalUpdatedNotice("roll back the migration")))
	history = append(history, textMessage("user", "roll back the migration now"))
	sender := &fakeSender{t: t}
	sender.script = func(call int, req messageRequest) messageResponse {
		return reviewVerdictResponse("APPROVE: on task")
	}
	cfg := config{model: "test-model", goal: "roll back the migration"}
	approved, _, _, err := reviewToolCall(context.Background(), sender, cfg, history,
		"bash", map[string]any{"command": "echo hi"})
	if err != nil || !approved {
		t.Fatalf("reviewToolCall = (%v, %v), want approval", approved, err)
	}
	blob := requestJSON(t, sender.requests[0])
	if !strings.Contains(blob, "Current task: roll back the migration") {
		t.Errorf("explicit goal must pin the task line:\n%s", blob)
	}
	if strings.Contains(blob, "user instructions, in order") {
		t.Errorf("an explicit goal replaces the instruction list:\n%s", blob)
	}
	// Without an explicit goal the retained list carries every prompt.
	cfg.goal = ""
	approved, _, _, err = reviewToolCall(context.Background(), sender, cfg, history,
		"bash", map[string]any{"command": "echo hi"})
	if err != nil || !approved {
		t.Fatalf("reviewToolCall = (%v, %v), want approval", approved, err)
	}
	blob = requestJSON(t, sender.requests[1])
	want := "user instructions, in order (later ones supersede earlier ones):\\n" +
		"migrate the database\\ninventory the replicas\\nroll back the migration now"
	if !strings.Contains(blob, want) {
		t.Errorf("review request must retain all user instructions in order:\\n%s", blob)
	}
	// The notice stays visible in the transcript (the model saw it), but
	// must not join the retained instruction list.
	if strings.Contains(blob, "earlier ones):\\nmigrate the database\\nThe active session goal") {
		t.Errorf("the goal-supersede notice must not join the instruction list:\\n%s", blob)
	}
}

// TestReviewPrefersExplicitGoal verifies an explicit /goal objective pins
// the reviewer's task even when older prompts surround it in history.
func TestReviewPrefersExplicitGoal(t *testing.T) {
	history := []message{textMessage("user", "migrate the database")}
	sender := &fakeSender{t: t}
	sender.script = func(call int, req messageRequest) messageResponse {
		return reviewVerdictResponse("APPROVE: on task")
	}
	cfg := config{model: "test-model", goal: "roll back the migration"}
	approved, _, _, err := reviewToolCall(context.Background(), sender, cfg, history,
		"bash", map[string]any{"command": "echo hi"})
	if err != nil || !approved {
		t.Fatalf("reviewToolCall = (%v, %v), want approval", approved, err)
	}
	blob := requestJSON(t, sender.requests[0])
	if !strings.Contains(blob, "Current task: roll back the migration") {
		t.Errorf("review request must pin the explicit goal:\n%s", blob)
	}
	if strings.Contains(blob, "Current task: migrate the database") {
		t.Errorf("an explicit goal must replace history-derived tasks:\n%s", blob)
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

// TestGoalLoopContinuesUntilComplete verifies the Codex continue_if_idle
// contract: with an active goal, an ended turn is followed by a
// continuation seeded with the goal, and only update_goal with
// complete/blocked ends the loop. The goal turn also advertises
// update_goal and drops it once the loop exits.
func TestGoalLoopContinuesUntilComplete(t *testing.T) {
	sender := &fakeSender{t: t}
	agentCalls := 0
	sender.script = func(call int, req messageRequest) messageResponse {
		if len(req.Tools) == 0 {
			return reviewVerdictResponse("APPROVE: harmless")
		}
		agentCalls++
		names := advertisedTools(req)
		switch agentCalls {
		case 1:
			if !strings.Contains(names, "update_goal") {
				t.Errorf("goal turn must advertise update_goal: %s", names)
			}
			return messageResponse{
				ID: "msg_1",
				Content: []contentBlock{
					{Type: "text", Text: "halfway there"},
					{Type: "tool_use", ID: "toolu_1", Name: "bash",
						Input: json.RawMessage(`{"command":"echo progress"}`)},
				},
				StopReason: "tool_use",
			}
		case 2:
			if !strings.Contains(names, "update_goal") {
				t.Errorf("update_goal must stay advertised while the goal is active: %s", names)
			}
			return messageResponse{
				ID: "msg_2", Content: []contentBlock{{Type: "text", Text: "turn over, but the goal is not done"}},
				StopReason: "end_turn",
			}
		case 3:
			// The continuation must carry the goal steering and the
			// earlier answer, so the model sees where it left off.
			blob := requestJSON(t, req)
			if !strings.Contains(blob, "Continue working toward the active session goal") ||
				!strings.Contains(blob, "turn over, but the goal is not done") {
				t.Errorf("continuation missing goal steering or prior turn:\n%s", blob)
			}
			return messageResponse{
				ID: "msg_3",
				Content: []contentBlock{
					{Type: "text", Text: "finished and verified"},
					{Type: "tool_use", ID: "toolu_2", Name: "update_goal",
						Input: json.RawMessage(`{"status":"complete"}`)},
				},
				StopReason: "tool_use",
			}
		default:
			return messageResponse{
				ID: "msg_4", Content: []contentBlock{{Type: "text", Text: "goal complete summary"}},
				StopReason: "end_turn",
			}
		}
	}
	cfg := config{model: "test-model", goal: "ship the feature"}
	answer, history, err := runPrompt(context.Background(), sender, cfg, nil, "start the work",
		func(string, ...any) {})
	if err != nil {
		t.Fatalf("runPrompt: %v", err)
	}
	if answer != "goal complete summary" {
		t.Fatalf("answer after update_goal = %q", answer)
	}
	if agentCalls != 4 {
		t.Errorf("expected 4 agent model calls (tool, end, continuation+update_goal, summary), got %d", agentCalls)
	}
	if len(history) == 0 {
		t.Fatal("history must be returned")
	}
}

// TestGoalLoopBlockedAudit verifies Codex's blocked audit: update_goal
// "blocked" is rejected by the host before the same impasse has survived
// goalTurnsToBlock consecutive goal turns (a turn that ran no tools), and
// accepted once it has. A successful tool run resets the counter.
func TestGoalLoopBlockedAudit(t *testing.T) {
	sender := &fakeSender{t: t}
	agentCalls := 0
	sender.script = func(call int, req messageRequest) messageResponse {
		if len(req.Tools) == 0 {
			return reviewVerdictResponse("APPROVE: harmless")
		}
		agentCalls++
		blob := requestJSON(t, req)
		if !strings.Contains(blob, "Continue working toward the active session goal") {
			t.Errorf("goal steering must be present from the first turn")
		}
		switch agentCalls {
		case 1: // impasse turn 1
			return endTurn("stuck on the missing credentials")
		case 2: // premature blocked — 1 < 3, rejected (the attempt is itself a no-progress turn)
			return blockedCall()
		case 3: // impasse turn 2; the rejection must be in context
			if !strings.Contains(blob, "blocked requires the same blocker") {
				t.Errorf("rejection must reach the model")
			}
			return endTurn("still stuck on the missing credentials")
		case 4: // 3 >= 3: blocked is now accepted
			return messageResponse{
				ID: "msg_4",
				Content: []contentBlock{
					{Type: "text", Text: "cannot proceed"},
					{Type: "tool_use", ID: "toolu_1", Name: "update_goal",
						Input: json.RawMessage(`{"status":"blocked"}`)},
				},
				StopReason: "tool_use",
			}
		default: // summary after the loop exits
			return endTurn("blocked summary")
		}
	}
	cfg := config{model: "test-model", goal: "needs external credentials"}
	answer, _, err := runPrompt(context.Background(), sender, cfg, nil, "go",
		func(string, ...any) {})
	if err != nil {
		t.Fatalf("runPrompt: %v", err)
	}
	if answer != "blocked summary" {
		t.Fatalf("answer after accepted blocked = %q", answer)
	}
	if agentCalls != 5 {
		t.Errorf("expected 5 agent model calls, got %d", agentCalls)
	}
}

func endTurn(text string) messageResponse {
	return messageResponse{
		ID: "msg_end", Content: []contentBlock{{Type: "text", Text: text}},
		StopReason: "end_turn",
	}
}

func blockedCall() messageResponse {
	return messageResponse{
		ID: "msg_blocked",
		Content: []contentBlock{
			{Type: "text", Text: "giving up"},
			{Type: "tool_use", ID: "toolu_b", Name: "update_goal",
				Input: json.RawMessage(`{"status":"blocked"}`)},
		},
		StopReason: "tool_use",
	}
}

// TestGoalProgressResetsBlockedAudit verifies a successful tool run breaks
// the impasse streak: blocked is rejected again after progress.
func TestGoalProgressResetsBlockedAudit(t *testing.T) {
	sender := &fakeSender{t: t}
	agentCalls := 0
	sender.script = func(call int, req messageRequest) messageResponse {
		if len(req.Tools) == 0 {
			return reviewVerdictResponse("APPROVE: harmless")
		}
		agentCalls++
		switch agentCalls {
		case 1: // impasse turn 1
			return endTurn("stuck")
		case 2: // premature blocked — rejected
			return blockedCall()
		case 3: // progress: a tool actually runs, streak resets to 0
			return messageResponse{
				ID: "msg_3",
				Content: []contentBlock{
					{Type: "tool_use", ID: "toolu_p", Name: "bash",
						Input: json.RawMessage(`{"command":"echo found a path"}`)},
				},
				StopReason: "tool_use",
			}
		case 4: // impasse turn 1 again
			return endTurn("stuck again")
		case 5: // rejected — the streak restarted (1 < 3)
			return blockedCall()
		case 6: // impasse turn 2
			return endTurn("same blocker returns")
		case 7: // 3 >= 3: accepted
			return messageResponse{
				ID: "msg_7",
				Content: []contentBlock{
					{Type: "text", Text: "truly at an impasse"},
					{Type: "tool_use", ID: "toolu_2", Name: "update_goal",
						Input: json.RawMessage(`{"status":"blocked"}`)},
				},
				StopReason: "tool_use",
			}
		default:
			return endTurn("gave up after the full audit")
		}
	}
	cfg := config{model: "test-model", goal: "hard goal"}
	answer, _, err := runPrompt(context.Background(), sender, cfg, nil, "go", func(string, ...any) {})
	if err != nil {
		t.Fatalf("runPrompt: %v", err)
	}
	if answer != "gave up after the full audit" {
		t.Fatalf("answer after post-reset blocked = %q", answer)
	}
	if agentCalls != 8 {
		t.Errorf("expected 8 agent model calls, got %d", agentCalls)
	}
}

func TestNoGoalNoUpdateGoalTool(t *testing.T) {
	sender := &fakeSender{t: t}
	turns := 0
	sender.script = func(call int, req messageRequest) messageResponse {
		if len(req.Tools) == 0 {
			return reviewVerdictResponse("APPROVE: harmless")
		}
		turns++
		if strings.Contains(advertisedTools(req), "update_goal") {
			t.Errorf("update_goal must not be advertised without a goal")
		}
		if strings.Contains(requestJSON(t, req), "Continue working toward the active session goal") {
			t.Errorf("continuation steering must not appear without a goal")
		}
		return messageResponse{
			ID: fmt.Sprintf("msg_%d", call), Content: []contentBlock{{Type: "text", Text: "done"}},
			StopReason: "end_turn",
		}
	}
	cfg := config{model: "test-model"}
	answer, _, err := runPrompt(context.Background(), sender, cfg, nil, "hi", func(string, ...any) {})
	if err != nil {
		t.Fatalf("runPrompt: %v", err)
	}
	if answer != "done" || turns != 1 {
		t.Fatalf("plain turn changed: %q after %d model calls", answer, turns)
	}
}
