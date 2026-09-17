package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

// fakeSender scripts model replies so the agent loop is testable without
// network access. All calls happen on the test goroutine.
type fakeSender struct {
	t        *testing.T
	requests []messageRequest
	script   func(call int, req messageRequest) messageResponse
}

func (f *fakeSender) createMessage(_ context.Context, req messageRequest) (*messageResponse, error) {
	f.requests = append(f.requests, req)
	resp := f.script(len(f.requests), req)
	return &resp, nil
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

	cfg := config{model: "test-model", maxTokens: 128, maxIters: 5, skipConfirm: true}
	answer, history, err := runPrompt(context.Background(), sender, cfg, nil, "hi",
		func(tool, summary string) bool { return true },
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

// TestAgentLoopDeniesWithoutConfirm verifies a denied tool call is reported
// back to the model instead of executed.
func TestAgentLoopDeniesWithoutConfirm(t *testing.T) {
	sender := &fakeSender{t: t}
	sender.script = func(call int, req messageRequest) messageResponse {
		if call == 1 {
			return messageResponse{
				ID: "msg_1",
				Content: []contentBlock{
					{Type: "tool_use", ID: "toolu_9", Name: "bash",
						Input: json.RawMessage(`{"command":"touch /tmp/harnais-must-not-exist"}`)},
				},
				StopReason: "tool_use",
			}
		}
		if blob := requestJSON(t, req); !strings.Contains(blob, "denied by user") {
			t.Errorf("expected denial in tool result: %s", blob)
		}
		return messageResponse{
			ID:         "msg_2",
			Content:    []contentBlock{{Type: "text", Text: "stood down"}},
			StopReason: "end_turn",
		}
	}

	cfg := config{model: "test-model", maxTokens: 128, maxIters: 5}
	answer, _, err := runPrompt(context.Background(), sender, cfg, nil, "hi",
		func(tool, summary string) bool { return false },
		func(format string, args ...any) {})
	if err != nil {
		t.Fatalf("runPrompt: %v", err)
	}
	if answer != "stood down" {
		t.Fatalf("unexpected answer: %q", answer)
	}
}
