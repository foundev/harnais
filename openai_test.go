package main

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestOpenRouterRequestTranslation(t *testing.T) {
	req := messageRequest{
		Model:     "anthropic/claude-sonnet-4.5",
		MaxTokens: 128,
		System:    "be brief",
		Messages: []message{
			textMessage("user", "list files"),
			blocksMessage("assistant", []contentBlock{
				{Type: "text", Text: "on it"},
				{Type: "tool_use", ID: "call_1", Name: "bash",
					Input: json.RawMessage(`{"command":"ls"}`)},
			}),
			blocksMessage("user", []contentBlock{
				{Type: "tool_result", ToolUseID: "call_1", Content: "main.go"},
			}),
		},
		Tools: toolDefinitions(),
	}
	chat := toChatRequest(req)

	if chat.Model != req.Model || chat.MaxTokens != req.MaxTokens {
		t.Fatalf("model/tokens not carried over: %+v", chat)
	}
	if len(chat.Messages) != 4 {
		t.Fatalf("expected 4 chat messages, got %d: %+v", len(chat.Messages), chat.Messages)
	}
	if chat.Messages[0].Role != "system" || chat.Messages[0].Content != "be brief" {
		t.Errorf("system message wrong: %+v", chat.Messages[0])
	}
	assistant := chat.Messages[2]
	if assistant.Role != "assistant" || assistant.Content != "on it" {
		t.Errorf("assistant text wrong: %+v", assistant)
	}
	if len(assistant.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %+v", assistant.ToolCalls)
	}
	call := assistant.ToolCalls[0]
	if call.ID != "call_1" || call.Type != "function" || call.Function.Name != "bash" ||
		call.Function.Arguments != `{"command":"ls"}` {
		t.Errorf("tool call wrong: %+v", call)
	}
	toolMsg := chat.Messages[3]
	if toolMsg.Role != "tool" || toolMsg.ToolCallID != "call_1" || toolMsg.Content != "main.go" {
		t.Errorf("tool result message wrong: %+v", toolMsg)
	}
	if len(chat.Tools) != 4 || chat.Tools[0].Type != "function" || chat.Tools[0].Function.Name != "bash" {
		t.Fatalf("tools not converted: %+v", chat.Tools)
	}
	if chat.Tools[0].Function.Parameters["type"] != "object" {
		t.Errorf("tool schema not carried over: %+v", chat.Tools[0].Function.Parameters)
	}
}

func TestOpenRouterResponseWithTools(t *testing.T) {
	raw := []byte(`{"id":"gen_1","choices":[{"message":{"role":"assistant","content":"running","tool_calls":[{"id":"call_2","type":"function","function":{"name":"read","arguments":"{\"path\":\"main.go\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}`)
	resp, err := decodeChatResponse(200, raw)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if resp.StopReason != "tool_use" {
		t.Errorf("finish_reason tool_calls must continue the loop, got %q", resp.StopReason)
	}
	if len(resp.Content) != 2 || resp.Content[0].Text != "running" {
		t.Fatalf("text block wrong: %+v", resp.Content)
	}
	use := resp.Content[1]
	if use.Type != "tool_use" || use.ID != "call_2" || use.Name != "read" ||
		string(use.Input) != `{"path":"main.go"}` {
		t.Errorf("tool_use block wrong: %+v", use)
	}
	if resp.Usage.InputTokens != 10 || resp.Usage.OutputTokens != 5 {
		t.Errorf("usage wrong: %+v", resp.Usage)
	}
}

func TestOpenRouterResponseFinalText(t *testing.T) {
	raw := []byte(`{"id":"gen_2","choices":[{"message":{"role":"assistant","content":"done"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	resp, err := decodeChatResponse(200, raw)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if resp.StopReason != "end_turn" || responseText(resp) != "done" {
		t.Errorf("final answer wrong: %+v", resp)
	}
}

func TestOpenRouterResponseErrors(t *testing.T) {
	if _, err := decodeChatResponse(401, []byte(`{"error":{"message":"bad key"}}`)); err == nil ||
		!strings.Contains(err.Error(), "401") {
		t.Errorf("expected status error, got %v", err)
	}
	if _, err := decodeChatResponse(200, []byte(`{"error":{"message":"overloaded"}}`)); err == nil ||
		!strings.Contains(err.Error(), "overloaded") {
		t.Errorf("expected body error, got %v", err)
	}
	if _, err := decodeChatResponse(200, []byte(`{"id":"x","choices":[]}`)); err == nil {
		t.Error("expected error for empty choices")
	}
}

func TestResolveBackend(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "ak")
	t.Setenv("OPENROUTER_API_KEY", "ok")

	sender, model, err := resolveBackend("anthropic", "", "")
	if err != nil {
		t.Fatalf("anthropic: %v", err)
	}
	if _, ok := sender.(*anthropicClient); !ok {
		t.Errorf("expected *anthropicClient, got %T", sender)
	}
	if model != "claude-sonnet-4-20250514" {
		t.Errorf("unexpected anthropic default model %q", model)
	}

	sender, model, err = resolveBackend("openrouter", "", "")
	if err != nil {
		t.Fatalf("openrouter: %v", err)
	}
	if _, ok := sender.(*openAICompatClient); !ok {
		t.Errorf("expected *openAICompatClient, got %T", sender)
	}
	if model != defaultOpenRouterModel {
		t.Errorf("unexpected openrouter default model %q", model)
	}

	_, model, err = resolveBackend("openrouter", "flag-key", "my/model")
	if err != nil {
		t.Fatalf("explicit values: %v", err)
	}
	if model != "my/model" {
		t.Errorf("flag model not honored, got %q", model)
	}

	t.Setenv("DEEPSEEK_API_KEY", "dk")
	sender, model, err = resolveBackend("deepseek", "", "")
	if err != nil {
		t.Fatalf("deepseek: %v", err)
	}
	if _, ok := sender.(*openAICompatClient); !ok {
		t.Errorf("expected *openAICompatClient, got %T", sender)
	}
	if model != defaultDeepSeekModel {
		t.Errorf("unexpected deepseek default model %q", model)
	}

	t.Setenv("OPENROUTER_API_KEY", "")
	if _, _, err := resolveBackend("openrouter", "", ""); err == nil {
		t.Error("expected missing-key error for openrouter")
	}
	t.Setenv("DEEPSEEK_API_KEY", "")
	if _, _, err := resolveBackend("deepseek", "", ""); err == nil {
		t.Error("expected missing-key error for deepseek")
	}
	if _, _, err := resolveBackend("nope", "k", "m"); err == nil {
		t.Error("expected error for unknown provider")
	}
}
