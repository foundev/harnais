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

func TestBuildChatRequestEffort(t *testing.T) {
	req := messageRequest{Model: "m", Effort: "high", Messages: []message{textMessage("user", "hi")}}

	chat := newOpenAIClient("k", "http://x").buildChatRequest(req)
	if chat.ReasoningEffort == nil || *chat.ReasoningEffort != "high" {
		t.Errorf("openai client should keep reasoning_effort: %+v", chat.ReasoningEffort)
	}
	chat = newOpenRouterClient("k", "http://x").buildChatRequest(req)
	if chat.ReasoningEffort == nil || *chat.ReasoningEffort != "high" {
		t.Errorf("openrouter client should keep reasoning_effort: %+v", chat.ReasoningEffort)
	}
	chat = newDeepSeekClient("k", "http://x").buildChatRequest(req)
	if chat.ReasoningEffort == nil || *chat.ReasoningEffort != "high" {
		t.Errorf("deepseek client should keep reasoning_effort: %+v", chat.ReasoningEffort)
	}
	chat = newOpenAIClient("k", "http://x").buildChatRequest(messageRequest{Model: "m"})
	if chat.ReasoningEffort != nil {
		t.Errorf("unset effort should send nothing: %+v", chat.ReasoningEffort)
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

// TestChatRequestSkipsEmptyAssistant guards the DeepSeek 400
// ("Invalid assistant message: content or tool_calls must be set"): an empty
// final answer stored in history must not be sent back as a content-less
// assistant message on the next turn.
func TestChatRequestSkipsEmptyAssistant(t *testing.T) {
	req := messageRequest{
		Model:  "deepseek-chat",
		System: "s",
		Messages: []message{
			textMessage("user", "do thing"),
			blocksMessage("assistant", []contentBlock{
				{Type: "tool_use", ID: "call_1", Name: "bash",
					Input: json.RawMessage(`{"command":"ls"}`)},
			}),
			blocksMessage("user", []contentBlock{
				{Type: "tool_result", ToolUseID: "call_1", Content: "main.go"},
			}),
			blocksMessage("assistant", nil), // empty final answer
			textMessage("user", "what did you do?"),
		},
		Tools: toolDefinitions(),
	}
	chat := toChatRequest(req)
	raw, _ := json.Marshal(chat)
	var wire struct {
		Messages []chatMessage `json:"messages"`
	}
	if err := json.Unmarshal(raw, &wire); err != nil {
		t.Fatal(err)
	}
	for _, m := range wire.Messages {
		if m.Role == "assistant" && strings.TrimSpace(m.Content) == "" && len(m.ToolCalls) == 0 {
			t.Errorf("invalid assistant message would get a 400: %+v", m)
		}
	}
	foundFollowUp := false
	for _, m := range wire.Messages {
		if m.Role == "user" && m.Content == "what did you do?" {
			foundFollowUp = true
		}
	}
	if !foundFollowUp {
		t.Errorf("follow-up prompt missing: %+v", wire.Messages)
	}
}

func TestChatRequestEmptyToolResult(t *testing.T) {
	req := messageRequest{
		Model: "deepseek-chat",
		Messages: []message{
			textMessage("user", "hi"),
			blocksMessage("user", []contentBlock{
				{Type: "tool_result", ToolUseID: "call_1", Content: ""},
			}),
		},
	}
	chat := toChatRequest(req)
	for _, m := range chat.Messages {
		if m.Role == "tool" && strings.TrimSpace(m.Content) == "" {
			t.Errorf("empty tool message would be rejected: %+v", m)
		}
	}
}

func TestChatResponseReasoningFallback(t *testing.T) {
	raw := []byte(`{"id":"gen_3","choices":[{"message":{"role":"assistant","content":"","reasoning_content":"thinking trace"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	resp, err := decodeChatResponse(200, raw)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if responseText(resp) != "thinking trace" {
		t.Errorf("reasoning_content should back empty content, got %q", responseText(resp))
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
	t.Setenv("OPENAI_API_KEY", "oai")
	sender, model, err = resolveBackend("openai", "", "")
	if err != nil {
		t.Fatalf("openai: %v", err)
	}
	if _, ok := sender.(*openAICompatClient); !ok {
		t.Errorf("expected *openAICompatClient, got %T", sender)
	}
	if model != defaultOpenAIModel {
		t.Errorf("unexpected openai default model %q", model)
	}

	t.Setenv("DEEPSEEK_API_KEY", "")
	if _, _, err := resolveBackend("deepseek", "", ""); err == nil {
		t.Error("expected missing-key error for deepseek")
	}
	t.Setenv("OPENAI_API_KEY", "")
	if _, _, err := resolveBackend("openai", "", ""); err == nil {
		t.Error("expected missing-key error for openai")
	}
	if _, _, err := resolveBackend("nope", "k", "m"); err == nil {
		t.Error("expected error for unknown provider")
	}
}
