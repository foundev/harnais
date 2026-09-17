package main

import (
	"encoding/base64"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func craftJWT(t *testing.T, claims map[string]any) string {
	t.Helper()
	raw, err := json.Marshal(claims)
	if err != nil {
		t.Fatal(err)
	}
	return "h." + base64.RawURLEncoding.EncodeToString(raw) + ".s"
}

func writeAuthJSON(t *testing.T, dir, content string) string {
	t.Helper()
	path := filepath.Join(dir, "auth.json")
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestCodexResponsesTranslation(t *testing.T) {
	req := messageRequest{
		Model:  "gpt-5.3-codex",
		System: "be brief",
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
	r := toResponsesRequest(req)

	if r.Model != req.Model || r.Instructions != "be brief" {
		t.Fatalf("model/instructions not carried over: %+v", r)
	}
	if len(r.Input) != 4 {
		t.Fatalf("expected 4 input items, got %d: %+v", len(r.Input), r.Input)
	}
	if r.Input[0].Type != "message" || r.Input[0].Role != "user" ||
		r.Input[0].Content[0].Type != "input_text" {
		t.Errorf("user item wrong: %+v", r.Input[0])
	}
	call := r.Input[2]
	if call.Type != "function_call" || call.CallID != "call_1" ||
		call.Name != "bash" || call.Arguments != `{"command":"ls"}` {
		t.Errorf("function_call wrong: %+v", call)
	}
	out := r.Input[3]
	if out.Type != "function_call_output" || out.CallID != "call_1" || out.Output != "main.go" {
		t.Errorf("function_call_output wrong: %+v", out)
	}
	if len(r.Tools) != 4 || r.Tools[0].Type != "function" || r.Tools[0].Name != "bash" {
		t.Fatalf("tools not converted: %+v", r.Tools)
	}
}

func TestCodexDecodeWithFunctionCall(t *testing.T) {
	raw := []byte("event: response.created\n" +
		"data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"status\":\"in_progress\",\"output\":[]}}\n" +
		"\n" +
		"event: response.output_text.delta\n" +
		"data: {\"type\":\"response.output_text.delta\",\"delta\":\"running\"}\n" +
		"\n" +
		"event: response.completed\n" +
		`data: {"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"running"}]},{"type":"function_call","call_id":"call_2","name":"read","arguments":"{\"path\":\"main.go\"}"}],"usage":{"input_tokens":10,"output_tokens":5}}}` + "\n")
	resp, err := decodeResponsesStream(raw)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if resp.StopReason != "tool_use" {
		t.Errorf("function_call must continue the loop, got %q", resp.StopReason)
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

func TestDumpDebug(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "debug.log")
	t.Setenv("HARNAIS_DEBUG", path)
	dumpDebug("thing", []byte(`{"a":1}`))
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(raw), "thing") || !strings.Contains(string(raw), `{"a":1}`) {
		t.Errorf("debug file should hold label and body: %s", raw)
	}
}

func TestCodexRequestStoreAndStream(t *testing.T) {
	raw, err := json.Marshal(toResponsesRequest(messageRequest{Model: "m"}))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(raw), `"store":false`) {
		t.Errorf("backend requires store:false: %s", raw)
	}
	if !strings.Contains(string(raw), `"stream":true`) {
		t.Errorf("backend requires stream:true: %s", raw)
	}
}

func TestCodexDetailError(t *testing.T) {
	// Exact shape the ChatGPT backend returns for a stored request.
	err := codexStatusError(400, []byte(`{"detail":"Store must be set to false"}`))
	if err == nil || err.Error() != "api error 400: Store must be set to false" {
		t.Errorf("detail message should surface cleanly, got %v", err)
	}
}

func TestCodexResponsesReasoning(t *testing.T) {
	req := messageRequest{Model: "m", Effort: "low", Messages: []message{textMessage("user", "hi")}}
	r := toResponsesRequest(req)
	if r.Reasoning == nil || r.Reasoning.Effort != "low" {
		t.Errorf("effort should map to reasoning: %+v", r.Reasoning)
	}
	r = toResponsesRequest(messageRequest{Model: "m"})
	if r.Reasoning != nil {
		t.Errorf("unset effort should send no reasoning block: %+v", r.Reasoning)
	}
}

func TestCodexDecodeFinalText(t *testing.T) {
	raw := []byte("event: response.completed\n" +
		`data: {"type":"response.completed","response":{"id":"resp_2","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"done"}]}]}}` + "\n")
	resp, err := decodeResponsesStream(raw)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if resp.StopReason != "end_turn" || responseText(resp) != "done" {
		t.Errorf("final answer wrong: %+v", resp)
	}
}

func TestCodexStreamItemsFromDoneEvents(t *testing.T) {
	// Observed live against the ChatGPT backend: complete items stream via
	// response.output_item.done, then response.completed arrives with an
	// empty output array. The done items are the answer.
	raw := []byte("event: response.output_item.done\n" +
		`data: {"type":"response.output_item.done","output_index":0,"item":{"id":"msg_1","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"ok","annotations":[]}],"phase":"final_answer"}}` + "\n" +
		"\n" +
		"event: response.completed\n" +
		`data: {"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[],"usage":{"input_tokens":474,"output_tokens":5}}}` + "\n")
	resp, err := decodeResponsesStream(raw)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if resp.StopReason != "end_turn" || responseText(resp) != "ok" {
		t.Errorf("done-item text lost: %+v", resp)
	}
	if resp.Usage.InputTokens != 474 || resp.Usage.OutputTokens != 5 {
		t.Errorf("terminal usage lost: %+v", resp.Usage)
	}
}

func TestCodexStreamToolCallFromDoneEvents(t *testing.T) {
	raw := []byte("event: response.output_item.done\n" +
		`data: {"type":"response.output_item.done","output_index":0,"item":{"id":"fc_1","type":"function_call","status":"completed","call_id":"call_2","name":"read","arguments":"{\"path\":\"main.go\"}"}}` + "\n" +
		"\n" +
		"event: response.completed\n" +
		`data: {"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[]}}` + "\n")
	resp, err := decodeResponsesStream(raw)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if resp.StopReason != "tool_use" || len(resp.Content) != 1 || resp.Content[0].ID != "call_2" {
		t.Errorf("done-item tool call lost: %+v", resp)
	}
}

func TestCodexDecodeErrors(t *testing.T) {
	if err := codexStatusError(401, []byte(`unauthorized`)); err == nil ||
		!strings.Contains(err.Error(), "401") {
		t.Errorf("expected status error, got %v", err)
	}
	// A stream that ends mid-response is a failure, never a silent success.
	truncated := []byte("event: response.output_text.delta\n" +
		"data: {\"type\":\"response.output_text.delta\",\"delta\":\"half\"}\n")
	if _, err := decodeResponsesStream(truncated); err == nil ||
		!strings.Contains(err.Error(), "without a completed response") {
		t.Errorf("expected truncated-stream error, got %v", err)
	}
	if _, err := decodeResponsesStream([]byte("event: error\n" +
		`data: {"type":"error","message":"usage limited"}` + "\n")); err == nil ||
		!strings.Contains(err.Error(), "usage limited") {
		t.Errorf("expected stream error, got %v", err)
	}
	failed := []byte("event: response.failed\n" +
		`data: {"type":"response.failed","response":{"id":"resp_9","status":"failed","output":[],"error":{"message":"boom"}}}` + "\n")
	if _, err := decodeResponsesStream(failed); err == nil ||
		!strings.Contains(err.Error(), "boom") {
		t.Errorf("expected failed-response error, got %v", err)
	}
	// A plain (non-stream) error body still decodes instead of confusing.
	if _, err := decodeResponsesStream([]byte(`{"id":"x","error":{"message":"plain boom"}}`)); err == nil ||
		!strings.Contains(err.Error(), "plain boom") {
		t.Errorf("expected plain-body error, got %v", err)
	}
}

func TestLoadCodexTokens(t *testing.T) {
	dir := t.TempDir()
	access := craftJWT(t, map[string]any{"exp": time.Now().Add(time.Hour).Unix()})
	path := writeAuthJSON(t, dir, `{"auth_mode":"chatgpt","tokens":{"id_token":"i","access_token":"`+access+`","refresh_token":"r","account_id":"acct_1"},"last_refresh":"2026-09-14T04:36:32Z"}`)
	tok, err := loadCodexTokens(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if tok.RefreshToken != "r" || tok.AccountID != "acct_1" {
		t.Errorf("tokens wrong: %+v", tok)
	}

	apiKeyPath := writeAuthJSON(t, t.TempDir(), `{"auth_mode":"apikey","OPENAI_API_KEY":"sk-x"}`)
	if _, err := loadCodexTokens(apiKeyPath); err == nil ||
		!strings.Contains(err.Error(), "codex login") {
		t.Errorf("expected codex-login error for apikey mode, got %v", err)
	}
	if _, err := loadCodexTokens(filepath.Join(dir, "missing.json")); err == nil ||
		!strings.Contains(err.Error(), "codex login") {
		t.Errorf("expected codex-login error for missing file, got %v", err)
	}
}

func TestCodexJWTExp(t *testing.T) {
	future := time.Now().Add(time.Hour).Truncate(time.Second)
	exp, err := codexJWTExp(craftJWT(t, map[string]any{"exp": future.Unix()}))
	if err != nil {
		t.Fatalf("exp: %v", err)
	}
	if !exp.Equal(future) {
		t.Errorf("exp wrong: got %v want %v", exp, future)
	}
	if _, err := codexJWTExp("not-a-jwt"); err == nil {
		t.Error("expected error for malformed token")
	}
}

func TestCodexAccountIDFallback(t *testing.T) {
	access := craftJWT(t, map[string]any{
		"exp":                         time.Now().Add(time.Hour).Unix(),
		"https://api.openai.com/auth": map[string]any{"chatgpt_account_id": "acct_jwt"},
	})
	got, err := codexAccountID(&codexTokens{AccessToken: access})
	if err != nil {
		t.Fatalf("fallback: %v", err)
	}
	if got != "acct_jwt" {
		t.Errorf("fallback account wrong: %q", got)
	}
	got, err = codexAccountID(&codexTokens{AccessToken: access, AccountID: "acct_stored"})
	if err != nil || got != "acct_stored" {
		t.Errorf("stored account should win: %q, %v", got, err)
	}
	if _, err := codexAccountID(&codexTokens{AccessToken: "bad"}); err == nil {
		t.Error("expected error when no account is resolvable")
	}
}

func TestResolveBackendCodex(t *testing.T) {
	dir := t.TempDir()
	access := craftJWT(t, map[string]any{"exp": time.Now().Add(time.Hour).Unix()})
	writeAuthJSON(t, dir, `{"auth_mode":"chatgpt","tokens":{"access_token":"`+access+`","refresh_token":"r","account_id":"acct_1"}}`)
	t.Setenv("CODEX_HOME", dir)

	sender, model, err := resolveBackend("codex", "", "")
	if err != nil {
		t.Fatalf("codex: %v", err)
	}
	client, ok := sender.(*codexClient)
	if !ok {
		t.Fatalf("expected *codexClient, got %T", sender)
	}
	if model != defaultCodexModel {
		t.Errorf("unexpected codex default model %q", model)
	}
	if client.authPath != filepath.Join(dir, "auth.json") {
		t.Errorf("auth path should honor CODEX_HOME: %q", client.authPath)
	}

	t.Setenv("CODEX_HOME", t.TempDir())
	if _, _, err := resolveBackend("codex", "", ""); err == nil {
		t.Error("expected error when auth.json is missing")
	}
}
