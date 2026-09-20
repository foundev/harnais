package main

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"
	"time"
)

func TestBuildAnthropicRequestEffort(t *testing.T) {
	req := buildAnthropicRequest(messageRequest{
		Model:    "m",
		Effort:   "high",
		Messages: []message{textMessage("user", "hi")},
	})
	if req.OutputConfig == nil || req.OutputConfig.Effort != "high" {
		t.Fatalf("anthropic client should project effort: %+v", req.OutputConfig)
	}
	raw, err := json.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}
	if blob := string(raw); !strings.Contains(blob, `"output_config":{"effort":"high"}`) {
		t.Errorf("wire body must carry output_config.effort:\n%s", blob)
	}

	plain := buildAnthropicRequest(messageRequest{Model: "m"})
	if plain.OutputConfig != nil {
		t.Errorf("unset effort should send nothing: %+v", plain.OutputConfig)
	}
	raw, err = json.Marshal(plain)
	if err != nil {
		t.Fatal(err)
	}
	if blob := string(raw); strings.Contains(blob, "output_config") || strings.Contains(blob, `"Effort"`) {
		t.Errorf("wire body must omit effort when unset:\n%s", blob)
	}
}

func TestAnthropicMaxTokensOnTheWire(t *testing.T) {
	// The Messages API takes a max_tokens, so an uncapped request asks for
	// the largest output current Claude models allow. -max-tokens overrides
	// it, and nothing else inspects the value: it is the API's business.
	for _, tc := range []struct {
		name     string
		request  int
		wantSent int
	}{
		{"uncapped asks for the model's maximum", 0, anthropicMaxOutputTokens},
		{"an explicit cap is sent as-is", 4096, 4096},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := buildAnthropicRequest(messageRequest{Model: "m", MaxTokens: tc.request})
			if req.MaxTokens != tc.wantSent {
				t.Errorf("max_tokens = %d, want %d", req.MaxTokens, tc.wantSent)
			}
			if tc.request == 0 && anthropicMaxOutputTokens != 128000 {
				t.Errorf("uncapped anthropic requests must ask for 128K, got %d", anthropicMaxOutputTokens)
			}
		})
	}
}

func TestDecodeAnthropicModelsPage(t *testing.T) {
	ids, lastID, hasMore, err := decodeAnthropicModelsPage([]byte(`{"data":[{"type":"model","id":"claude-sonnet-4-20250514"},{"id":"claude-opus-4"}],"has_more":true,"last_id":"claude-opus-4"}`))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(ids) != 2 || ids[0] != "claude-sonnet-4-20250514" || ids[1] != "claude-opus-4" {
		t.Errorf("ids wrong: %v", ids)
	}
	if !hasMore || lastID != "claude-opus-4" {
		t.Errorf("pagination wrong: %q %v", lastID, hasMore)
	}
	if _, _, _, err := decodeAnthropicModelsPage([]byte(`{broken`)); err == nil {
		t.Error("expected decode error for broken JSON")
	}
}

// anthropicStream is a realistic SSE body: a thinking block, a text block,
// and a tool call whose arguments arrive in fragments.
const anthropicStream = `event: message_start
data: {"type":"message_start","message":{"id":"msg_1","role":"assistant","usage":{"input_tokens":42,"output_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"weigh the options"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig-abc"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Running "}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"the tests."}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}

event: content_block_start
data: {"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_1","name":"bash","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"command\":"}}

event: content_block_delta
data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"\"go test ./...\"}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":2}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":57}}

event: message_stop
data: {"type":"message_stop"}

`

func TestAnthropicStreamAssemblesAMessage(t *testing.T) {
	resp, err := readAnthropicStream(strings.NewReader(anthropicStream))
	if err != nil {
		t.Fatalf("stream decode: %v", err)
	}
	if resp.ID != "msg_1" || resp.StopReason != "tool_use" {
		t.Errorf("id/stop wrong: %+v", resp)
	}
	if resp.Usage.InputTokens != 42 || resp.Usage.OutputTokens != 57 {
		t.Errorf("usage wrong: %+v", resp.Usage)
	}
	if len(resp.Content) != 3 {
		t.Fatalf("expected three blocks in index order, got %+v", resp.Content)
	}
	// Extended thinking must round-trip byte-for-byte: the API rejects a
	// replayed turn whose thinking block was altered.
	if resp.Content[0].Type != "thinking" || resp.Content[0].Thinking != "weigh the options" || resp.Content[0].Signature != "sig-abc" {
		t.Errorf("thinking block mangled: %+v", resp.Content[0])
	}
	if got := resp.Content[1].Text; got != "Running the tests." {
		t.Errorf("text deltas must join in order, got %q", got)
	}
	call := resp.Content[2]
	if call.Type != "tool_use" || call.ID != "toolu_1" || call.Name != "bash" {
		t.Fatalf("tool block wrong: %+v", call)
	}
	if string(call.Input) != `{"command":"go test ./..."}` {
		t.Errorf("tool input must assemble from partial_json fragments, got %s", call.Input)
	}
	// The assembled history has to survive a replay unchanged.
	raw, err := json.Marshal(blocksMessage("assistant", resp.Content))
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`"thinking":"weigh the options"`, `"signature":"sig-abc"`, `"type":"tool_use"`} {
		if !strings.Contains(string(raw), want) {
			t.Errorf("replayed history lost %s: %s", want, raw)
		}
	}
}

func TestAnthropicRequestAsksForAStream(t *testing.T) {
	req := buildAnthropicRequest(messageRequest{Model: "m"})
	if !req.Stream {
		t.Error("every anthropic request must stream")
	}
	raw, err := json.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(raw), `"stream":true`) {
		t.Errorf("stream must reach the wire: %s", raw)
	}
}

func TestAnthropicStreamErrorsAreReported(t *testing.T) {
	body := "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n"
	_, err := readAnthropicStream(strings.NewReader(body))
	if err == nil || !strings.Contains(err.Error(), "Overloaded") {
		t.Fatalf("a stream error must surface the API's message, got %v", err)
	}
}

func TestAnthropicStreamFallsBackToPlainJSON(t *testing.T) {
	// A proxy that ignores "stream": true answers with the whole message at
	// once; that must decode rather than fail the turn.
	resp, err := readAnthropicStream(strings.NewReader(
		`{"id":"msg_9","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	if err != nil {
		t.Fatalf("plain JSON must decode: %v", err)
	}
	if len(resp.Content) != 1 || resp.Content[0].Text != "hi" || resp.StopReason != "end_turn" {
		t.Errorf("plain JSON decoded wrong: %+v", resp)
	}
}

// silentBody models a stream that has gone quiet: reads only end when the
// request's context is canceled, which is what a torn-down connection does.
type silentBody struct{ ctx context.Context }

func (b silentBody) Read([]byte) (int, error) {
	<-b.ctx.Done()
	return 0, b.ctx.Err()
}

func (b silentBody) Close() error { return nil }

// silentTransport answers every request with a body that only ends when the
// request's context does — a stream that has gone quiet.
type silentTransport struct{ calls *int }

func (t silentTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	*t.calls++
	return &http.Response{StatusCode: http.StatusOK, Body: silentBody{ctx: r.Context()}, Header: http.Header{}}, nil
}

func TestAnthropicDropsAStalledStream(t *testing.T) {
	// Streaming removes the total-request timeout that used to cut off long
	// answers, so the only remaining guard is silence: a stream that sends
	// nothing for anthropicStreamIdle is dropped and reported instead of
	// hanging the turn forever.
	defer func(prev time.Duration) { anthropicStreamIdle = prev }(anthropicStreamIdle)
	anthropicStreamIdle = 20 * time.Millisecond

	var calls int
	c := newAnthropicClient("k", "http://unused")
	c.http = &http.Client{Transport: silentTransport{calls: &calls}}
	c.retry = fastRetry()
	_, err := c.createMessage(context.Background(), messageRequest{
		Model: "m", Messages: []message{textMessage("user", "hi")},
	})
	if err == nil || !strings.Contains(err.Error(), "went quiet") {
		t.Fatalf("a stalled stream must be reported, got %v", err)
	}
}
