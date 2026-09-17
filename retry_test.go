package main

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"
)

func TestRetryBackoffGrows(t *testing.T) {
	base := 10 * time.Millisecond
	if got := retryBackoff(base, 0); got != base {
		t.Errorf("attempt 0 should be exactly base, got %v", got)
	}
	for attempt, want := range map[int]time.Duration{
		1: base,
		2: 2 * base,
		3: 4 * base,
	} {
		got := retryBackoff(base, attempt)
		lo := time.Duration(0.9 * float64(want))
		hi := time.Duration(1.1*float64(want)) + 1
		if got < lo || got > hi {
			t.Errorf("attempt %d: got %v, want within [%v,%v]", attempt, got, lo, hi)
		}
	}
}

func TestRetryableStatusDefault(t *testing.T) {
	p := defaultRetryPolicy()
	for _, status := range []int{500, 502, 503, 599} {
		if !p.retryableStatus(status) {
			t.Errorf("status %d should be retryable", status)
		}
	}
	for _, status := range []int{200, 400, 401, 403, 404, 429} {
		if p.retryableStatus(status) {
			t.Errorf("status %d should not be retried by default", status)
		}
	}
}

func fastRetry() retryPolicy {
	p := defaultRetryPolicy()
	p.baseDelay = time.Nanosecond
	return p
}

// stubTransport replays canned responses without touching the network.
type stubTransport struct {
	calls  *int
	handle func(call int) (*http.Response, error)
}

func (s stubTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	*s.calls++
	return s.handle(*s.calls)
}

func stubResponse(status int, body string) *http.Response {
	return &http.Response{
		StatusCode: status,
		Body:       io.NopCloser(strings.NewReader(body)),
		Header:     make(http.Header),
	}
}

func TestAnthropicRetries5xxThenSucceeds(t *testing.T) {
	var calls int
	c := newAnthropicClient("k", "http://unused")
	c.http = &http.Client{Transport: stubTransport{calls: &calls, handle: func(call int) (*http.Response, error) {
		if call < 3 {
			return stubResponse(http.StatusBadGateway, "busy"), nil
		}
		return stubResponse(http.StatusOK, `{"id":"msg_1","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`), nil
	}}}
	c.retry = fastRetry()
	resp, err := c.createMessage(context.Background(), messageRequest{
		Model: "m", MaxTokens: 8, Messages: []message{textMessage("user", "hi")},
	})
	if err != nil {
		t.Fatalf("createMessage: %v", err)
	}
	if responseText(resp) != "hi" {
		t.Errorf("unexpected text: %+v", resp.Content)
	}
	if calls != 3 {
		t.Errorf("expected 3 attempts, got %d", calls)
	}
}

func TestAnthropicSkips429ByDefault(t *testing.T) {
	var calls int
	c := newAnthropicClient("k", "http://unused")
	c.http = &http.Client{Transport: stubTransport{calls: &calls, handle: func(call int) (*http.Response, error) {
		return stubResponse(http.StatusTooManyRequests, "slow down"), nil
	}}}
	c.retry = fastRetry()
	_, err := c.createMessage(context.Background(), messageRequest{
		Model: "m", MaxTokens: 8, Messages: []message{textMessage("user", "hi")},
	})
	if err == nil || !strings.Contains(err.Error(), "429") {
		t.Fatalf("expected 429 error, got %v", err)
	}
	if calls != 1 {
		t.Errorf("429 must fail fast by default, got %d attempts", calls)
	}
}

func TestOpenAICompatRetriesTransportThenGivesUp(t *testing.T) {
	var calls int
	c := newOpenAIClient("k", "http://unused")
	c.http = &http.Client{Transport: stubTransport{calls: &calls, handle: func(call int) (*http.Response, error) {
		return nil, fmt.Errorf("connection reset")
	}}}
	c.retry = fastRetry()
	c.retry.maxAttempts = 2

	_, err := c.createMessage(context.Background(), messageRequest{
		Model: "m", Messages: []message{textMessage("user", "hi")},
	})
	if err == nil || !strings.Contains(err.Error(), "connection reset") {
		t.Fatalf("expected transport error, got %v", err)
	}
	if calls != 3 {
		t.Errorf("expected maxAttempts+1 transport attempts, got %d", calls)
	}
}

func TestOpenAICompatExhaustsStatusRetries(t *testing.T) {
	var calls int
	c := newOpenAIClient("k", "http://unused")
	c.http = &http.Client{Transport: stubTransport{calls: &calls, handle: func(call int) (*http.Response, error) {
		return stubResponse(http.StatusServiceUnavailable, `{"error":{"message":"boom"}}`), nil
	}}}
	c.retry = fastRetry()
	c.retry.maxAttempts = 2
	_, err := c.createMessage(context.Background(), messageRequest{
		Model: "m", Messages: []message{textMessage("user", "hi")},
	})
	if err == nil || !strings.Contains(err.Error(), "boom") {
		t.Fatalf("expected backend message, got %v", err)
	}
	if calls != 3 {
		t.Errorf("expected maxAttempts+1 status attempts, got %d", calls)
	}
}

func TestCodexRetries5xxThenSucceeds(t *testing.T) {
	dir := t.TempDir()
	access := craftJWT(t, map[string]any{"exp": time.Now().Add(time.Hour).Unix()})
	authPath := writeAuthJSON(t, dir, `{"auth_mode":"chatgpt","tokens":{"id_token":"i","access_token":"`+access+`","refresh_token":"r","account_id":"acct_1"}}`)
	c := newCodexClient(authPath, "http://unused")
	var calls int
	c.http = &http.Client{Transport: stubTransport{calls: &calls, handle: func(call int) (*http.Response, error) {
		if call == 1 {
			return stubResponse(http.StatusBadGateway, `{"detail":"busy"}`), nil
		}
		return stubResponse(http.StatusOK, "event: response.completed\n"+
			`data: {"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}]}}`+"\n"), nil
	}}}
	c.retry = fastRetry()

	resp, err := c.createMessage(context.Background(), messageRequest{
		Model: "m", Messages: []message{textMessage("user", "hi")},
	})
	if err != nil {
		t.Fatalf("createMessage: %v", err)
	}
	if responseText(resp) != "ok" {
		t.Errorf("unexpected text: %+v", resp.Content)
	}
	if calls != 2 {
		t.Errorf("expected 2 attempts, got %d", calls)
	}
}
