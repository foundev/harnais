package main

import (
	"encoding/json"
	"strings"
	"testing"
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
