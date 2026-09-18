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
