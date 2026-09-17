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
