package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// anthropicVersion pins the Messages API version header.
const anthropicVersion = "2023-06-01"

// defaultBaseURL is the Anthropic API root. Override with ANTHROPIC_BASE_URL
// to point at a compatible proxy.
const defaultBaseURL = "https://api.anthropic.com"

// contentBlock is one block of a message. Text, tool_use, and tool_result
// share the wire shape; unused fields stay empty.
type contentBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text,omitempty"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   string          `json:"content,omitempty"`
	IsError   *bool           `json:"is_error,omitempty"`
}

// message is one turn. Content holds either a plain string or an array of
// content blocks, encoded as raw JSON.
type message struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

func textMessage(role, text string) message {
	raw, _ := json.Marshal(text)
	return message{Role: role, Content: raw}
}

func blocksMessage(role string, blocks []contentBlock) message {
	raw, _ := json.Marshal(blocks)
	return message{Role: role, Content: raw}
}

type messageRequest struct {
	Model     string           `json:"model"`
	MaxTokens int              `json:"max_tokens"`
	System    string           `json:"system,omitempty"`
	Messages  []message        `json:"messages"`
	Tools     []toolDefinition `json:"tools,omitempty"`
}

type messageResponse struct {
	ID         string         `json:"id"`
	Content    []contentBlock `json:"content"`
	StopReason string         `json:"stop_reason"`
	Usage      struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

type anthropicClient struct {
	http    *http.Client
	apiKey  string
	baseURL string
}

func newAnthropicClient(apiKey, baseURL string) *anthropicClient {
	return &anthropicClient{
		http:    &http.Client{Timeout: 180 * time.Second},
		apiKey:  apiKey,
		baseURL: baseURL,
	}
}

func (c *anthropicClient) createMessage(ctx context.Context, req messageRequest) (*messageResponse, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("encode request: %w", err)
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/v1/messages", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("X-Api-Key", c.apiKey)
	httpReq.Header.Set("Anthropic-Version", anthropicVersion)

	resp, err := c.http.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("api call: %w", err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 16<<20))
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("api error %d: %s", resp.StatusCode, truncateOutput(string(raw)))
	}
	var out messageResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}
	return &out, nil
}
