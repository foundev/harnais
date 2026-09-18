package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"sort"
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
	// Effort is harness-internal: reasoning effort (low/medium/high),
	// projected onto output_config.effort on the wire (see createMessage).
	Effort string `json:"-"`
	// OutputConfig carries effort to the Anthropic API; nil (the default
	// when no effort is set) omits it so backend-default behavior stays.
	OutputConfig *outputConfig `json:"output_config,omitempty"`
}

// outputConfig mirrors the Anthropic Messages API output_config block,
// currently used only for reasoning effort.
type outputConfig struct {
	Effort string `json:"effort,omitempty"`
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
	retry   retryPolicy
}

func newAnthropicClient(apiKey, baseURL string) *anthropicClient {
	return &anthropicClient{
		http:    &http.Client{Timeout: 180 * time.Second},
		apiKey:  apiKey,
		baseURL: baseURL,
		retry:   defaultRetryPolicy(),
	}
}

// buildAnthropicRequest projects harness-internal effort onto the wire
// shape. Pure so the mapping stays testable without network.
func buildAnthropicRequest(req messageRequest) messageRequest {
	if req.Effort != "" {
		req.OutputConfig = &outputConfig{Effort: req.Effort}
	}
	return req
}

func (c *anthropicClient) createMessage(ctx context.Context, req messageRequest) (*messageResponse, error) {
	body, err := json.Marshal(buildAnthropicRequest(req))
	if err != nil {
		return nil, fmt.Errorf("encode request: %w", err)
	}
	var lastErr error
	for attempt := 0; ; attempt++ {
		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/v1/messages", bytes.NewReader(body))
		if err != nil {
			return nil, fmt.Errorf("build request: %w", err)
		}
		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("X-Api-Key", c.apiKey)
		httpReq.Header.Set("Anthropic-Version", anthropicVersion)

		status, raw, err := c.do(httpReq)
		if err != nil {
			// Transport failure (timeout, connection, half-read body).
			lastErr = err
			if !c.retry.retryTransport || attempt >= c.retry.maxAttempts {
				return nil, lastErr
			}
			if serr := sleepRetry(ctx, c.retry.delayFor(attempt+1)); serr != nil {
				return nil, lastErr
			}
			continue
		}
		if c.retry.retryableStatus(status) && attempt < c.retry.maxAttempts {
			lastErr = fmt.Errorf("api error %d: %s", status, truncateOutput(string(raw)))
			if serr := sleepRetry(ctx, c.retry.delayFor(attempt+1)); serr != nil {
				return nil, lastErr
			}
			continue
		}
		if status < 200 || status >= 300 {
			return nil, fmt.Errorf("api error %d: %s", status, truncateOutput(string(raw)))
		}
		var out messageResponse
		if err := json.Unmarshal(raw, &out); err != nil {
			return nil, fmt.Errorf("decode response: %w", err)
		}
		return &out, nil
	}
}

func (c *anthropicClient) do(httpReq *http.Request) (int, []byte, error) {
	resp, err := c.http.Do(httpReq)
	if err != nil {
		return 0, nil, fmt.Errorf("api call: %w", err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 16<<20))
	if err != nil {
		return 0, nil, fmt.Errorf("read response: %w", err)
	}
	return resp.StatusCode, raw, nil
}

// anthropicModelsPage is one page of GET /v1/models output.
type anthropicModelsPage struct {
	Data []struct {
		ID string `json:"id"`
	} `json:"data"`
	HasMore bool   `json:"has_more"`
	LastID  string `json:"last_id"`
}

// decodeAnthropicModelsPage extracts one page's IDs plus the pagination
// cursor. Pure so the mapping stays testable without network.
func decodeAnthropicModelsPage(raw []byte) (ids []string, lastID string, hasMore bool, err error) {
	var page anthropicModelsPage
	if err := json.Unmarshal(raw, &page); err != nil {
		return nil, "", false, fmt.Errorf("decode models: %w", err)
	}
	for _, m := range page.Data {
		if m.ID != "" {
			ids = append(ids, m.ID)
		}
	}
	return ids, page.LastID, page.HasMore, nil
}

// listModels fetches the backend's model IDs for /models and TAB
// completion (see main.go), following the cursor until the pages run out
// (bounded: a broken has_more loop must not spin forever).
func (c *anthropicClient) listModels(ctx context.Context) ([]string, error) {
	var ids []string
	afterID := ""
	for page := 0; ; page++ {
		u := c.baseURL + "/v1/models?limit=100"
		if afterID != "" {
			u += "&after_id=" + url.QueryEscape(afterID)
		}
		httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
		if err != nil {
			return nil, fmt.Errorf("build request: %w", err)
		}
		httpReq.Header.Set("X-Api-Key", c.apiKey)
		httpReq.Header.Set("Anthropic-Version", anthropicVersion)
		status, raw, err := c.do(httpReq)
		if err != nil {
			return nil, err
		}
		if status < 200 || status >= 300 {
			return nil, fmt.Errorf("api error %d: %s", status, truncateOutput(string(raw)))
		}
		pageIDs, lastID, hasMore, err := decodeAnthropicModelsPage(raw)
		if err != nil {
			return nil, err
		}
		ids = append(ids, pageIDs...)
		if !hasMore || lastID == "" || page >= 4 {
			break
		}
		afterID = lastID
	}
	sort.Strings(ids)
	return ids, nil
}
