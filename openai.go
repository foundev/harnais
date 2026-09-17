package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

// Provider roots and default models for the OpenAI-compatible backends.
// Base URLs are overridable with OPENROUTER_BASE_URL / DEEPSEEK_BASE_URL /
// OPENAI_BASE_URL.
const defaultOpenRouterBaseURL = "https://openrouter.ai/api/v1"

// defaultOpenRouterModel applies when -model is unset with -provider=openrouter.
const defaultOpenRouterModel = "anthropic/claude-sonnet-4.5"

const defaultDeepSeekBaseURL = "https://api.deepseek.com"

// defaultDeepSeekModel applies when -model is unset with -provider=deepseek.
const defaultDeepSeekModel = "deepseek-chat"

const defaultOpenAIBaseURL = "https://api.openai.com/v1"

// defaultOpenAIModel applies when -model is unset with -provider=openai.
const defaultOpenAIModel = "gpt-5-mini"

// openAICompatClient speaks OpenAI-style /chat/completions, shared by the
// OpenRouter, DeepSeek, and OpenAI backends. Translation to and from the
// harness message shape lives in toChatRequest / decodeChatResponse.
type openAICompatClient struct {
	http    *http.Client
	apiKey  string
	baseURL string
	referer string
}

func newOpenRouterClient(apiKey, baseURL string) *openAICompatClient {
	return &openAICompatClient{
		http:    &http.Client{Timeout: 180 * time.Second},
		apiKey:  apiKey,
		baseURL: baseURL,
		referer: os.Getenv("OPENROUTER_REFERER"),
	}
}

func newDeepSeekClient(apiKey, baseURL string) *openAICompatClient {
	return &openAICompatClient{
		http:    &http.Client{Timeout: 180 * time.Second},
		apiKey:  apiKey,
		baseURL: baseURL,
	}
}

func newOpenAIClient(apiKey, baseURL string) *openAICompatClient {
	return &openAICompatClient{
		http:    &http.Client{Timeout: 180 * time.Second},
		apiKey:  apiKey,
		baseURL: baseURL,
	}
}

// buildChatRequest translates the harness request. Every OpenAI-compatible
// backend (OpenAI, OpenRouter, DeepSeek) accepts reasoning_effort, so it
// passes through untouched. Pure so the mapping stays testable without
// network.
func (c *openAICompatClient) buildChatRequest(req messageRequest) chatRequest {
	return toChatRequest(req)
}

func (c *openAICompatClient) createMessage(ctx context.Context, req messageRequest) (*messageResponse, error) {
	body, err := json.Marshal(c.buildChatRequest(req))
	if err != nil {
		return nil, fmt.Errorf("encode request: %w", err)
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	httpReq.Header.Set("X-Title", "harnais")
	if c.referer != "" {
		httpReq.Header.Set("HTTP-Referer", c.referer)
	}

	resp, err := c.http.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("api call: %w", err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 16<<20))
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}
	return decodeChatResponse(resp.StatusCode, raw)
}

// decodeChatResponse maps an OpenAI-style chat completion onto the harness
// message shape. Pure function so the mapping is testable without network.
func decodeChatResponse(status int, raw []byte) (*messageResponse, error) {
	if status < 200 || status >= 300 {
		return nil, fmt.Errorf("api error %d: %s", status, truncateOutput(string(raw)))
	}
	var chat chatResponse
	if err := json.Unmarshal(raw, &chat); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}
	if chat.Error != nil && chat.Error.Message != "" {
		return nil, fmt.Errorf("api error: %s", chat.Error.Message)
	}
	if len(chat.Choices) == 0 {
		return nil, fmt.Errorf("api error: response had no choices")
	}
	return fromChatChoice(chat.Choices[0], chat.ID, chat.Usage.PromptTokens, chat.Usage.CompletionTokens), nil
}

type chatMessage struct {
	Role       string         `json:"role"`
	Content    string         `json:"content,omitempty"`
	ToolCalls  []chatToolCall `json:"tool_calls,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
}

type chatToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function chatFunctionCall `json:"function"`
}

type chatFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type chatTool struct {
	Type     string       `json:"type"`
	Function chatFunction `json:"function"`
}

type chatFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

type chatRequest struct {
	Model           string        `json:"model"`
	MaxTokens       int           `json:"max_tokens,omitempty"`
	ReasoningEffort *string       `json:"reasoning_effort,omitempty"`
	Messages        []chatMessage `json:"messages"`
	Tools           []chatTool    `json:"tools,omitempty"`
}

type chatResponse struct {
	ID      string `json:"id"`
	Choices []struct {
		Message      chatMessage `json:"message"`
		FinishReason string      `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

// toChatRequest converts the harness message history to OpenAI-style chat
// messages. Assistant tool_use blocks become tool_calls; tool_result blocks
// become standalone tool messages carrying the tool_call_id.
func toChatRequest(req messageRequest) chatRequest {
	out := chatRequest{Model: req.Model, MaxTokens: req.MaxTokens}
	if req.Effort != "" {
		effort := req.Effort
		out.ReasoningEffort = &effort
	}
	if req.System != "" {
		out.Messages = append(out.Messages, chatMessage{Role: "system", Content: req.System})
	}
	for _, msg := range req.Messages {
		text, blocks := splitContent(msg.Content)
		switch msg.Role {
		case "assistant":
			assistant := chatMessage{Role: "assistant"}
			assistant.Content = text
			for _, b := range blocks {
				if b.Type != "tool_use" {
					continue
				}
				args := string(b.Input)
				if args == "" {
					args = "{}"
				}
				assistant.ToolCalls = append(assistant.ToolCalls, chatToolCall{
					ID:   b.ID,
					Type: "function",
					Function: chatFunctionCall{
						Name:      b.Name,
						Arguments: args,
					},
				})
			}
			out.Messages = append(out.Messages, assistant)
		default: // user (and anything else) maps to user/tool messages
			if text != "" || len(blocks) == 0 {
				out.Messages = append(out.Messages, chatMessage{Role: "user", Content: text})
			}
			for _, b := range blocks {
				if b.Type != "tool_result" {
					continue
				}
				out.Messages = append(out.Messages, chatMessage{
					Role:       "tool",
					Content:    b.Content,
					ToolCallID: b.ToolUseID,
				})
			}
		}
	}
	for _, t := range req.Tools {
		out.Tools = append(out.Tools, chatTool{
			Type: "function",
			Function: chatFunction{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.InputSchema,
			},
		})
	}
	return out
}

// fromChatChoice converts one completed choice back to the harness shape.
// finish_reason "tool_calls" continues the agent loop like "tool_use".
func fromChatChoice(choice struct {
	Message      chatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}, id string, promptTokens, completionTokens int) *messageResponse {
	resp := &messageResponse{ID: id}
	if choice.Message.Content != "" {
		resp.Content = append(resp.Content, contentBlock{Type: "text", Text: choice.Message.Content})
	}
	for _, call := range choice.Message.ToolCalls {
		args := call.Function.Arguments
		if args == "" {
			args = "{}"
		}
		resp.Content = append(resp.Content, contentBlock{
			Type:  "tool_use",
			ID:    call.ID,
			Name:  call.Function.Name,
			Input: json.RawMessage(args),
		})
	}
	if choice.FinishReason == "tool_calls" {
		resp.StopReason = "tool_use"
	} else {
		resp.StopReason = "end_turn"
	}
	resp.Usage.InputTokens = promptTokens
	resp.Usage.OutputTokens = completionTokens
	return resp
}

// splitContent decodes a harness message body into its plain text and its
// content blocks (whichever wire form it holds).
func splitContent(raw json.RawMessage) (string, []contentBlock) {
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return text, nil
	}
	var blocks []contentBlock
	if err := json.Unmarshal(raw, &blocks); err == nil {
		var sb []string
		var rest []contentBlock
		for _, b := range blocks {
			if b.Type == "text" {
				sb = append(sb, b.Text)
			} else {
				rest = append(rest, b)
			}
		}
		joined := ""
		for i, s := range sb {
			if i > 0 {
				joined += "\n"
			}
			joined += s
		}
		return joined, rest
	}
	return "", nil
}
