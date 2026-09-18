package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
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
const defaultDeepSeekModel = "deepseek-flash"

const defaultOpenAIBaseURL = "https://api.openai.com/v1"

// defaultOpenAIModel applies when -model is unset with -provider=openai.
const defaultOpenAIModel = "gpt-5-mini"

const defaultInceptronBaseURL = "https://api.inceptron.io/v1"

// defaultInceptronModel applies when -model is unset with -provider=inceptron.
const defaultInceptronModel = "zai-org/GLM-5.3"

// openAICompatClient speaks OpenAI-style /chat/completions, shared by the
// OpenRouter, DeepSeek, and OpenAI backends. Translation to and from the
// harness message shape lives in toChatRequest / decodeChatResponse.
type openAICompatClient struct {
	http    *http.Client
	apiKey  string
	baseURL string
	referer string
	retry   retryPolicy
	// echoReasoning sends stored reasoning blocks back as
	// reasoning_content on assistant messages. DeepSeek requires this
	// whenever tools are in play (400 otherwise) and ignores it
	// without tools; other backends must not receive the field.
	echoReasoning bool
}

func newOpenRouterClient(apiKey, baseURL string) *openAICompatClient {
	return &openAICompatClient{
		http:    &http.Client{Timeout: 180 * time.Second},
		apiKey:  apiKey,
		baseURL: baseURL,
		referer: os.Getenv("OPENROUTER_REFERER"),
		retry:   defaultRetryPolicy(),
	}
}

func newDeepSeekClient(apiKey, baseURL string) *openAICompatClient {
	return &openAICompatClient{
		http:          &http.Client{Timeout: 180 * time.Second},
		apiKey:        apiKey,
		baseURL:       baseURL,
		retry:         defaultRetryPolicy(),
		echoReasoning: true,
	}
}

func newOpenAIClient(apiKey, baseURL string) *openAICompatClient {
	return &openAICompatClient{
		http:    &http.Client{Timeout: 180 * time.Second},
		apiKey:  apiKey,
		baseURL: baseURL,
		retry:   defaultRetryPolicy(),
	}
}

func newInceptronClient(apiKey, baseURL string) *openAICompatClient {
	return &openAICompatClient{
		http:    &http.Client{Timeout: 180 * time.Second},
		apiKey:  apiKey,
		baseURL: baseURL,
		retry:   defaultRetryPolicy(),
	}
}

// buildChatRequest translates the harness request. Every OpenAI-compatible
// backend (OpenAI, OpenRouter, DeepSeek) accepts reasoning_effort, so it
// passes through untouched. Pure so the mapping stays testable without
// network.
func (c *openAICompatClient) buildChatRequest(req messageRequest) chatRequest {
	return toChatRequest(req, c.echoReasoning)
}

func (c *openAICompatClient) createMessage(ctx context.Context, req messageRequest) (*messageResponse, error) {
	body, err := json.Marshal(c.buildChatRequest(req))
	if err != nil {
		return nil, fmt.Errorf("encode request: %w", err)
	}
	var lastErr error
	for attempt := 0; ; attempt++ {
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
			_, lastErr = decodeChatResponse(status, raw)
			if serr := sleepRetry(ctx, c.retry.delayFor(attempt+1)); serr != nil {
				return nil, lastErr
			}
			continue
		}
		return decodeChatResponse(status, raw)
	}
}

func (c *openAICompatClient) do(httpReq *http.Request) (int, []byte, error) {
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
	resp := fromChatChoice(chat.Choices[0], chat.ID, chat.Usage.PromptTokens, chat.Usage.CompletionTokens)
	if len(resp.Content) == 0 {
		// An empty end_turn would otherwise surface as a blank answer
		// with no error. Fail loudly, naming the finish reason so a
		// length cut carries its own remedy.
		if fr := chat.Choices[0].FinishReason; fr == "length" {
			return nil, fmt.Errorf("api error: model returned no content (finish_reason %q) — output hit the token limit; retry with a higher -max-tokens", fr)
		}
		return nil, fmt.Errorf("api error: model returned no content (finish_reason %q)", chat.Choices[0].FinishReason)
	}
	return resp, nil
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content,omitempty"`
	// ReasoningContent carries DeepSeek thinking-mode reasoning. Read off
	// every response; sent back on assistant messages only by backends
	// that require it (DeepSeek with tools), never otherwise.
	ReasoningContent string         `json:"reasoning_content,omitempty"`
	ToolCalls        []chatToolCall `json:"tool_calls,omitempty"`
	ToolCallID       string         `json:"tool_call_id,omitempty"`
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
// become standalone tool messages carrying the tool_call_id. Assistant
// reasoning blocks round-trip as reasoning_content only when echoReasoning
// is set (DeepSeek thinking mode with tools requires the full chain back;
// other backends must not see the field).
func toChatRequest(req messageRequest, echoReasoning bool) chatRequest {
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
			var reasoning []string
			for _, b := range blocks {
				switch b.Type {
				case "tool_use":
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
				case "reasoning":
					if strings.TrimSpace(b.Text) != "" {
						reasoning = append(reasoning, b.Text)
					}
				}
			}
			if echoReasoning && len(reasoning) > 0 {
				assistant.ReasoningContent = strings.Join(reasoning, "\n")
			}
			// DeepSeek (and strict OpenAI-compatible endpoints) reject
			// assistant messages with neither content nor tool_calls, so
			// drop them: they carry no information the model can use.
			if strings.TrimSpace(assistant.Content) == "" && len(assistant.ToolCalls) == 0 {
				continue
			}
			out.Messages = append(out.Messages, assistant)
		default: // user (and anything else) maps to user/tool messages
			if text != "" || len(blocks) == 0 {
				if strings.TrimSpace(text) == "" && len(blocks) == 0 {
					// An empty user turn carries nothing; skip it
					// rather than sending a content-less message.
					continue
				}
				if strings.TrimSpace(text) != "" {
					out.Messages = append(out.Messages, chatMessage{Role: "user", Content: text})
				}
			}
			for _, b := range blocks {
				if b.Type != "tool_result" {
					continue
				}
				content := b.Content
				if strings.TrimSpace(content) == "" {
					content = "(no output)"
				}
				out.Messages = append(out.Messages, chatMessage{
					Role:       "tool",
					Content:    content,
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
	text := choice.Message.Content
	if strings.TrimSpace(text) == "" {
		// DeepSeek thinking-mode responses put the chain of thought in
		// reasoning_content with an empty content; prefer it over an
		// empty final answer so the turn still carries text.
		text = choice.Message.ReasoningContent
	}
	if text != "" {
		resp.Content = append(resp.Content, contentBlock{Type: "text", Text: text})
	}
	// Keep the raw chain of thought alongside the display text so
	// reasoning-aware backends get it echoed back on the next turn.
	// responseText and the reviewer transcript only read text blocks.
	if strings.TrimSpace(choice.Message.ReasoningContent) != "" {
		resp.Content = append(resp.Content, contentBlock{Type: "reasoning", Text: choice.Message.ReasoningContent})
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
