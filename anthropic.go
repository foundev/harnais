package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
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
	// Thinking and Signature carry extended-thinking blocks. The API
	// requires them back byte-for-byte when a turn is replayed, so they
	// round-trip through history untouched.
	Thinking  string `json:"thinking,omitempty"`
	Signature string `json:"signature,omitempty"`
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
	Model string `json:"model"`
	// MaxTokens exists for the one backend that insists on a number:
	// Anthropic's Messages API. Nothing else sets it, and the OpenAI-shaped
	// wire struct has no such field at all (see toChatRequest).
	MaxTokens int `json:"max_tokens"`
	// Stream asks for the streaming Messages API, which is how every
	// anthropic request is sent (see buildAnthropicRequest).
	Stream   bool             `json:"stream,omitempty"`
	System   string           `json:"system,omitempty"`
	Messages []message        `json:"messages"`
	Tools    []toolDefinition `json:"tools,omitempty"`
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
		// No total request timeout: answers arrive over a stream, and a
		// deadline on the whole request would cut off a long, healthy
		// generation. A stalled stream is caught by the idle watchdog
		// instead (see anthropicStreamIdle).
		http:    &http.Client{},
		apiKey:  apiKey,
		baseURL: baseURL,
		retry:   defaultRetryPolicy(),
	}
}

// anthropicMaxOutputTokens is the max_tokens every request carries: the
// Messages API takes a number, so "uncapped" is expressed as the largest
// output current Claude models allow (128K).
const anthropicMaxOutputTokens = 128000

// buildAnthropicRequest projects harness-internal effort onto the wire
// shape, and fills in the one field Anthropic wants a number for. Pure so
// the mapping stays testable without network.
func buildAnthropicRequest(req messageRequest) messageRequest {
	if req.Effort != "" {
		req.OutputConfig = &outputConfig{Effort: req.Effort}
	}
	if req.MaxTokens <= 0 {
		req.MaxTokens = anthropicMaxOutputTokens
	}
	// Always stream: a long answer must not have to arrive in one burst.
	req.Stream = true
	return req
}

func (c *anthropicClient) createMessage(ctx context.Context, req messageRequest) (*messageResponse, error) {
	body, err := json.Marshal(buildAnthropicRequest(req))
	if err != nil {
		return nil, fmt.Errorf("encode request: %w", err)
	}
	var lastErr error
	for attempt := 0; ; attempt++ {
		out, err := c.streamMessage(ctx, body)
		if err == nil {
			return out, nil
		}
		lastErr = err

		// Nothing has been handed to the caller yet — the stream is
		// assembled in memory — so a retry cannot duplicate output. It can
		// still be a wasted attempt, so it stays bounded.
		var api *apiError
		switch {
		case errors.As(err, &api):
			if !api.retryable || attempt >= c.retry.maxAttempts {
				return nil, lastErr
			}
		case c.retry.retryTransport && attempt < c.retry.maxAttempts:
		default:
			return nil, lastErr
		}
		if ctx.Err() != nil {
			return nil, lastErr
		}
		if serr := sleepRetry(ctx, c.retry.delayFor(attempt+1)); serr != nil {
			return nil, lastErr
		}
	}
}

// apiError is a refusal from the Messages API: the body carries the API's
// own message, and retryable says whether another attempt could help.
type apiError struct {
	status    int
	message   string
	retryable bool
}

func (e *apiError) Error() string {
	return fmt.Sprintf("api error %d: %s", e.status, e.message)
}

// anthropicStreamIdle is how long a streaming request may go without a
// byte before harnais gives up on it. The API emits pings while a model
// thinks, so this much silence means the connection is gone — unlike a
// deadline on the whole request, it never cuts off a slow-but-alive
// answer. A variable so tests can shrink it.
var anthropicStreamIdle = 2 * time.Minute

// streamMessage posts one streaming Messages request and assembles its
// events into the same messageResponse the non-streaming API would return.
// Streaming is what keeps a long answer alive: bytes keep moving, so no
// idle timeout can kill a generation that is still producing.
func (c *anthropicClient) streamMessage(ctx context.Context, body []byte) (*messageResponse, error) {
	// Each attempt gets its own cancelable context so the idle watchdog can
	// drop this attempt without poisoning the caller's context, which a
	// retry still needs.
	attemptCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	idle := time.AfterFunc(anthropicStreamIdle, cancel)
	defer idle.Stop()

	httpReq, err := http.NewRequestWithContext(attemptCtx, http.MethodPost, c.baseURL+"/v1/messages", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")
	httpReq.Header.Set("X-Api-Key", c.apiKey)
	httpReq.Header.Set("Anthropic-Version", anthropicVersion)

	resp, err := c.http.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("api call: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		return nil, &apiError{
			status:    resp.StatusCode,
			message:   truncateOutput(string(raw)),
			retryable: c.retry.retryableStatus(resp.StatusCode),
		}
	}
	out, err := readAnthropicStream(&idleReader{r: resp.Body, timer: idle})
	if err != nil {
		// A stall is the watchdog firing, not a broken connection: say so
		// rather than reporting a bare "context canceled".
		if ctx.Err() == nil && attemptCtx.Err() != nil {
			return nil, fmt.Errorf("api call: stream went quiet for %s and was dropped: %w", anthropicStreamIdle, err)
		}
		return nil, err
	}
	return out, nil
}

// idleReader feeds the stream and pushes the watchdog back on every byte,
// so only silence cancels an attempt.
type idleReader struct {
	r     io.Reader
	timer *time.Timer
}

func (r *idleReader) Read(p []byte) (int, error) {
	n, err := r.r.Read(p)
	if n > 0 {
		r.timer.Reset(anthropicStreamIdle)
	}
	return n, err
}

// streamState accumulates one streaming response into the message shape the
// rest of the harness speaks. Events arrive per content block by index, and
// deltas apply in order, so blocks are rebuilt as they stream.
type streamState struct {
	out    messageResponse
	blocks map[int]*contentBlock
}

// streamEvent is one server-sent event. Every event names itself in `type`,
// so a single shape decodes them all; fields an event does not carry stay
// zero.
type streamEvent struct {
	Type    string `json:"type"`
	Index   int    `json:"index"`
	Message struct {
		ID    string `json:"id"`
		Usage struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
		} `json:"usage"`
	} `json:"message"`
	ContentBlock struct {
		Type string `json:"type"`
		ID   string `json:"id"`
		Name string `json:"name"`
	} `json:"content_block"`
	Delta struct {
		Type        string `json:"type"`
		Text        string `json:"text"`
		Thinking    string `json:"thinking"`
		Signature   string `json:"signature"`
		PartialJSON string `json:"partial_json"`
		StopReason  string `json:"stop_reason"`
	} `json:"delta"`
	Usage struct {
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
	Error struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

// readAnthropicStream turns an SSE body into a message response. A proxy
// that ignores "stream": true answers with the plain JSON message instead,
// so that shape is decoded too rather than failing the turn.
func readAnthropicStream(r io.Reader) (*messageResponse, error) {
	br := bufio.NewReaderSize(r, 64*1024)
	head, err := br.ReadBytes('\n')
	if err != nil && len(head) == 0 {
		return nil, fmt.Errorf("read stream: %w", err)
	}
	if !looksLikeSSE(head) {
		rest, rerr := io.ReadAll(io.LimitReader(br, 16<<20))
		if rerr != nil {
			return nil, fmt.Errorf("read response: %w", rerr)
		}
		var out messageResponse
		if err := json.Unmarshal(append(head, rest...), &out); err != nil {
			return nil, fmt.Errorf("decode response: %w", err)
		}
		return &out, nil
	}

	state := &streamState{blocks: map[int]*contentBlock{}}
	line := head
	for {
		if data, ok := sseData(line); ok {
			var ev streamEvent
			if err := json.Unmarshal(data, &ev); err != nil {
				return nil, fmt.Errorf("decode stream event: %w", err)
			}
			if err := state.apply(ev); err != nil {
				return nil, err
			}
		}
		if err != nil {
			if err == io.EOF {
				break
			}
			return nil, fmt.Errorf("read stream: %w", err)
		}
		line, err = br.ReadBytes('\n')
	}
	return state.response(), nil
}

// looksLikeSSE tells the streaming body from the plain-JSON fallback: SSE
// lines are data, event, id, or a comment, and a JSON body is none of them.
func looksLikeSSE(line []byte) bool {
	trimmed := bytes.TrimSpace(line)
	if len(trimmed) == 0 {
		return true
	}
	for _, field := range []string{"data:", "event:", "id:", ":"} {
		if bytes.HasPrefix(trimmed, []byte(field)) {
			return true
		}
	}
	return false
}

// sseData extracts the JSON payload of one SSE line, reporting whether the
// line carried any. Other fields (event, id), comments, and blank lines are
// skipped: the payload repeats the event type.
func sseData(line []byte) ([]byte, bool) {
	data := bytes.TrimSpace(line)
	if !bytes.HasPrefix(data, []byte("data:")) {
		return nil, false
	}
	data = bytes.TrimSpace(data[len("data:"):])
	if len(data) == 0 || bytes.Equal(data, []byte("[DONE]")) {
		return nil, false
	}
	return data, true
}

func (s *streamState) apply(ev streamEvent) error {
	switch ev.Type {
	case "message_start":
		s.out.ID = ev.Message.ID
		s.out.Usage.InputTokens = ev.Message.Usage.InputTokens
	case "content_block_start":
		block := s.block(ev.Index)
		block.Type = ev.ContentBlock.Type
		block.ID = ev.ContentBlock.ID
		block.Name = ev.ContentBlock.Name
	case "content_block_delta":
		block := s.block(ev.Index)
		switch ev.Delta.Type {
		case "text_delta":
			block.Text += ev.Delta.Text
		case "thinking_delta":
			block.Thinking += ev.Delta.Thinking
		case "signature_delta":
			block.Signature += ev.Delta.Signature
		case "input_json_delta":
			block.Input = json.RawMessage(string(block.Input) + ev.Delta.PartialJSON)
		}
	case "message_delta":
		if ev.Delta.StopReason != "" {
			s.out.StopReason = ev.Delta.StopReason
		}
		s.out.Usage.OutputTokens = ev.Usage.OutputTokens
	case "error":
		return fmt.Errorf("stream error: %s (%s)", ev.Error.Message, ev.Error.Type)
	}
	return nil
}

// block returns the accumulator for one content block, creating it on first
// mention so a delta that arrives without its start event is not lost.
func (s *streamState) block(index int) *contentBlock {
	if b, ok := s.blocks[index]; ok {
		return b
	}
	b := &contentBlock{}
	s.blocks[index] = b
	return b
}

// response assembles the blocks in index order — the order the model
// produced them, which history replay and tool execution both depend on.
func (s *streamState) response() *messageResponse {
	indexes := make([]int, 0, len(s.blocks))
	for index := range s.blocks {
		indexes = append(indexes, index)
	}
	sort.Ints(indexes)
	for _, index := range indexes {
		block := s.blocks[index]
		if block.Type == "tool_use" && len(block.Input) == 0 {
			// A tool call with no arguments streams no JSON at all.
			block.Input = json.RawMessage(`{}`)
		}
		s.out.Content = append(s.out.Content, *block)
	}
	return &s.out
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

// listModels fetches the backend's model IDs for live completion (see
// main.go), following the cursor until the pages run out (bounded: a broken
// has_more loop must not spin forever).
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
