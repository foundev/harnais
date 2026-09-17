package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Codex subscription support reuses the login Codex CLI already performed:
// `codex login` writes OAuth tokens to $CODEX_HOME/auth.json (default
// ~/.codex/auth.json), and harnais reads them read-only, refreshing the
// access token via the refresh-token grant when it nears expiry (exactly
// like Codex CLI does, writing back atomically so both tools keep working).
//
// Requests go to the ChatGPT backend Responses API with the account header
// Codex itself sends. Wire shape confirmed against codex-cli 0.154.0:
// base https://chatgpt.com/backend-api/codex, chatgpt-account-id and
// originator headers, token endpoint https://auth.openai.com/oauth/token.

const defaultCodexBaseURL = "https://chatgpt.com/backend-api/codex"

// defaultCodexModel applies when -model is unset with -provider=codex.
const defaultCodexModel = "gpt-5.3-codex"

// codexOAuthClientID is Codex CLI's public OAuth client, observed in the
// access-token JWTs `codex login` mints.
const codexOAuthClientID = "app_EMoamEEZ73f0CkXaXp7hrann"

const codexTokenURL = "https://auth.openai.com/oauth/token"

// codexRefreshSkew refreshes the access token this far before its exp claim.
const codexRefreshSkew = 300 * time.Second

type codexTokens struct {
	IDToken      string `json:"id_token"`
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	AccountID    string `json:"account_id"`
}

type codexAuthFile struct {
	AuthMode    string       `json:"auth_mode"`
	APIKey      *string      `json:"OPENAI_API_KEY"`
	Tokens      *codexTokens `json:"tokens"`
	LastRefresh string       `json:"last_refresh"`
}

// codexAuthPath resolves the auth file Codex CLI maintains.
func codexAuthPath() (string, error) {
	if dir := os.Getenv("CODEX_HOME"); dir != "" {
		return filepath.Join(dir, "auth.json"), nil
	}
	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		return "", fmt.Errorf("cannot locate ~/.codex/auth.json — set CODEX_HOME")
	}
	return filepath.Join(home, ".codex", "auth.json"), nil
}

// loadCodexTokens reads OAuth tokens without touching them. A missing file
// or API-key-mode login errors out pointing at `codex login`.
func loadCodexTokens(path string) (*codexTokens, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read codex auth %s: %v — run `codex login` first", path, err)
	}
	var file codexAuthFile
	if err := json.Unmarshal(raw, &file); err != nil {
		return nil, fmt.Errorf("parse codex auth %s: %v", path, err)
	}
	if file.Tokens == nil || file.Tokens.AccessToken == "" || file.Tokens.RefreshToken == "" {
		return nil, fmt.Errorf("no ChatGPT login in %s (mode %q) — run `codex login` with your subscription, not an API key", path, file.AuthMode)
	}
	return file.Tokens, nil
}

// codexJWTExp returns the exp claim of an (unverified) JWT access token.
// The signature is never checked: the token is opaque bearer material that
// only the server validates; harnais needs exp solely for refresh timing.
func codexJWTExp(accessToken string) (time.Time, error) {
	parts := strings.Split(accessToken, ".")
	if len(parts) != 3 {
		return time.Time{}, fmt.Errorf("malformed JWT")
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return time.Time{}, fmt.Errorf("decode JWT payload: %v", err)
	}
	var claims struct {
		Exp int64 `json:"exp"`
	}
	if err := json.Unmarshal(payload, &claims); err != nil {
		return time.Time{}, fmt.Errorf("decode JWT claims: %v", err)
	}
	if claims.Exp == 0 {
		return time.Time{}, fmt.Errorf("JWT has no exp claim")
	}
	return time.Unix(claims.Exp, 0), nil
}

// codexAccountID resolves the ChatGPT account, preferring the stored value
// and falling back to the JWT's auth claim (as Codex CLI does).
func codexAccountID(tok *codexTokens) (string, error) {
	if tok.AccountID != "" {
		return tok.AccountID, nil
	}
	parts := strings.Split(tok.AccessToken, ".")
	if len(parts) != 3 {
		return "", fmt.Errorf("no account id stored and access token is malformed — run `codex login` again")
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return "", fmt.Errorf("no account id stored and token is undecodable — run `codex login` again")
	}
	var claims struct {
		Auth struct {
			AccountID string `json:"chatgpt_account_id"`
		} `json:"https://api.openai.com/auth"`
	}
	if err := json.Unmarshal(payload, &claims); err != nil || claims.Auth.AccountID == "" {
		return "", fmt.Errorf("no account id stored and none in token — run `codex login` again")
	}
	return claims.Auth.AccountID, nil
}

// refreshCodexToken exchanges the refresh token for a new access token and
// writes the pair back atomically, preserving every other auth.json field so
// Codex CLI keeps working in parallel.
func refreshCodexToken(ctx context.Context, httpClient *http.Client, path string, refreshToken string) (*codexTokens, error) {
	form := url.Values{
		"grant_type":    {"refresh_token"},
		"refresh_token": {refreshToken},
		"client_id":     {codexOAuthClientID},
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, codexTokenURL, strings.NewReader(form.Encode()))
	if err != nil {
		return nil, fmt.Errorf("build refresh request: %w", err)
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("refresh token: %w", err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return nil, fmt.Errorf("read refresh response: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("refresh token (status %d) — run `codex login` again", resp.StatusCode)
	}
	var rotated struct {
		AccessToken  string `json:"access_token"`
		RefreshToken string `json:"refresh_token"`
		IDToken      string `json:"id_token"`
	}
	if err := json.Unmarshal(raw, &rotated); err != nil || rotated.AccessToken == "" {
		return nil, fmt.Errorf("decode refresh response — run `codex login` again")
	}

	// Merge into the existing file so codex-only fields survive.
	raw, err = os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("re-read codex auth: %v", err)
	}
	var file map[string]any
	if err := json.Unmarshal(raw, &file); err != nil {
		return nil, fmt.Errorf("parse codex auth: %v", err)
	}
	tokens, _ := file["tokens"].(map[string]any)
	if tokens == nil {
		tokens = map[string]any{}
	}
	tokens["access_token"] = rotated.AccessToken
	if rotated.RefreshToken != "" {
		tokens["refresh_token"] = rotated.RefreshToken
	}
	if rotated.IDToken != "" {
		tokens["id_token"] = rotated.IDToken
	}
	file["tokens"] = tokens
	file["last_refresh"] = time.Now().UTC().Format(time.RFC3339Nano)
	return writeCodexAuthFile(path, file)
}

func writeCodexAuthFile(path string, file map[string]any) (*codexTokens, error) {
	raw, err := json.MarshalIndent(file, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("encode codex auth: %w", err)
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".auth-*.tmp")
	if err != nil {
		return nil, fmt.Errorf("write codex auth: %w", err)
	}
	tmpName := tmp.Name()
	if _, err := tmp.Write(append(raw, '\n')); err != nil {
		tmp.Close()
		os.Remove(tmpName)
		return nil, fmt.Errorf("write codex auth: %w", err)
	}
	if err := tmp.Chmod(0o600); err != nil {
		tmp.Close()
		os.Remove(tmpName)
		return nil, fmt.Errorf("write codex auth: %w", err)
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmpName)
		return nil, fmt.Errorf("write codex auth: %w", err)
	}
	if err := os.Rename(tmpName, path); err != nil {
		os.Remove(tmpName)
		return nil, fmt.Errorf("write codex auth: %w", err)
	}
	var tokens codexTokens
	rawTokens, _ := json.Marshal(file["tokens"])
	if err := json.Unmarshal(rawTokens, &tokens); err != nil {
		return nil, fmt.Errorf("decode refreshed tokens: %w", err)
	}
	return &tokens, nil
}

type codexClient struct {
	http      *http.Client
	baseURL   string
	authPath  string
	access    string
	accountID string
}

func newCodexClient(authPath, baseURL string) *codexClient {
	return &codexClient{
		http:     &http.Client{Timeout: 180 * time.Second},
		baseURL:  baseURL,
		authPath: authPath,
	}
}

// ensureAuth (re)loads tokens from disk — picking up `codex login` runs and
// Codex CLI refreshes — and rotates them when forced or near expiry.
func (c *codexClient) ensureAuth(ctx context.Context, force bool) error {
	tok, err := loadCodexTokens(c.authPath)
	if err != nil {
		return err
	}
	refresh := force
	if !refresh {
		exp, err := codexJWTExp(tok.AccessToken)
		refresh = err != nil || time.Until(exp) < codexRefreshSkew
	}
	if refresh {
		if tok.RefreshToken == "" {
			return fmt.Errorf("codex access token expired with no refresh token — run `codex login` again")
		}
		tok, err = refreshCodexToken(ctx, c.http, c.authPath, tok.RefreshToken)
		if err != nil {
			return err
		}
	}
	accountID, err := codexAccountID(tok)
	if err != nil {
		return err
	}
	c.access = tok.AccessToken
	c.accountID = accountID
	return nil
}

// dumpDebug appends a labeled body to the HARNAIS_DEBUG file when set.
// Only request/response JSON bodies are logged — auth headers and tokens
// never pass through here.
func dumpDebug(label string, body []byte) {
	path := os.Getenv("HARNAIS_DEBUG")
	if path == "" {
		return
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600)
	if err != nil {
		return
	}
	defer f.Close()
	fmt.Fprintf(f, "=== %s (%d bytes, %s) ===\n%s\n", label, len(body), time.Now().UTC().Format(time.RFC3339), body)
}

func (c *codexClient) createMessage(ctx context.Context, req messageRequest) (*messageResponse, error) {
	body, err := json.Marshal(toResponsesRequest(req))
	if err != nil {
		return nil, fmt.Errorf("encode request: %w", err)
	}
	dumpDebug("codex request", body)
	if err := c.ensureAuth(ctx, false); err != nil {
		return nil, err
	}
	for attempt := 0; ; attempt++ {
		status, raw, err := c.post(ctx, body)
		if err != nil {
			return nil, err
		}
		dumpDebug(fmt.Sprintf("codex response status=%d attempt=%d", status, attempt), raw)
		if status == http.StatusUnauthorized && attempt == 0 {
			// Token may have been revoked or rotated elsewhere; force a
			// refresh and retry once before surfacing the failure.
			if rerr := c.ensureAuth(ctx, true); rerr != nil {
				return nil, rerr
			}
			continue
		}
		if status == http.StatusUnauthorized {
			return nil, fmt.Errorf("codex backend rejected auth (401) — run `codex login` again")
		}
		if status < 200 || status >= 300 {
			return nil, codexStatusError(status, raw)
		}
		return decodeResponsesStream(raw)
	}
}

func (c *codexClient) post(ctx context.Context, body []byte) (int, []byte, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/responses", bytes.NewReader(body))
	if err != nil {
		return 0, nil, fmt.Errorf("build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+c.access)
	httpReq.Header.Set("ChatGPT-Account-ID", c.accountID)
	httpReq.Header.Set("Originator", "codex_cli_rs")
	httpReq.Header.Set("OpenAI-Beta", "responses=v1")
	httpReq.Header.Set("User-Agent", "harnais/"+version)

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

type responsesItem struct {
	Type      string             `json:"type"` // message | function_call | function_call_output
	Role      string             `json:"role,omitempty"`
	Content   []responsesContent `json:"content,omitempty"`
	CallID    string             `json:"call_id,omitempty"`
	Name      string             `json:"name,omitempty"`
	Arguments string             `json:"arguments,omitempty"`
	Output    string             `json:"output,omitempty"`
}

type responsesContent struct {
	Type string `json:"type"` // input_text | output_text
	Text string `json:"text"`
}

type responsesTool struct {
	Type        string         `json:"type"` // function
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

type responsesReasoning struct {
	Effort string `json:"effort"`
}

type responsesRequest struct {
	Model        string              `json:"model"`
	Instructions string              `json:"instructions,omitempty"`
	Reasoning    *responsesReasoning `json:"reasoning,omitempty"`
	// Store must be false: the ChatGPT backend rejects stored responses.
	// A pointer with omitempty keeps the field present-but-false rather
	// than dropping it.
	Store *bool `json:"store,omitempty"`
	// Stream must be true: the ChatGPT backend rejects non-streaming
	// responses calls.
	Stream bool            `json:"stream"`
	Input  []responsesItem `json:"input"`
	Tools  []responsesTool `json:"tools,omitempty"`
}

type responsesResponse struct {
	ID     string          `json:"id"`
	Status string          `json:"status,omitempty"`
	Output []responsesItem `json:"output"`
	Usage  *struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage,omitempty"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

// toResponsesRequest converts the harness history to Responses input items.
// Assistant tool_use blocks become function_call items; tool_result blocks
// become function_call_output items keyed by call id.
func toResponsesRequest(req messageRequest) responsesRequest {
	out := responsesRequest{Model: req.Model, Instructions: req.System, Stream: true}
	noStore := false
	out.Store = &noStore
	if req.Effort != "" {
		out.Reasoning = &responsesReasoning{Effort: req.Effort}
	}
	for _, msg := range req.Messages {
		text, blocks := splitContent(msg.Content)
		switch msg.Role {
		case "assistant":
			if text != "" {
				out.Input = append(out.Input, responsesItem{
					Type:    "message",
					Role:    "assistant",
					Content: []responsesContent{{Type: "output_text", Text: text}},
				})
			}
			for _, b := range blocks {
				if b.Type != "tool_use" {
					continue
				}
				args := string(b.Input)
				if args == "" {
					args = "{}"
				}
				out.Input = append(out.Input, responsesItem{
					Type: "function_call", CallID: b.ID, Name: b.Name, Arguments: args,
				})
			}
		default: // user (and anything else) maps to user/function_call_output items
			if text != "" || len(blocks) == 0 {
				out.Input = append(out.Input, responsesItem{
					Type:    "message",
					Role:    "user",
					Content: []responsesContent{{Type: "input_text", Text: text}},
				})
			}
			for _, b := range blocks {
				if b.Type != "tool_result" {
					continue
				}
				out.Input = append(out.Input, responsesItem{
					Type: "function_call_output", CallID: b.ToolUseID, Output: b.Content,
				})
			}
		}
	}
	for _, t := range req.Tools {
		out.Tools = append(out.Tools, responsesTool{
			Type: "function", Name: t.Name, Description: t.Description, Parameters: t.InputSchema,
		})
	}
	return out
}

// codexStatusError prefers the backend's own message — this API reports
// failures as {"detail": ...} — over a raw body dump.
func codexStatusError(status int, raw []byte) error {
	var body struct {
		Detail string `json:"detail"`
		Error  *struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(raw, &body); err == nil {
		if body.Detail != "" {
			return fmt.Errorf("api error %d: %s", status, body.Detail)
		}
		if body.Error != nil && body.Error.Message != "" {
			return fmt.Errorf("api error %d: %s", status, body.Error.Message)
		}
	}
	return fmt.Errorf("api error %d: %s", status, truncateOutput(string(raw)))
}

// decodeResponsesStream consumes a text/event-stream body and converts the
// terminal response.completed event's embedded response object. Incremental
// delta events are intentionally ignored: per the Responses API docs the
// terminal event carries the complete object including usage. The ChatGPT
// backend deviates here — observed live, it streams complete items via
// response.output_item.done and then sends response.completed with an empty
// output array — so done items are accumulated as the output fallback.
// A stream that ends without a terminal event is an error, never a silent
// success.
func decodeResponsesStream(raw []byte) (*messageResponse, error) {
	var terminal json.RawMessage
	var terminalType string
	var streamErr string
	var doneItems []responsesItem

	dispatch := func(event, data string) {
		switch event {
		case "response.output_item.done":
			var env struct {
				Item responsesItem `json:"item"`
			}
			if json.Unmarshal([]byte(data), &env) == nil && env.Item.Type != "" {
				doneItems = append(doneItems, env.Item)
			}
		case "response.completed", "response.failed", "response.incomplete":
			var env struct {
				Response json.RawMessage `json:"response"`
			}
			if json.Unmarshal([]byte(data), &env) == nil && len(env.Response) > 0 {
				terminal, terminalType = env.Response, event
			}
		case "error":
			var env struct {
				Message string `json:"message"`
				Code    string `json:"code"`
				Err     string `json:"error"`
			}
			msg := ""
			if json.Unmarshal([]byte(data), &env) == nil {
				msg = env.Message
				if msg == "" {
					msg = env.Err
				}
				if msg == "" {
					msg = env.Code
				}
			}
			if msg == "" {
				msg = truncateOutput(strings.TrimSpace(data))
			}
			streamErr = msg
		}
	}

	var event string
	var data []string
	flush := func() {
		if event != "" || len(data) > 0 {
			dispatch(event, strings.Join(data, "\n"))
		}
		event, data = "", nil
	}
	for _, line := range strings.Split(string(raw), "\n") {
		line = strings.TrimSuffix(line, "\r")
		if line == "" {
			flush()
			continue
		}
		if strings.HasPrefix(line, ":") {
			continue // keep-alive comment
		}
		name, value, _ := strings.Cut(line, ":")
		value = strings.TrimPrefix(value, " ")
		switch name {
		case "event":
			event = value
		case "data":
			data = append(data, value)
		}
	}
	flush()

	if streamErr != "" {
		return nil, fmt.Errorf("api error: %s", streamErr)
	}
	if len(terminal) > 0 {
		var r responsesResponse
		if err := json.Unmarshal(terminal, &r); err != nil {
			return nil, fmt.Errorf("decode response: %w", err)
		}
		if terminalType == "response.failed" && (r.Error == nil || r.Error.Message == "") {
			return nil, fmt.Errorf("api error: response failed")
		}
		if len(r.Output) == 0 {
			r.Output = doneItems
		}
		return responsesToMessage(&r)
	}
	// Not an event stream at all: accept a plain response object so a
	// non-streaming error body still decodes instead of confusing.
	var r responsesResponse
	trimmed := bytes.TrimSpace(raw)
	if json.Unmarshal(trimmed, &r) == nil && (len(r.Output) > 0 || r.Error != nil) {
		return responsesToMessage(&r)
	}
	return nil, fmt.Errorf("api error: stream ended without a completed response")
}

func responsesToMessage(r *responsesResponse) (*messageResponse, error) {
	if r.Error != nil && r.Error.Message != "" {
		return nil, fmt.Errorf("api error: %s", r.Error.Message)
	}
	if len(r.Output) == 0 {
		return nil, fmt.Errorf("api error: response had no output items")
	}
	resp := &messageResponse{ID: r.ID}
	if r.Usage != nil {
		resp.Usage.InputTokens = r.Usage.InputTokens
		resp.Usage.OutputTokens = r.Usage.OutputTokens
	}
	sawCall := false
	for _, item := range r.Output {
		switch item.Type {
		case "message":
			for _, c := range item.Content {
				if c.Type == "output_text" && c.Text != "" {
					resp.Content = append(resp.Content, contentBlock{Type: "text", Text: c.Text})
				}
			}
		case "function_call":
			args := item.Arguments
			if args == "" {
				args = "{}"
			}
			resp.Content = append(resp.Content, contentBlock{
				Type: "tool_use", ID: item.CallID, Name: item.Name, Input: json.RawMessage(args),
			})
			sawCall = true
		}
	}
	if sawCall {
		resp.StopReason = "tool_use"
	} else {
		resp.StopReason = "end_turn"
	}
	return resp, nil
}
