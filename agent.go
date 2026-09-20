package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

type config struct {
	model       string
	maxTokens   int
	maxIters    int
	extraSystem string
	// effort is reasoning effort (low/medium/high, "" = backend default).
	effort string
	// sandbox runs bash under the OS sandbox. Zero value off; run() sets
	// it from --no-sandbox so production defaults to on.
	sandbox bool
}

func systemPrompt(extra string) string {
	cwd, _ := os.Getwd()
	var sb strings.Builder
	sb.WriteString("You are Harnais, a concise coding agent operating through a terminal harness.\n")
	sb.WriteString("You have exactly four tools: bash (run shell commands), read (read files), edit (exact unique-match file edit), write (create/overwrite files).\n")
	sb.WriteString("Guidelines: read a file before editing it; prefer edit over write for existing files; make the smallest change that solves the task; verify work by running the project's own build, tests, or checks; keep final answers short and state which files changed and which commands you ran.\n")
	sb.WriteString("Do not claim an action succeeded without tool output proving it. Do not run interactive or long-running foreground commands.\n")
	sb.WriteString("bash runs sandboxed (no network, writes confined to the work directory); the reviewer may let a single call run outside the sandbox when it genuinely needs network or outside files — a sandbox denial names this, so retry the call rather than working around it.\n")
	fmt.Fprintf(&sb, "Working directory: %s\n", cwd)
	if strings.TrimSpace(extra) != "" {
		sb.WriteString("Additional instructions:\n" + strings.TrimSpace(extra) + "\n")
	}
	return sb.String()
}

// messageSender is the model backend. *anthropicClient is the real one;
// tests substitute a scripted fake.
type messageSender interface {
	createMessage(ctx context.Context, req messageRequest) (*messageResponse, error)
}

// modelLister is the optional backend capability behind /models and TAB
// completion: the model IDs its list endpoint returns. Backends without
// one (codex) simply don't implement it.
type modelLister interface {
	listModels(ctx context.Context) ([]string, error)
}

// progressFunc reports agent activity to the user.
type progressFunc func(format string, args ...any)

// runPrompt runs one agentic turn: it appends the user prompt to history,
// loops model calls with tool execution, and returns the final text answer
// plus the updated history. Mutating tool calls pass a reviewer agent first
// (see review.go); reads run free. There is deliberately no human
// confirmation step and no per-tool policy matrix.
func runPrompt(ctx context.Context, sender messageSender, cfg config, history []message, prompt string, report progressFunc) (string, []message, error) {
	history = append(history, textMessage("user", prompt))
	consecutiveDenials := 0
	// maxIters <= 0 means no cap: the loop ends when the model stops asking
	// for tools, which is the only stop a real task should hit.
	for i := 0; cfg.maxIters <= 0 || i < cfg.maxIters; i++ {
		spin := startSpinner("thinking")
		resp, err := sender.createMessage(ctx, messageRequest{
			Model:     cfg.model,
			MaxTokens: cfg.maxTokens,
			System:    systemPrompt(cfg.extraSystem),
			Messages:  history,
			Tools:     toolDefinitions(),
			Effort:    cfg.effort,
		})
		spin.finish()
		if err != nil {
			return "", history, err
		}
		history = append(history, blocksMessage("assistant", resp.Content))

		if resp.StopReason != "tool_use" {
			reportTruncation(resp.StopReason, cfg, report)
			return responseText(resp), history, nil
		}

		var results []contentBlock
		for _, block := range resp.Content {
			if block.Type != "tool_use" {
				continue
			}
			var input map[string]any
			if len(block.Input) > 0 {
				if err := json.Unmarshal(block.Input, &input); err != nil {
					results = append(results, toolError(block.ID, fmt.Sprintf("invalid tool input: %v", err)))
					continue
				}
			}
			escalated := false
			summary := summarizeInput(block.Name, input)
			call := paint(ansiCyan, fmt.Sprintf("● %s(%s)", block.Name, summary))
			if block.Name == "bash" && !cfg.sandbox {
				call += " " + paint(ansiYellow, "[no sandbox]")
			}
			report("%s", call)
			if isMutating(block.Name) {
				approved, esc, rationale, rerr := reviewToolCall(ctx, sender, cfg, history, block.Name, input)
				if rerr != nil {
					return "", history, rerr
				}
				if !approved {
					consecutiveDenials++
					results = append(results, toolError(block.ID,
						fmt.Sprintf("Reviewer denied this action: %s. Do not pursue the same outcome via workaround, indirect execution, or policy circumvention; continue only with a materially safer alternative, otherwise stop.", rationale)))
					if rationale == "" {
						report("%s", paint(ansiYellow, "  review: denied"))
					} else {
						report("%s", paint(ansiYellow, fmt.Sprintf("  review: denied — %s", firstLine(rationale))))
					}
					if consecutiveDenials >= reviewDenialsToAbort {
						return "", history, fmt.Errorf("reviewer denied %d actions in a row — turn aborted", consecutiveDenials)
					}
					continue
				}
				escalated = esc
				if escalated && block.Name == "bash" && cfg.sandbox {
					report("%s", paint(ansiYellow, "  review: approved, escalated outside the sandbox"))
				} else if rationale == "" {
					report("  review: approved")
				} else {
					report("  review: approved — %s", firstLine(rationale))
				}
			}
			out, err := executeToolSandboxed(ctx, block.Name, input, sandboxForCall(cfg.sandbox, block.Name, escalated))
			consecutiveDenials = 0
			if err != nil {
				if out == "" {
					out = err.Error()
				}
				results = append(results, toolError(block.ID, out))
				report("%s", paint(ansiRed, fmt.Sprintf("  error: %s", firstLine(err.Error()))))
				continue
			}
			results = append(results, contentBlock{Type: "tool_result", ToolUseID: block.ID, Content: out})
			report("  %s", firstLine(out))
		}
		if len(results) == 0 {
			return "", history, fmt.Errorf("model stopped for tool use but supplied no tool calls")
		}
		history = append(history, blocksMessage("user", results))
	}
	return "", history, iterationCapError{limit: cfg.maxIters}
}

// iterationCapError reports a turn that ran out of its per-prompt budget
// with tool results still pending. It is a soft stop, not a failure: the
// history comes back with the turn's work in it, so the next prompt picks
// up where this one stopped instead of starting over.
type iterationCapError struct {
	limit int
}

func (e iterationCapError) Error() string {
	return fmt.Sprintf("stopped at the iteration cap (%d) with work in progress — send another message and it will carry on from here (raise or drop the cap with -n)", e.limit)
}

// reportTruncation says so when a stop reason means the model was cut off
// mid-answer rather than finishing. Nothing in harnais caps a reply by
// default, so this fires only when -max-tokens is set or a model runs into
// its own ceiling — either way the answer on screen is incomplete and the
// user is the one who needs to know.
func reportTruncation(stopReason string, cfg config, report progressFunc) {
	switch stopReason {
	case "max_tokens", "length": // Anthropic, then OpenAI-compatible backends
	default:
		return
	}
	if cfg.maxTokens > 0 {
		report("%s", paint(ansiYellow, fmt.Sprintf(
			"answer cut off at the -max-tokens cap (%d) — raise it, or drop -max-tokens to let the model finish", cfg.maxTokens)))
		return
	}
	report("%s", paint(ansiYellow,
		"answer cut off at the model's own output limit — ask for the rest, or have it write to a file in parts"))
}

func responseText(resp *messageResponse) string {
	var sb strings.Builder
	for _, block := range resp.Content {
		if block.Type == "text" {
			sb.WriteString(block.Text)
		}
	}
	return strings.TrimSpace(sb.String())
}

func toolError(id, msg string) contentBlock {
	isErr := true
	return contentBlock{Type: "tool_result", ToolUseID: id, Content: msg, IsError: &isErr}
}

func summarizeInput(tool string, input map[string]any) string {
	switch tool {
	case "bash":
		return strArg(input, "command")
	case "read", "edit", "write":
		return strArg(input, "path")
	default:
		raw, _ := json.Marshal(input)
		return string(raw)
	}
}

func firstLine(s string) string {
	s = strings.TrimSpace(s)
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	if len(s) > 160 {
		return s[:160] + "…"
	}
	return s
}
