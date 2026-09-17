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
	skipConfirm bool
	extraSystem string
}

func systemPrompt(extra string) string {
	cwd, _ := os.Getwd()
	var sb strings.Builder
	sb.WriteString("You are Harnais, a concise coding agent operating through a terminal harness.\n")
	sb.WriteString("You have exactly four tools: bash (run shell commands), read (read files), edit (exact unique-match file edit), write (create/overwrite files).\n")
	sb.WriteString("Guidelines: read a file before editing it; prefer edit over write for existing files; make the smallest change that solves the task; verify work by running the project's own build, tests, or checks; keep final answers short and state which files changed and which commands you ran.\n")
	sb.WriteString("Do not claim an action succeeded without tool output proving it. Do not run interactive or long-running foreground commands.\n")
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

// confirmFunc decides whether a mutating tool call may run.
type confirmFunc func(tool, summary string) bool

// progressFunc reports agent activity to the user.
type progressFunc func(format string, args ...any)

// runPrompt runs one agentic turn: it appends the user prompt to history,
// loops model calls with tool execution, and returns the final text answer
// plus the updated history.
func runPrompt(ctx context.Context, sender messageSender, cfg config, history []message, prompt string, confirm confirmFunc, report progressFunc) (string, []message, error) {
	history = append(history, textMessage("user", prompt))
	for i := 0; i < cfg.maxIters; i++ {
		resp, err := sender.createMessage(ctx, messageRequest{
			Model:     cfg.model,
			MaxTokens: cfg.maxTokens,
			System:    systemPrompt(cfg.extraSystem),
			Messages:  history,
			Tools:     toolDefinitions(),
		})
		if err != nil {
			return "", history, err
		}
		history = append(history, blocksMessage("assistant", resp.Content))

		if resp.StopReason != "tool_use" {
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
			summary := summarizeInput(block.Name, input)
			report("● %s(%s)", block.Name, summary)
			if isMutating(block.Name) && !confirm(block.Name, summary) {
				results = append(results, toolError(block.ID, "denied by user; do not retry without asking"))
				report("  denied")
				continue
			}
			out, err := executeTool(ctx, block.Name, input)
			if err != nil {
				if out == "" {
					out = err.Error()
				}
				results = append(results, toolError(block.ID, out))
				report("  error: %s", firstLine(err.Error()))
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
	return "", history, fmt.Errorf("reached iteration limit (%d) without a final answer", cfg.maxIters)
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

func isMutating(tool string) bool {
	switch tool {
	case "bash", "edit", "write":
		return true
	default:
		return false
	}
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
