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
	extraSystem string
	// effort is reasoning effort (low/medium/high, "" = backend default).
	effort string
	// sandbox runs bash under the OS sandbox. Zero value off; run() sets
	// it from --no-sandbox so production defaults to on.
	sandbox bool
	// goal is the session's explicit /goal objective, set by the user and
	// pinned as the reviewer's task. While set, the turn loop continues
	// (Codex's continue_if_idle) until the model calls update_goal.
	goal string
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
//
// With an active goal (cfg.goal), the turn does not end when the model
// stops asking for tools: like Codex's continue_if_idle, a continuation
// turn seeded with the goal follows until the model calls update_goal with
// "complete" or "blocked".
func runPrompt(ctx context.Context, sender messageSender, cfg config, history []message, prompt string, report progressFunc) (string, []message, error) {
	history = append(history, textMessage("user", prompt))
	if cfg.goal != "" {
		// The goal rides in the visible conversation, Codex-style: every
		// turn (including continuations) sees what is being pursued.
		history = append(history, textMessage("user", goalContinuationPrompt(&threadGoal{Objective: cfg.goal})))
	}
	consecutiveDenials := 0
	// No step budget: the turn ends when the model stops asking for tools
	// and — under an active goal — has not marked the goal done, which is
	// the only stop a real task should hit. Codex works the same way — its
	// sampling loop has no counter either.
	for {
		tools := toolDefinitions()
		if cfg.goal != "" {
			tools = append(tools, goalToolDefinition())
		}
		spin := startSpinner("thinking")
		resp, err := sender.createMessage(ctx, messageRequest{
			Model:    cfg.model,
			System:   systemPrompt(cfg.extraSystem),
			Messages: history,
			Tools:    tools,
			Effort:   cfg.effort,
		})
		spin.finish()
		if err != nil {
			return "", history, err
		}
		history = append(history, blocksMessage("assistant", resp.Content))

		if resp.StopReason != "tool_use" {
			reportTruncation(resp.StopReason, report)
			if cfg.goal == "" {
				return responseText(resp), history, nil
			}
			// Goal still active: re-seed and continue, Codex's
			// continue_if_idle. The previous answer stays in history as
			// context; the continuation prompt asks for real progress.
			// A turn that ran no tools is the blocked audit's unit: the
			// same impasse persisting for another consecutive turn.
			goalBlockedTurns++
			history = append(history, textMessage("user", goalContinuationPrompt(&threadGoal{Objective: cfg.goal})))
			continue
		}

		var results []contentBlock
		goalDone := ""
		madeProgress := false
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
			if block.Name == "update_goal" {
				// The goal loop's only exit. Bookkeeping, not a mutation:
				// no reviewer pass, no execution. "complete" always ends
				// the loop; "blocked" is host-enforced like Codex's
				// blocked audit — the same impasse must have survived at
				// least goalTurnsToBlock consecutive goal turns, so the
				// model cannot quit a hard goal after one bad turn. A
				// premature call is rejected as a tool error and the
				// loop continues.
				status := strings.TrimSpace(strArg(input, "status"))
				if status == "blocked" && goalBlockedTurns < goalTurnsToBlock {
					results = append(results, toolError(block.ID,
						fmt.Sprintf("blocked requires the same blocker across %d consecutive goal turns — only %d so far. Continue, or report concrete progress.", goalTurnsToBlock, goalBlockedTurns)))
					report("%s", paint(ansiYellow, fmt.Sprintf("  goal: blocked rejected (%d/%d turns)", goalBlockedTurns, goalTurnsToBlock)))
					continue
				}
				results = append(results, contentBlock{Type: "tool_result", ToolUseID: block.ID,
					Content: "goal status recorded: " + status})
				if goalFinished(status) {
					goalDone = status
				}
				report("%s", paint(ansiGreen, fmt.Sprintf("  goal %s", status)))
				continue
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
			before := fileContentBefore(block.Name, input)
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
			// A tool that actually ran is progress for the blocked audit:
			// a genuine impasse runs no tools.
			madeProgress = true
			results = append(results, contentBlock{Type: "tool_result", ToolUseID: block.ID, Content: out})
			report("  %s", firstLine(out))
			// File diffs are terminal chrome for the user; the model
			// keeps its plain tool result.
			for _, line := range fileDiffLines(block.Name, input, before) {
				report("  %s", line)
			}
		}
		if len(results) == 0 {
			return "", history, fmt.Errorf("model stopped for tool use but supplied no tool calls")
		}
		history = append(history, blocksMessage("user", results))
		// The goal loop's exit: update_goal said complete/blocked, so the
		// goal stops driving turns and the model's next end_turn returns
		// normally — one last pass to summarize, no continuation. The
		// audit streak ends with the goal: it must never leak into the
		// next goal's blocked accounting.
		if goalDone != "" {
			cfg.goal = ""
			goalBlockedTurns = 0
		}
		// Blocked audit, continued: a turn whose tools all failed also
		// persists the impasse; any successful tool run resets the streak.
		// (Turns that ended without tool calls were counted above.) A
		// resolved goal skips both: cfg.goal is already clear.
		if cfg.goal != "" {
			if madeProgress {
				goalBlockedTurns = 0
			} else {
				goalBlockedTurns++
			}
		}
	}
}

// reportTruncation says so when a stop reason means the model was cut off
// mid-answer rather than finishing. harnais sets no output cap, so this
// means the model ran into its own limit — the answer on screen is
// incomplete and the user is the one who needs to know.
func reportTruncation(stopReason string, report progressFunc) {
	switch stopReason {
	case "max_tokens", "length": // Anthropic, then OpenAI-compatible backends
	default:
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

// fileContentBefore snapshots the on-disk content an edit or write is
// about to change so the diff shows what actually happened rather than
// what the model intended. Unreadable or non-matching cases become "" —
// for write that correctly reads as a new file, and a failed edit never
// reaches the diff because it errors out.
func fileContentBefore(tool string, input map[string]any) string {
	if tool != "edit" && tool != "write" {
		return ""
	}
	raw, err := os.ReadFile(strArg(input, "path"))
	if err != nil {
		return ""
	}
	return string(raw)
}

// fileDiffLines renders the change a completed edit or write made, nil for
// tools that do not touch files.
func fileDiffLines(tool string, input map[string]any, before string) []string {
	if tool != "edit" && tool != "write" {
		return nil
	}
	var after string
	if raw, err := os.ReadFile(strArg(input, "path")); err == nil {
		after = string(raw)
	}
	return renderFileDiff(strArg(input, "path"), before, after, colorsOn())
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
