package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

// AI review replaces manual approval at the tool boundary, mirroring Codex
// auto-review in simplified form: every mutating tool call (bash/edit/write)
// goes to a separate reviewer pass over the same backend before it runs, and
// the verdict — not a human prompt — decides. Reads never need review.
//
// Deliberate simplifications vs Codex: every mutating call is reviewed
// (there is no in-sandbox free tier), the reviewer gets transcript context
// only (no read-only check tools), and there is no policy config, no
// granular categories, and no override picker — a denial feeds back and the
// model must find a safer path. No human prompts, no options matrix.

// reviewDenialsToAbort matches Codex's consecutive-denial circuit breaker.
const reviewDenialsToAbort = 3

// reviewTranscriptMessages caps how much recent history the reviewer sees.
const reviewTranscriptMessages = 10

const reviewTranscriptChars = 1500
const reviewTranscriptTotal = 12000

func isMutating(tool string) bool {
	switch tool {
	case "bash", "edit", "write":
		return true
	default:
		return false
	}
}

func reviewerSystemPrompt() string {
	return `You are reviewing a single tool call proposed by a coding agent inside a terminal harness. ` +
		`The call runs confined (writes limited to the work directory, no network); judge its intent, not the mechanism. ` +
		`Reply with exactly one line starting with APPROVE or DENY, followed by a brief rationale. ` +
		`Deny actions that: send private data, secrets, or credentials anywhere; probe for credentials, tokens, cookies, or sessions; ` +
		`broadly or persistently weaken security; risk irreversible destructive damage; or look ambiguous about any of the above. ` +
		`When in doubt, deny.`
}

// reviewToolCall asks the reviewer pass for a verdict on one proposed call.
// Transport failures are returned as errors (surfaced, like Codex timeouts,
// distinctly from denials); an unclear verdict fails closed as a denial.
func reviewToolCall(ctx context.Context, sender messageSender, cfg config, history []message, tool string, input map[string]any) (bool, string, error) {
	inputJSON, _ := json.Marshal(input)
	user := fmt.Sprintf("Proposed action: %s %s\n\nTranscript (most recent last):\n%s\n\nVerdict:",
		tool, inputJSON, renderTranscript(history))
	spin := startSpinner("reviewing")
	resp, err := sender.createMessage(ctx, messageRequest{
		Model:     cfg.model,
		MaxTokens: cfg.maxTokens,
		System:    reviewerSystemPrompt(),
		Messages:  []message{textMessage("user", user)},
		Effort:    cfg.effort,
	})
	spin.finish()
	if err != nil {
		return false, "", fmt.Errorf("reviewer unavailable: %w", err)
	}
	approved, rationale := reviewVerdict(responseText(resp))
	return approved, rationale, nil
}

// reviewVerdict parses an APPROVE/DENY verdict line. Anything else fails
// closed as a denial.
func reviewVerdict(text string) (bool, string) {
	t := strings.TrimSpace(text)
	if rest, ok := splitVerdict(t, "APPROVE"); ok {
		return true, rest
	}
	if rest, ok := splitVerdict(t, "DENY"); ok {
		return false, rest
	}
	return false, "no clear verdict: " + firstLine(t)
}

func splitVerdict(t, word string) (string, bool) {
	if len(t) < len(word) || !strings.EqualFold(t[:len(word)], word) {
		return "", false
	}
	rest := t[len(word):]
	if rest != "" && rest[0] != ' ' && rest[0] != '\t' && rest[0] != '\n' && rest[0] != ':' && rest[0] != '-' {
		return "", false
	}
	rest = strings.TrimSpace(rest)
	rest = strings.TrimPrefix(rest, ":")
	rest = strings.TrimPrefix(rest, "-")
	return strings.TrimSpace(rest), true
}

// renderTranscript condenses recent history for the reviewer: user and
// assistant text, tool calls with inputs, and tool results. Only retained
// messages are included — there is no hidden reasoning to leak.
func renderTranscript(history []message) string {
	if len(history) > reviewTranscriptMessages {
		history = history[len(history)-reviewTranscriptMessages:]
	}
	var sb strings.Builder
	total := 0
	for _, msg := range history {
		var parts []string
		var text string
		var blocks []contentBlock
		if err := json.Unmarshal(msg.Content, &text); err == nil {
			parts = []string{text}
		} else if err := json.Unmarshal(msg.Content, &blocks); err == nil {
			for _, b := range blocks {
				switch b.Type {
				case "text":
					parts = append(parts, b.Text)
				case "tool_use":
					parts = append(parts, fmt.Sprintf("called %s %s", b.Name, strings.TrimSpace(string(b.Input))))
				case "tool_result":
					parts = append(parts, fmt.Sprintf("result: %s", b.Content))
				}
			}
		} else {
			continue
		}
		for _, p := range parts {
			p = strings.TrimSpace(p)
			if p == "" {
				continue
			}
			if len(p) > reviewTranscriptChars {
				p = p[:reviewTranscriptChars] + "…"
			}
			line := msg.Role + ": " + strings.ReplaceAll(p, "\n", " / ")
			if total+len(line) > reviewTranscriptTotal {
				break
			}
			sb.WriteString(line + "\n")
			total += len(line)
		}
	}
	return strings.TrimSpace(sb.String())
}
