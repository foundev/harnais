package main

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

const (
	// maxToolOutput caps captured tool output so one noisy command
	// cannot blow up the context window.
	maxToolOutput = 32 * 1024
	// defaultBashTimeoutMs applies when the agent omits timeout_ms.
	defaultBashTimeoutMs = 30000
	// maxBashTimeoutMs bounds a single bash invocation.
	maxBashTimeoutMs = 300000
)

// toolDefinition mirrors the Anthropic Messages API tool schema.
type toolDefinition struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"input_schema"`
}

// toolDefinitions advertises the only four tools the agent may call.
func toolDefinitions() []toolDefinition {
	return []toolDefinition{
		{
			Name:        "bash",
			Description: "Run a shell command via sh -c and capture its combined stdout/stderr. Prefer absolute paths or the workdir argument. Never run interactive commands.",
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"command":    map[string]any{"type": "string", "description": "Shell command to run."},
					"workdir":    map[string]any{"type": "string", "description": "Working directory. Defaults to the harness working directory."},
					"timeout_ms": map[string]any{"type": "number", "description": "Timeout in milliseconds, 1000-300000. Defaults to 30000."},
				},
				"required":             []string{"command"},
				"additionalProperties": false,
			},
		},
		{
			Name:        "read",
			Description: "Read a file window with 1-based line numbers. Directories cannot be read; use bash ls instead.",
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"path":   map[string]any{"type": "string", "description": "File path, absolute or relative to the harness working directory."},
					"offset": map[string]any{"type": "number", "description": "First line to read, 1-based. Defaults to 1."},
					"limit":  map[string]any{"type": "number", "description": "Max lines to read, 1-2000. Defaults to 500."},
				},
				"required":             []string{"path"},
				"additionalProperties": false,
			},
		},
		{
			Name:        "edit",
			Description: "Replace one exact, unique occurrence of text in an existing file. Fails when the match is missing or ambiguous; read the file first and choose a larger unique context.",
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"path":    map[string]any{"type": "string", "description": "File path, absolute or relative to the harness working directory."},
					"find":    map[string]any{"type": "string", "description": "Exact text to replace. Must occur exactly once."},
					"replace": map[string]any{"type": "string", "description": "Replacement text."},
				},
				"required":             []string{"path", "find", "replace"},
				"additionalProperties": false,
			},
		},
		{
			Name:        "write",
			Description: "Create or overwrite a file with the given content, creating parent directories as needed.",
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"path":    map[string]any{"type": "string", "description": "File path, absolute or relative to the harness working directory."},
					"content": map[string]any{"type": "string", "description": "Complete file content."},
				},
				"required":             []string{"path", "content"},
				"additionalProperties": false,
			},
		},
	}
}

// executeTool dispatches one tool call with sandboxing off (unit-test and
// library default). The agent loop uses executeToolSandboxed so bash runs
// under the OS sandbox unless the process opted out with --no-sandbox —
// the sandbox switch is deliberately not a tool argument, so a prompt
// injection cannot talk the model into disabling it.
func executeTool(ctx context.Context, name string, input map[string]any) (string, error) {
	return executeToolSandboxed(ctx, name, input, false)
}

func executeToolSandboxed(ctx context.Context, name string, input map[string]any, sandbox bool) (string, error) {
	switch name {
	case "bash":
		return runBash(ctx, input, sandbox)
	case "read":
		return runRead(input)
	case "edit":
		return runEdit(input)
	case "write":
		return runWrite(input)
	default:
		return "", fmt.Errorf("unknown tool %q", name)
	}
}

func strArg(input map[string]any, key string) string {
	v, _ := input[key].(string)
	return v
}

func numArg(input map[string]any, key string, def float64) float64 {
	if v, ok := input[key].(float64); ok {
		return v
	}
	return def
}

func runBash(ctx context.Context, input map[string]any, sandbox bool) (string, error) {
	command := strArg(input, "command")
	if strings.TrimSpace(command) == "" {
		return "", fmt.Errorf("bash: command is required")
	}
	workdir := strArg(input, "workdir")
	if workdir == "" {
		workdir = "."
	}
	timeoutMs := numArg(input, "timeout_ms", defaultBashTimeoutMs)
	if timeoutMs < 1000 {
		timeoutMs = 1000
	}
	if timeoutMs > maxBashTimeoutMs {
		timeoutMs = maxBashTimeoutMs
	}
	ctx, cancel := context.WithTimeout(ctx, time.Duration(timeoutMs)*time.Millisecond)
	defer cancel()

	name, args := "sh", []string{"-c", command}
	sandboxed := false
	env := []string{}
	if sandbox {
		if !sandboxSupported() {
			warnSandboxUnsupported()
		} else {
			roots := sandboxRoots(workdir)
			argv := sandboxArgv(roots, sandboxProfile(roots), command)
			name, args = argv[0], argv[1:]
			sandboxed = true
			env = append(os.Environ(),
				"HARNAIS_SANDBOX=1",
				"HARNAIS_SANDBOX_NETWORK_DISABLED=1",
			)
		}
	}
	cmd := exec.CommandContext(ctx, name, args...)
	cmd.Dir = workdir
	if sandboxed {
		cmd.Env = env
	}
	var buf bytes.Buffer
	cmd.Stdout = &buf
	cmd.Stderr = &buf
	start := time.Now()
	err := cmd.Run()
	elapsed := time.Since(start)

	out := truncateOutput(buf.String())
	if out == "" {
		out = "(no output)"
	}
	// Seatbelt denials surface as cryptic "operation not permitted" exits;
	// say what actually happened and name the escape hatch.
	hint := ""
	if sandboxed {
		hint = "\n(sandboxed: denied by the OS sandbox — retry the call so the reviewer can approve it outside the sandbox if it genuinely needs network or files outside the workdir)"
	}
	if ctx.Err() == context.DeadlineExceeded {
		return fmt.Sprintf("timeout after %v\n--- output ---\n%s%s", elapsed.Round(time.Millisecond), out, hint),
			fmt.Errorf("bash: command timed out after %v", elapsed.Round(time.Millisecond))
	}
	if err != nil {
		return fmt.Sprintf("exit: %v\nduration: %v\n--- output ---\n%s%s", err, elapsed.Round(time.Millisecond), out, hint),
			fmt.Errorf("bash: %v", err)
	}
	return fmt.Sprintf("duration: %v\n--- output ---\n%s", elapsed.Round(time.Millisecond), out), nil
}

func runRead(input map[string]any) (string, error) {
	path := strArg(input, "path")
	if path == "" {
		return "", fmt.Errorf("read: path is required")
	}
	offset := int(numArg(input, "offset", 1))
	limit := int(numArg(input, "limit", 500))
	if offset < 1 {
		offset = 1
	}
	if limit < 1 {
		limit = 1
	}
	if limit > 2000 {
		limit = 2000
	}
	info, err := os.Stat(path)
	if err != nil {
		return "", fmt.Errorf("read: %v", err)
	}
	if info.IsDir() {
		return "", fmt.Errorf("read: %s is a directory, use bash ls to list it", path)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read: %v", err)
	}
	lines := strings.Split(string(raw), "\n")
	total := len(lines)
	if offset > total {
		return "", fmt.Errorf("read: offset %d beyond end of file (%d lines)", offset, total)
	}
	end := offset - 1 + limit
	if end > total {
		end = total
	}
	var sb strings.Builder
	for i := offset - 1; i < end; i++ {
		fmt.Fprintf(&sb, "%d|%s\n", i+1, lines[i])
	}
	if end < total {
		fmt.Fprintf(&sb, "(lines %d-%d of %d, use offset %d for more)\n", offset, end, total, end+1)
	}
	return sb.String(), nil
}

func runEdit(input map[string]any) (string, error) {
	path := strArg(input, "path")
	find := strArg(input, "find")
	replace, hasReplace := input["replace"].(string)
	if path == "" {
		return "", fmt.Errorf("edit: path is required")
	}
	if find == "" {
		return "", fmt.Errorf("edit: find must not be empty")
	}
	if !hasReplace {
		return "", fmt.Errorf("edit: replace is required (use empty string to delete)")
	}
	info, err := os.Stat(path)
	if err != nil {
		return "", fmt.Errorf("edit: %v (use write to create new files)", err)
	}
	if info.IsDir() {
		return "", fmt.Errorf("edit: %s is a directory", path)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("edit: %v", err)
	}
	count := strings.Count(string(raw), find)
	switch {
	case count == 0:
		return "", fmt.Errorf("edit: match not found in %s", path)
	case count > 1:
		return "", fmt.Errorf("edit: match occurs %d times in %s, must be unique — add surrounding context", count, path)
	}
	updated := strings.Replace(string(raw), find, replace, 1)
	if err := os.WriteFile(path, []byte(updated), info.Mode().Perm()); err != nil {
		return "", fmt.Errorf("edit: %v", err)
	}
	abs, _ := filepath.Abs(path)
	return fmt.Sprintf("edited %s", abs), nil
}

func runWrite(input map[string]any) (string, error) {
	path := strArg(input, "path")
	content, ok := input["content"].(string)
	if path == "" {
		return "", fmt.Errorf("write: path is required")
	}
	if !ok {
		return "", fmt.Errorf("write: content is required")
	}
	if dir := filepath.Dir(path); dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return "", fmt.Errorf("write: %v", err)
		}
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		return "", fmt.Errorf("write: %v", err)
	}
	abs, _ := filepath.Abs(path)
	return fmt.Sprintf("wrote %s (%d bytes)", abs, len(content)), nil
}

func truncateOutput(s string) string {
	if len(s) <= maxToolOutput {
		return s
	}
	return s[:maxToolOutput] + fmt.Sprintf("\n(output truncated, %d of %d bytes shown)", maxToolOutput, len(s))
}
