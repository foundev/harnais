package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strings"
)

const version = "0.1.0"

func main() {
	os.Exit(run(os.Args[1:]))
}

func run(args []string) int {
	fs := flag.NewFlagSet("harnais", flag.ContinueOnError)
	provider := fs.String("provider", envOr("HARNAIS_PROVIDER", "anthropic"), "Backend: anthropic, openrouter or deepseek.")
	model := fs.String("model", envOr("ANTHROPIC_MODEL", ""), "Model ID (default depends on -provider).")
	apiKey := fs.String("key", "", "API key (or ANTHROPIC_API_KEY / OPENROUTER_API_KEY).")
	maxTokens := fs.Int("max-tokens", 8192, "Max tokens per model response.")
	maxIters := fs.Int("n", 25, "Max agent iterations per prompt.")
	skipConfirm := fs.Bool("y", false, "Run mutating tools (bash, edit, write) without asking.")
	extraSystem := fs.String("system", "", "Extra system instructions for the agent.")
	showVersion := fs.Bool("version", false, "Print version and exit.")
	fs.Usage = func() {
		fmt.Fprintf(os.Stderr, `harnais %s — a small Claude-style agentic harness (Go, stdlib only).

Usage:
  harnais [flags] "prompt"     ask once and print the answer
  harnais [flags]              start an interactive session

Flags:
`, version)
		fs.PrintDefaults()
		fmt.Fprintf(os.Stderr, `
Environment:
  HARNAIS_PROVIDER     Backend, anthropic or openrouter (default anthropic).
  ANTHROPIC_API_KEY    API key for -provider=anthropic.
  ANTHROPIC_MODEL      Default -model value (either provider).
  ANTHROPIC_BASE_URL   Anthropic API root (default %s).
  OPENROUTER_API_KEY   API key for -provider=openrouter.
  OPENROUTER_BASE_URL  OpenRouter API root (default %s).
  OPENROUTER_REFERER   Optional HTTP-Referer header for OpenRouter rankings.
  DEEPSEEK_API_KEY     API key for -provider=deepseek.
  DEEPSEEK_BASE_URL    DeepSeek API root (default %s).

Interactive commands: /help  /reset  /exit
`, defaultBaseURL, defaultOpenRouterBaseURL, defaultDeepSeekBaseURL)
	}
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *showVersion {
		fmt.Printf("harnais %s\n", version)
		return 0
	}
	if *maxIters < 1 {
		fmt.Fprintln(os.Stderr, "harnais: -n must be at least 1.")
		return 2
	}

	sender, resolvedModel, err := resolveBackend(*provider, *apiKey, *model)
	if err != nil {
		fmt.Fprintf(os.Stderr, "harnais: %v\n", err)
		return 2
	}
	cfg := config{
		model:       resolvedModel,
		maxTokens:   *maxTokens,
		maxIters:    *maxIters,
		skipConfirm: *skipConfirm,
		extraSystem: *extraSystem,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	in := bufio.NewReader(os.Stdin)
	confirm := func(tool, summary string) bool {
		if cfg.skipConfirm {
			return true
		}
		fmt.Fprintf(os.Stderr, "run %s(%s)? [y/N] ", tool, summary)
		line, err := in.ReadString('\n')
		if err != nil {
			return false
		}
		switch strings.ToLower(strings.TrimSpace(line)) {
		case "y", "yes":
			return true
		default:
			return false
		}
	}
	report := func(format string, args ...any) {
		fmt.Fprintf(os.Stderr, format+"\n", args...)
	}

	if rest := fs.Args(); len(rest) > 0 {
		answer, _, err := runPrompt(ctx, sender, cfg, nil, strings.Join(rest, " "), confirm, report)
		if err != nil {
			fmt.Fprintf(os.Stderr, "harnais: %v\n", err)
			return 1
		}
		fmt.Println(answer)
		return 0
	}
	return repl(ctx, sender, cfg, in, confirm, report)
}

// resolveBackend picks the model backend, applying per-provider key, model,
// and base-URL defaults. It returns the sender and the resolved model ID.
func resolveBackend(provider, keyFlag, modelFlag string) (messageSender, string, error) {
	switch provider {
	case "anthropic":
		key := firstNonEmpty(keyFlag, os.Getenv("ANTHROPIC_API_KEY"))
		if key == "" {
			return nil, "", fmt.Errorf("missing API key — set ANTHROPIC_API_KEY or pass -key")
		}
		model := firstNonEmpty(modelFlag, "claude-sonnet-4-20250514")
		return newAnthropicClient(key, envOr("ANTHROPIC_BASE_URL", defaultBaseURL)), model, nil
	case "openrouter":
		key := firstNonEmpty(keyFlag, os.Getenv("OPENROUTER_API_KEY"))
		if key == "" {
			return nil, "", fmt.Errorf("missing API key — set OPENROUTER_API_KEY or pass -key")
		}
		model := firstNonEmpty(modelFlag, defaultOpenRouterModel)
		return newOpenRouterClient(key, envOr("OPENROUTER_BASE_URL", defaultOpenRouterBaseURL)), model, nil
	case "deepseek":
		key := firstNonEmpty(keyFlag, os.Getenv("DEEPSEEK_API_KEY"))
		if key == "" {
			return nil, "", fmt.Errorf("missing API key — set DEEPSEEK_API_KEY or pass -key")
		}
		model := firstNonEmpty(modelFlag, defaultDeepSeekModel)
		return newDeepSeekClient(key, envOr("DEEPSEEK_BASE_URL", defaultDeepSeekBaseURL)), model, nil
	default:
		return nil, "", fmt.Errorf("unknown provider %q — use anthropic, openrouter or deepseek", provider)
	}
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}

func repl(ctx context.Context, sender messageSender, cfg config, in *bufio.Reader, confirm confirmFunc, report progressFunc) int {
	var history []message
	fmt.Fprintf(os.Stderr, "harnais %s — model %s (type /help for commands)\n", version, cfg.model)
	for {
		fmt.Fprint(os.Stderr, "> ")
		line, err := in.ReadString('\n')
		if err != nil {
			fmt.Fprintln(os.Stderr)
			return 0
		}
		prompt := strings.TrimSpace(line)
		switch prompt {
		case "":
			continue
		case "/exit", "/quit":
			return 0
		case "/reset":
			history = nil
			fmt.Fprintln(os.Stderr, "conversation reset")
			continue
		case "/help":
			fmt.Fprintln(os.Stderr, "Enter a prompt per line. Commands: /help /reset /exit")
			continue
		}
		answer, updated, err := runPrompt(ctx, sender, cfg, history, prompt, confirm, report)
		if err != nil {
			fmt.Fprintf(os.Stderr, "harnais: %v\n", err)
			continue
		}
		history = updated
		fmt.Println(answer)
	}
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}
