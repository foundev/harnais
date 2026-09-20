package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

var version = "0.1.0"

func main() {
	os.Exit(run(os.Args[1:]))
}

func run(args []string) int {
	fs := flag.NewFlagSet("harnais", flag.ContinueOnError)
	providerFlag := fs.String("provider", "", "Backend for this run: anthropic, openrouter, deepseek, openai, inceptron or codex (overrides the saved default without changing it).")
	model := fs.String("model", envOr("ANTHROPIC_MODEL", ""), "Model ID (default depends on the saved provider).")
	printMode := fs.String("p", "", "Run one non-interactive prompt and exit, e.g. -p \"fix the failing test\" (so flags can follow the prompt).")
	apiKey := fs.String("key", "", "API key (not used by the codex backend, which reads the Codex login).")
	effort := fs.String("effort", "", "Reasoning effort: low, medium or high (sent to all backends).")
	resume := fs.Bool("resume", false, "Continue the most recent saved session (interactive mode only).")
	noSandbox := fs.Bool("no-sandbox", false, "Run bash without the OS sandbox (macOS Seatbelt).")
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
  ANTHROPIC_API_KEY    API key for the anthropic backend.
  ANTHROPIC_MODEL      Default -model value (any backend).
  ANTHROPIC_BASE_URL   Anthropic API root (default %s).
  OPENROUTER_API_KEY   API key for the openrouter backend.
  OPENROUTER_BASE_URL  OpenRouter API root (default %s).
  OPENROUTER_REFERER   Optional HTTP-Referer header for OpenRouter rankings.
  DEEPSEEK_API_KEY     API key for the deepseek backend.
  DEEPSEEK_BASE_URL    DeepSeek API root (default %s).
  OPENAI_API_KEY       API key for the openai backend.
  OPENAI_BASE_URL      OpenAI API root (default %s).
  INCEPTRON_API_KEY    API key for the inceptron backend.
  INCEPTRON_BASE_URL   Inceptron API root (default %s).
  CODEX_HOME           Directory holding Codex auth.json (default ~/.codex).
  CODEX_BASE_URL       Codex backend root (default %s).

In a session, type /help for slash commands (/provider, /model, /effort, /reset, /exit).

Interactive sessions are saved as they run; start the most recent one with -resume.
`, defaultBaseURL, defaultOpenRouterBaseURL, defaultDeepSeekBaseURL, defaultOpenAIBaseURL, defaultInceptronBaseURL, defaultCodexBaseURL)
	}
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *showVersion {
		fmt.Printf("harnais %s\n", version)
		return 0
	}
	if *effort != "" && !validEffort(*effort) {
		fmt.Fprintln(os.Stderr, "harnais: -effort must be low, medium or high.")
		return 2
	}

	// The backend defaults to the saved /provider choice (anthropic on
	// first run); -provider overrides it for one run without saving.
	cfgPath, err := configFilePath()
	if err != nil {
		fmt.Fprintf(os.Stderr, "harnais: %s\n", paint(ansiYellow, fmt.Sprintf("%v (provider choice will not persist)", err)))
		cfgPath = ""
	}
	stored, werr := loadStoredConfig(cfgPath)
	if werr != nil {
		fmt.Fprintf(os.Stderr, "harnais: %s\n", paint(ansiYellow, fmt.Sprintf("%v, using defaults", werr)))
	}
	provider := firstNonEmpty(*providerFlag, stored.Provider)
	defModel, err := defaultModelFor(provider)
	if err != nil {
		printErr("%v", err)
		return 2
	}
	cfg := config{
		// Flags win for this run; the saved triple is the default.
		model:       firstNonEmpty(*model, stored.Model, defModel),
		extraSystem: *extraSystem,
		effort:      firstNonEmpty(*effort, stored.Effort),
		sandbox:     !*noSandbox,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	in := bufio.NewReader(os.Stdin)
	report := func(format string, args ...any) {
		fmt.Fprintf(os.Stderr, format+"\n", args...)
	}

	prompt, err := resolvePrompt(*printMode, fs.Args())
	if err != nil {
		printErr("%v", err)
		return 2
	}
	if prompt != "" {
		// One-shot mode cannot switch later, so its credential is due now.
		sender, err := buildSender(provider, *apiKey)
		if err != nil {
			printErr("%v", err)
			return 2
		}
		answer, _, err := runPrompt(ctx, sender, cfg, nil, prompt, report)
		if err != nil {
			printErr("%v", err)
			return 1
		}
		fmt.Println(paintAnswer(answer))
		return 0
	}
	// Sessions start without touching any credential: the backend is built
	// lazily on the first prompt (or eagerly by /provider), so no key is
	// needed just to open the REPL.
	sess := &session{provider: provider, cfg: cfg, keyFlag: *apiKey, configPath: cfgPath}
	if *resume {
		st, path, err := latestSession("")
		if err != nil {
			printErr("%v", err)
			return 2
		}
		applySessionState(sess, st, path)
		report("resumed %s — %s (%d messages)", path, sess.provider+"/"+sess.cfg.model, len(sess.history))
		replayTranscript(sess.history, report)
	}
	return repl(ctx, sess, in, report)
}

// resolvePrompt picks the one-shot prompt: -p or positional args, never
// both. Empty means start a session.
func resolvePrompt(pFlag string, args []string) (string, error) {
	if pFlag != "" && len(args) > 0 {
		return "", fmt.Errorf("-p cannot be combined with a positional prompt")
	}
	if pFlag != "" {
		return pFlag, nil
	}
	if len(args) > 0 {
		return strings.Join(args, " "), nil
	}
	return "", nil
}

// storedConfig is the persisted session defaults ($XDG_CONFIG_HOME/harnais).
type storedConfig struct {
	Provider string `json:"provider"`
	Model    string `json:"model,omitempty"`
	Effort   string `json:"effort,omitempty"`
}

// configFilePath locates the persisted defaults file.
func configFilePath() (string, error) {
	dir := os.Getenv("XDG_CONFIG_HOME")
	if dir == "" {
		home, err := os.UserHomeDir()
		if err != nil || home == "" {
			return "", fmt.Errorf("cannot locate a config directory — set XDG_CONFIG_HOME")
		}
		dir = filepath.Join(home, ".config")
	}
	return filepath.Join(dir, "harnais", "config.json"), nil
}

// loadStoredConfig returns the saved session defaults. Anything unusable
// falls back field-by-field (provider anthropic, empty model/effort) with an
// error so the caller can warn once. An empty path disables persistence.
func loadStoredConfig(path string) (storedConfig, error) {
	sc := storedConfig{Provider: "anthropic"}
	raw, err := os.ReadFile(path)
	if err != nil {
		if path == "" || os.IsNotExist(err) {
			return sc, nil
		}
		return sc, fmt.Errorf("read config: %v", err)
	}
	if err := json.Unmarshal(raw, &sc); err != nil {
		return storedConfig{Provider: "anthropic"}, fmt.Errorf("parse config %s: %v", path, err)
	}
	if sc.Provider == "" {
		sc.Provider = "anthropic"
	} else if _, err := defaultModelFor(sc.Provider); err != nil {
		bad := sc.Provider
		sc.Provider, sc.Model = "anthropic", ""
		return sc, fmt.Errorf("config %s: unknown provider %q", path, bad)
	}
	if sc.Effort != "" && !validEffort(sc.Effort) {
		bad := sc.Effort
		sc.Effort = ""
		return sc, fmt.Errorf("config %s: invalid effort %q", path, bad)
	}
	return sc, nil
}

// saveStoredConfig records the session defaults for the next launch,
// creating the config directory as needed.
func saveStoredConfig(path string, sc storedConfig) error {
	if path == "" {
		return fmt.Errorf("no config path available")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create config dir: %w", err)
	}
	raw, err := json.MarshalIndent(sc, "", "  ")
	if err != nil {
		return fmt.Errorf("encode config: %w", err)
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".config-*.tmp")
	if err != nil {
		return fmt.Errorf("write config: %w", err)
	}
	tmpName := tmp.Name()
	if _, err := tmp.Write(append(raw, '\n')); err != nil {
		tmp.Close()
		os.Remove(tmpName)
		return fmt.Errorf("write config: %w", err)
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmpName)
		return fmt.Errorf("write config: %w", err)
	}
	if err := os.Rename(tmpName, path); err != nil {
		os.Remove(tmpName)
		return fmt.Errorf("write config: %w", err)
	}
	return nil
}

// session is the mutable interactive state: backend, config, and history.
// sender stays nil until the first prompt or /provider builds it.
type session struct {
	sender   messageSender
	provider string
	cfg      config
	history  []message
	// savePath is where the session is written for -resume; "" disables
	// saving. saved records whether the first successful save was
	// reported, so the path is announced once, not after every turn.
	savePath string
	saved    bool
	// keyFlag is the startup -key, used for the initial backend only;
	// switching providers falls back to environment credentials.
	keyFlag string
	// configPath persists /provider switches; "" disables persistence.
	configPath string
	// The model list feeds the editor's live completion.
	// modelsMu guards the cache shared with the background prefetch;
	// modelsGen is bumped by /provider so a fetch for the old backend
	// cannot land after the switch.
	modelsMu       sync.Mutex
	modelsGen      uint64
	modelIDs       []string
	modelsDone     bool
	modelsInflight bool
	// modelFetch lists a backend's models (nil means fetchModelIDs);
	// tests replace it to stay off the network.
	modelFetch func(ctx context.Context, provider, keyFlag string) ([]string, error)
}

// ensureSender builds the session backend on first use.
func ensureSender(sess *session) error {
	if sess.sender != nil {
		return nil
	}
	sender, err := buildSender(sess.provider, sess.keyFlag)
	if err != nil {
		return err
	}
	sess.sender = sender
	return nil
}

func validEffort(level string) bool {
	switch level {
	case "low", "medium", "high":
		return true
	default:
		return false
	}
}

func describeEffort(sess *session) string {
	if sess.cfg.effort == "" {
		return "default (backend decides)"
	}
	return sess.cfg.effort
}

// persistSession saves provider, model, and effort as the next launch's
// defaults, reporting the outcome to the user.
func persistSession(sess *session, report progressFunc, what string) {
	sc := storedConfig{Provider: sess.provider, Model: sess.cfg.model, Effort: sess.cfg.effort}
	if err := saveStoredConfig(sess.configPath, sc); err != nil {
		report("%s", paint(ansiYellow, fmt.Sprintf("%s (default not saved: %v)", what, err)))
		return
	}
	report("%s", paint(ansiGreen, fmt.Sprintf("%s (saved as default)", what)))
}

// handleSlash runs one REPL slash command. It returns handled=true for any
// slash line (so it never reaches the model) and exit=true for /exit|/quit.
func handleSlash(sess *session, line string, report progressFunc) (handled, exit bool) {
	fields := strings.Fields(line)
	if len(fields) == 0 || !strings.HasPrefix(fields[0], "/") {
		return false, false
	}
	switch fields[0] {
	case "/exit", "/quit":
		return true, true
	case "/reset":
		sess.history = nil
		report("conversation reset")
	case "/resume":
		// Load the most recent other session into this one. The live
		// session is excluded so /resume never just reloads itself, and
		// the current conversation is saved first so it stays resumable
		// after you switch away.
		if sess.history != nil && sess.savePath != "" {
			if err := saveSession(sess); err != nil {
				report("%s", paint(ansiYellow, fmt.Sprintf("could not save the current session before switching: %v", err)))
			}
		}
		st, path, err := latestSession(sess.savePath)
		if err != nil {
			report("%s", paint(ansiYellow, fmt.Sprintf("cannot resume: %v", err)))
			break
		}
		applySessionState(sess, st, path)
		report("%s", paint(ansiGreen, fmt.Sprintf("resumed %s — %s (%d messages)", filepath.Base(path), sess.provider+"/"+sess.cfg.model, len(sess.history))))
		replayTranscript(sess.history, report)
	case "/help":
		report(`Prompts go straight to the model. Commands:
  /provider [name]   show or switch backend (anthropic, openrouter, deepseek, openai, inceptron, codex); switching saves provider and model as the default for next launch and keeps history
  /model [id]        show or set the model id (saved as default)
  /effort [level]    show or set reasoning effort: low, medium, high, default (saved as default; sent to all backends)
  /reset             clear conversation history
  /resume            load the most recent saved session (excluding this one) and replay its transcript
  /exit              leave

Sessions auto-save as they run; restart the latest with: harnais -resume`)
	case "/provider":
		if len(fields) < 2 {
			report("provider: %s (model %s, effort %s)", sess.provider, sess.cfg.model, describeEffort(sess))
			break
		}
		sender, model, err := resolveBackend(fields[1], "", "")
		if err != nil {
			report("%s", paint(ansiRed, fmt.Sprintf("cannot switch: %v", err)))
			break
		}
		sess.sender = sender
		// The new backend has its own model list; drop the cache. The
		// editor warms it again as soon as the next line is typed.
		sess.setProvider(fields[1])
		sess.cfg.model = model
		persistSession(sess, report, fmt.Sprintf("provider: %s, model reset to %s (history kept)", sess.provider, model))
	case "/model":
		if len(fields) < 2 {
			report("model: %s", sess.cfg.model)
			break
		}
		sess.cfg.model = fields[1]
		persistSession(sess, report, fmt.Sprintf("model: %s", sess.cfg.model))
	case "/effort":
		if len(fields) < 2 {
			report("effort: %s (sent to all backends)", describeEffort(sess))
			break
		}
		if fields[1] == "default" {
			sess.cfg.effort = ""
			persistSession(sess, report, "effort: default (backend decides)")
			break
		}
		if !validEffort(fields[1]) {
			report("%s", paint(ansiRed, "effort must be low, medium, high, or default"))
			break
		}
		sess.cfg.effort = fields[1]
		persistSession(sess, report, fmt.Sprintf("effort: %s (sent to all backends)", sess.cfg.effort))
	default:
		report("%s", paint(ansiRed, fmt.Sprintf("unknown command %q — type /help", fields[0])))
	}
	return true, false
}

// defaultModelFor returns the default model of a known provider.
func defaultModelFor(provider string) (string, error) {
	switch provider {
	case "anthropic":
		return "claude-sonnet-4-20250514", nil
	case "openrouter":
		return defaultOpenRouterModel, nil
	case "deepseek":
		return defaultDeepSeekModel, nil
	case "openai":
		return defaultOpenAIModel, nil
	case "inceptron":
		return defaultInceptronModel, nil
	case "codex":
		return defaultCodexModel, nil
	default:
		return "", fmt.Errorf("unknown provider %q — use anthropic, openrouter, deepseek, openai, inceptron or codex", provider)
	}
}

// modelListTimeout bounds a background model-list fetch so a dead backend
// cannot leave the cache permanently cold.
const modelListTimeout = 20 * time.Second

// setProvider records a /provider switch and retires the cached model
// list; the next prefetch repopulates it for the new backend.
func (sess *session) setProvider(provider string) {
	sess.modelsMu.Lock()
	defer sess.modelsMu.Unlock()
	sess.provider = provider
	sess.keyFlag = ""
	sess.modelsGen++
	sess.modelIDs, sess.modelsDone = nil, false
}

// modelSnapshot returns the cached model list without touching the
// network: the editor asks for it on every keystroke and must never wait.
func modelSnapshot(sess *session) []string {
	sess.modelsMu.Lock()
	defer sess.modelsMu.Unlock()
	if !sess.modelsDone {
		return nil
	}
	return sess.modelIDs
}

// cacheModels stores a fetched list, unless /provider moved to another
// backend while the fetch was in flight. A failed or empty fetch caches
// as an empty list so the next keystroke does not ask again.
func (sess *session) cacheModels(gen uint64, provider string, ids []string) {
	sess.modelsMu.Lock()
	defer sess.modelsMu.Unlock()
	sess.cacheModelsLocked(gen, provider, ids)
}

func (sess *session) cacheModelsLocked(gen uint64, provider string, ids []string) {
	if gen != sess.modelsGen || provider != sess.provider {
		return
	}
	sess.modelIDs, sess.modelsDone = ids, true
}

// startModelFetch warms the model cache in the background, once per
// backend. Failures are cached too: the editor asks on every keystroke, so
// a backend without a list endpoint (or without a credential) must not
// turn typing into a burst of failing requests.
func startModelFetch(ctx context.Context, sess *session) {
	sess.modelsMu.Lock()
	if sess.modelsDone || sess.modelsInflight {
		sess.modelsMu.Unlock()
		return
	}
	sess.modelsInflight = true
	provider, keyFlag, gen := sess.provider, sess.keyFlag, sess.modelsGen
	fetch := sess.fetcher()
	sess.modelsMu.Unlock()

	go func() {
		ids, err := fetch(ctx, provider, keyFlag)
		if err != nil {
			ids = nil
		}
		// Clear the in-flight flag and commit in one step, so the next
		// keystroke never sees an idle cache that is about to fill.
		sess.modelsMu.Lock()
		defer sess.modelsMu.Unlock()
		sess.modelsInflight = false
		sess.cacheModelsLocked(gen, provider, ids)
	}()
}

// fetcher is how the session lists a backend's models. Tests replace
// modelFetch so no test needs a live backend.
func (sess *session) fetcher() func(ctx context.Context, provider, keyFlag string) ([]string, error) {
	if sess.modelFetch != nil {
		return sess.modelFetch
	}
	return fetchModelIDs
}

// fetchModelIDs builds the backend from the given credential and asks it
// for its model list. It touches no session state, so a prefetch can run
// while the REPL keeps editing.
func fetchModelIDs(ctx context.Context, provider, keyFlag string) ([]string, error) {
	sender, err := buildSender(provider, keyFlag)
	if err != nil {
		return nil, err
	}
	fctx, cancel := context.WithTimeout(ctx, modelListTimeout)
	defer cancel()
	return listModels(fctx, sender, provider)
}

// listModels asks an already-built backend for its model list.
func listModels(ctx context.Context, sender messageSender, provider string) ([]string, error) {
	lister, ok := sender.(modelLister)
	if !ok {
		return nil, fmt.Errorf("the %s backend has no model list endpoint", provider)
	}
	return lister.listModels(ctx)
}

// slashCommands, providerNames, and effortLevels feed live completion.
var (
	slashCommands = []string{"/effort", "/exit", "/help", "/model", "/provider", "/quit", "/reset", "/resume"}
	providerNames = []string{"anthropic", "codex", "deepseek", "inceptron", "openai", "openrouter"}
	effortLevels  = []string{"default", "high", "low", "medium"}
)

// replCompleter serves completion in the interactive editor: slash
// commands on the first word, then per-command values — provider names,
// effort levels, and the backend's whole model list after /model. The
// editor calls this on every keystroke, so it only ever reads the cache;
// the fetch happens in the background (see startModelFetch).
func replCompleter(ctx context.Context, sess *session) completer {
	return func(line string) []completion {
		if !strings.HasPrefix(line, "/") {
			return nil
		}
		if !strings.Contains(line, " ") {
			return prefixCompletions(slashCommands, line)
		}
		cmd, _, _ := strings.Cut(line, " ")
		word := lastWord(line)
		switch cmd {
		case "/provider":
			return prefixCompletions(providerNames, word)
		case "/model":
			startModelFetch(ctx, sess)
			return modelCompletions(modelSnapshot(sess), word, sess.provider)
		case "/effort":
			return prefixCompletions(effortLevels, word)
		}
		return nil
	}
}

// prefixCompletions offers the candidates the typed word starts.
func prefixCompletions(cands []string, word string) []completion {
	var out []completion
	for _, c := range cands {
		if strings.HasPrefix(c, word) {
			out = append(out, completion{Value: c, Label: c})
		}
	}
	return out
}

// modelCompletions matches model ids anywhere in the id and ignoring case
// — typing "glm" must find "zai-org/GLM-5.3" — and shows every id with the
// backend serving it.
func modelCompletions(ids []string, word, provider string) []completion {
	needle := strings.ToLower(word)
	var out []completion
	for _, id := range ids {
		if !strings.Contains(strings.ToLower(id), needle) {
			continue
		}
		out = append(out, completion{Value: id, Label: modelLabel(id, provider)})
	}
	return out
}

// modelLabel is a model id plus the backend it comes from, dimmed so the
// id itself stays the thing being read.
func modelLabel(id, provider string) string {
	return id + "  " + paint(ansiDim, "("+provider+")")
}

// buildSender constructs the backend client, requiring its credential.
// Model resolution lives in defaultModelFor so sessions can start and
// switch display state before any key exists.
func buildSender(provider, keyFlag string) (messageSender, error) {
	switch provider {
	case "anthropic":
		key := firstNonEmpty(keyFlag, os.Getenv("ANTHROPIC_API_KEY"))
		if key == "" {
			return nil, fmt.Errorf("missing API key — set ANTHROPIC_API_KEY or pass -key")
		}
		return newAnthropicClient(key, envOr("ANTHROPIC_BASE_URL", defaultBaseURL)), nil
	case "openrouter":
		key := firstNonEmpty(keyFlag, os.Getenv("OPENROUTER_API_KEY"))
		if key == "" {
			return nil, fmt.Errorf("missing API key — set OPENROUTER_API_KEY or pass -key")
		}
		return newOpenRouterClient(key, envOr("OPENROUTER_BASE_URL", defaultOpenRouterBaseURL)), nil
	case "deepseek":
		key := firstNonEmpty(keyFlag, os.Getenv("DEEPSEEK_API_KEY"))
		if key == "" {
			return nil, fmt.Errorf("missing API key — set DEEPSEEK_API_KEY or pass -key")
		}
		return newDeepSeekClient(key, envOr("DEEPSEEK_BASE_URL", defaultDeepSeekBaseURL)), nil
	case "openai":
		key := firstNonEmpty(keyFlag, os.Getenv("OPENAI_API_KEY"))
		if key == "" {
			return nil, fmt.Errorf("missing API key — set OPENAI_API_KEY or pass -key")
		}
		return newOpenAIClient(key, envOr("OPENAI_BASE_URL", defaultOpenAIBaseURL)), nil
	case "inceptron":
		key := firstNonEmpty(keyFlag, os.Getenv("INCEPTRON_API_KEY"))
		if key == "" {
			return nil, fmt.Errorf("missing API key — set INCEPTRON_API_KEY or pass -key")
		}
		return newInceptronClient(key, envOr("INCEPTRON_BASE_URL", defaultInceptronBaseURL)), nil
	case "codex":
		authPath, err := codexAuthPath()
		if err != nil {
			return nil, err
		}
		if _, err := loadCodexTokens(authPath); err != nil {
			return nil, err
		}
		return newCodexClient(authPath, envOr("CODEX_BASE_URL", defaultCodexBaseURL)), nil
	default:
		_, err := defaultModelFor(provider)
		return nil, err
	}
}

// resolveBackend picks the model backend, applying per-provider key, model,
// and base-URL defaults. It returns the sender and the resolved model ID.
func resolveBackend(provider, keyFlag, modelFlag string) (messageSender, string, error) {
	def, err := defaultModelFor(provider)
	if err != nil {
		return nil, "", err
	}
	sender, err := buildSender(provider, keyFlag)
	if err != nil {
		return nil, "", err
	}
	return sender, firstNonEmpty(modelFlag, def), nil
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}

func firstNonZero(vals ...int) int {
	for _, v := range vals {
		if v != 0 {
			return v
		}
	}
	return 0
}

func repl(ctx context.Context, sess *session, in *bufio.Reader, report progressFunc) int {
	// Line editing with live completion needs both ends on the terminal:
	// keystrokes from stdin, echo to stderr. Pipes and tests keep the
	// plain bufio path.
	interactive := isTerminal(os.Stdin) && isTerminal(os.Stderr)
	report("harnais %s — %s (type /help for commands)", version, paint(ansiCyan, sess.provider+"/"+sess.cfg.model))
	var completeFn completer
	if interactive {
		completeFn = replCompleter(ctx, sess)
		// Warm the model list now so the first /model keystroke already
		// has something to suggest.
		startModelFetch(ctx, sess)
	}
	for {
		var line string
		if interactive {
			edited, exit := readLineEdited(paint(ansiBoldCyan, sess.provider+"> "), completeFn)
			if exit {
				return 0
			}
			line = edited
		} else {
			fmt.Fprintf(os.Stderr, "%s", paint(ansiBoldCyan, sess.provider+"> "))
			raw, err := in.ReadString('\n')
			if err != nil {
				fmt.Fprintln(os.Stderr)
				return 0
			}
			line = raw
		}
		prompt := strings.TrimSpace(line)
		if prompt == "" {
			continue
		}
		if handled, exit := handleSlash(sess, prompt, report); handled {
			if exit {
				return 0
			}
			continue
		}
		if err := ensureSender(sess); err != nil {
			printErr("%v", err)
			continue
		}
		answer, updated, err := runPrompt(ctx, sess.sender, sess.cfg, sess.history, prompt, report)
		// Keep the turn's work even when it stopped early: the tool results
		// in `updated` are the only record of what it did, so the next
		// prompt resumes the task instead of starting it over.
		if updated != nil {
			sess.history = updated
			// Persist after every turn — including turns that stopped
			// early — so -resume always picks up exactly what is on
			// screen. The first save also assigns the fresh session its
			// timestamped file and announces where it lives; a resumed
			// session already has both (see applySessionState).
			if sess.savePath == "" {
				if p, err := newSessionPath(); err == nil {
					sess.savePath = p
				}
			}
			if err := saveSession(sess); err != nil {
				report("%s", paint(ansiYellow, fmt.Sprintf("session not saved: %v", err)))
			} else if !sess.saved {
				sess.saved = true
				report("%s", paint(ansiDim, fmt.Sprintf("session saving to %s (continue later with -resume)", sess.savePath)))
			}
		}
		if err != nil {
			printErr("%v", err)
			continue
		}
		fmt.Println(paintAnswer(answer))
	}
}

// printErr reports a failure in red (TTY only, see color.go).
func printErr(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "harnais: %s\n", paint(ansiRed, fmt.Sprintf(format, args...)))
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}
