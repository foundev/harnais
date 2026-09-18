package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func slashReporter() (progressFunc, *[]string) {
	var got []string
	return func(format string, args ...any) {
		got = append(got, fmt.Sprintf(format, args...))
	}, &got
}

func newSlashSession(t *testing.T) *session {
	t.Helper()
	sender, model, err := resolveBackend("anthropic", "test-key", "")
	if err != nil {
		t.Fatal(err)
	}
	return &session{
		sender:     sender,
		provider:   "anthropic",
		cfg:        config{model: model, maxIters: 1},
		configPath: filepath.Join(t.TempDir(), "config.json"),
	}
}

func lastReport(t *testing.T, got *[]string) string {
	t.Helper()
	if len(*got) == 0 {
		t.Fatal("expected a report, got none")
	}
	return (*got)[len(*got)-1]
}

func TestSlashModelShowSet(t *testing.T) {
	sess := newSlashSession(t)
	report, got := slashReporter()

	if handled, exit := handleSlash(sess, "/model", report); !handled || exit {
		t.Fatalf("handled=%v exit=%v", handled, exit)
	}
	if !strings.Contains(lastReport(t, got), "claude-sonnet-4-20250514") {
		t.Errorf("show should print current model: %q", lastReport(t, got))
	}
	if handled, _ := handleSlash(sess, "/model custom-1", report); !handled {
		t.Fatal("not handled")
	}
	if sess.cfg.model != "custom-1" {
		t.Errorf("model not set: %q", sess.cfg.model)
	}
}

func TestSlashEffortShowSetClearInvalid(t *testing.T) {
	sess := newSlashSession(t)
	report, got := slashReporter()

	handleSlash(sess, "/effort", report)
	if !strings.Contains(lastReport(t, got), "default") {
		t.Errorf("unset effort should show default: %q", lastReport(t, got))
	}
	handleSlash(sess, "/effort high", report)
	if sess.cfg.effort != "high" {
		t.Errorf("effort not set: %q", sess.cfg.effort)
	}
	handleSlash(sess, "/effort ultra", report)
	if sess.cfg.effort != "high" {
		t.Errorf("invalid effort must not change state: %q", sess.cfg.effort)
	}
	if !strings.Contains(lastReport(t, got), "low, medium, high") {
		t.Errorf("invalid effort should list levels: %q", lastReport(t, got))
	}
	handleSlash(sess, "/effort default", report)
	if sess.cfg.effort != "" {
		t.Errorf("default should clear effort: %q", sess.cfg.effort)
	}
}

func TestSlashProviderSwitch(t *testing.T) {
	t.Setenv("OPENROUTER_API_KEY", "ok")
	sess := newSlashSession(t)
	report, got := slashReporter()

	handleSlash(sess, "/provider", report)
	if !strings.Contains(lastReport(t, got), "provider: anthropic") {
		t.Errorf("show should print provider: %q", lastReport(t, got))
	}
	handleSlash(sess, "/provider openrouter", report)
	if sess.provider != "openrouter" {
		t.Fatalf("provider not switched: %q", sess.provider)
	}
	if _, ok := sess.sender.(*openAICompatClient); !ok {
		t.Errorf("sender not rebuilt: %T", sess.sender)
	}
	if sess.cfg.model != defaultOpenRouterModel {
		t.Errorf("model should reset to provider default: %q", sess.cfg.model)
	}
	handleSlash(sess, "/provider nope", report)
	if sess.provider != "openrouter" {
		t.Errorf("failed switch must keep provider: %q", sess.provider)
	}
	if !strings.Contains(lastReport(t, got), "cannot switch") {
		t.Errorf("failed switch should report: %q", lastReport(t, got))
	}
}

func TestSlashProviderCodex(t *testing.T) {
	dir := t.TempDir()
	access := craftJWT(t, map[string]any{"exp": time.Now().Add(time.Hour).Unix()})
	writeAuthJSON(t, dir, `{"auth_mode":"chatgpt","tokens":{"access_token":"`+access+`","refresh_token":"r","account_id":"a"}}`)
	t.Setenv("CODEX_HOME", dir)

	sess := newSlashSession(t)
	report, _ := slashReporter()
	handleSlash(sess, "/provider codex", report)
	if sess.provider != "codex" || sess.cfg.model != defaultCodexModel {
		t.Errorf("codex switch wrong: %+v", sess)
	}
	if _, ok := sess.sender.(*codexClient); !ok {
		t.Errorf("sender not rebuilt: %T", sess.sender)
	}
}

func TestEnsureSenderLazy(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "")
	sess := &session{provider: "anthropic", cfg: config{model: "m"}}
	if err := ensureSender(sess); err == nil {
		t.Fatal("expected missing-key error before any credential exists")
	}
	if sess.sender != nil {
		t.Fatal("failed build must not set a sender")
	}
	t.Setenv("ANTHROPIC_API_KEY", "k")
	if err := ensureSender(sess); err != nil {
		t.Fatalf("ensure: %v", err)
	}
	if sess.sender == nil {
		t.Fatal("sender not built")
	}
	first := sess.sender
	if err := ensureSender(sess); err != nil {
		t.Fatalf("second ensure: %v", err)
	}
	if sess.sender != first {
		t.Error("built sender should be reused")
	}
}

func TestReplPromptWithoutCredential(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "")
	sess := &session{provider: "anthropic", cfg: config{model: "m", maxIters: 1}}
	report, _ := slashReporter()
	in := bufio.NewReader(strings.NewReader("hi\n/exit\n"))
	if code := repl(context.Background(), sess, in, report); code != 0 {
		t.Fatalf("repl should survive a credential failure, got exit %d", code)
	}
	if sess.sender != nil || len(sess.history) != 0 {
		t.Error("failed prompt must not build a sender or record history")
	}
}

func TestLoadStoredConfig(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.json")

	got, err := loadStoredConfig(path)
	if got.Provider != "anthropic" || err != nil {
		t.Errorf("missing file should default: %+v, %v", got, err)
	}
	if err := saveStoredConfig(path, storedConfig{Provider: "openrouter", Model: "m", Effort: "high"}); err != nil {
		t.Fatal(err)
	}
	if got, err := loadStoredConfig(path); err != nil || got.Provider != "openrouter" || got.Model != "m" || got.Effort != "high" {
		t.Errorf("saved triple should round-trip: %+v, %v", got, err)
	}
	if err := os.WriteFile(path, []byte(`{broken`), 0o600); err != nil {
		t.Fatal(err)
	}
	if got, err := loadStoredConfig(path); got.Provider != "anthropic" || err == nil {
		t.Errorf("corrupt file should default with error: %+v, %v", got, err)
	}
	if err := os.WriteFile(path, []byte(`{"provider":"nope","model":"m","effort":"high"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if got, err := loadStoredConfig(path); err == nil || got.Provider != "anthropic" || got.Model != "" {
		t.Errorf("unknown provider should reset provider and model with error: %+v, %v", got, err)
	}
	if err := os.WriteFile(path, []byte(`{"provider":"openrouter","effort":"ultra"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if got, err := loadStoredConfig(path); err == nil || got.Effort != "" || got.Provider != "openrouter" {
		t.Errorf("invalid effort should clear with error, keeping provider: %+v, %v", got, err)
	}
	if _, err := loadStoredConfig(""); err != nil {
		t.Errorf("empty path disables persistence quietly: %v", err)
	}
}

func TestSlashPersistsTriple(t *testing.T) {
	t.Setenv("OPENROUTER_API_KEY", "ok")
	sess := newSlashSession(t)
	report, _ := slashReporter()

	handleSlash(sess, "/model custom-1", report)
	handleSlash(sess, "/effort high", report)
	got, err := loadStoredConfig(sess.configPath)
	if err != nil {
		t.Fatalf("load saved: %v", err)
	}
	if got.Provider != "anthropic" || got.Model != "custom-1" || got.Effort != "high" {
		t.Errorf("/model and /effort should persist the triple: %+v", got)
	}
	handleSlash(sess, "/provider openrouter", report)
	if got, err := loadStoredConfig(sess.configPath); err != nil || got.Provider != "openrouter" || got.Model != defaultOpenRouterModel {
		t.Errorf("/provider should persist provider and reset model: %+v, %v", got, err)
	}
	handleSlash(sess, "/provider nope", report)
	if got, _ := loadStoredConfig(sess.configPath); got.Provider != "openrouter" {
		t.Errorf("failed switch must not overwrite the saved default: %+v", got)
	}
}

func TestResolvePrompt(t *testing.T) {
	if _, err := resolvePrompt("x", []string{"y"}); err == nil {
		t.Error("expected error combining -p with positional args")
	}
	if got, err := resolvePrompt("x", nil); err != nil || got != "x" {
		t.Errorf("-p should win alone: %q, %v", got, err)
	}
	if got, err := resolvePrompt("", []string{"a", "b"}); err != nil || got != "a b" {
		t.Errorf("positional args should join: %q, %v", got, err)
	}
	if got, err := resolvePrompt("", nil); err != nil || got != "" {
		t.Errorf("neither should mean session: %q, %v", got, err)
	}
}

func TestSlashResetUnknownExit(t *testing.T) {
	sess := newSlashSession(t)
	sess.history = []message{textMessage("user", "hi")}
	report, got := slashReporter()

	if handled, _ := handleSlash(sess, "/reset", report); !handled {
		t.Fatal("reset not handled")
	}
	if sess.history != nil {
		t.Error("reset should clear history")
	}
	if handled, _ := handleSlash(sess, "/frobnicate", report); !handled {
		t.Fatal("unknown slash should still be handled, never sent to the model")
	}
	if !strings.Contains(lastReport(t, got), "unknown command") {
		t.Errorf("unknown slash should hint: %q", lastReport(t, got))
	}
	if handled, exit := handleSlash(sess, "just a prompt", report); handled || exit {
		t.Errorf("plain text is not a command: %v %v", handled, exit)
	}
	if handled, exit := handleSlash(sess, "/exit", report); !handled || !exit {
		t.Errorf("/exit should exit: %v %v", handled, exit)
	}
}

// fakeListerSender extends fakeSender with a model list, standing in for
// a backend that implements modelLister.
type fakeListerSender struct {
	fakeSender
	ids   []string
	calls int
}

func (f *fakeListerSender) listModels(_ context.Context) ([]string, error) {
	f.calls++
	return f.ids, nil
}

func TestSlashModelsCommand(t *testing.T) {
	sess := &session{
		sender:   &fakeListerSender{ids: []string{"zai-org/GLM-5.3", "zai-org/GLM-5.2"}},
		provider: "inceptron",
		cfg:      config{model: "zai-org/GLM-5.3", maxIters: 1},
	}
	report, got := slashReporter()
	handleSlash(sess, "/models", report)
	joined := strings.Join(*got, "\n")
	if !strings.Contains(joined, "zai-org/GLM-5.2") || !strings.Contains(joined, "zai-org/GLM-5.3") {
		t.Errorf("/models must list ids: %q", joined)
	}
	// The fetch caches: a second /models does not refetch.
	handleSlash(sess, "/models", report)
	if sess.sender.(*fakeListerSender).calls != 1 {
		t.Errorf("model list must be fetched once, got %d calls", sess.sender.(*fakeListerSender).calls)
	}

	// A backend without a list endpoint reports instead of hanging.
	sess = &session{sender: &fakeSender{}, provider: "codex", cfg: config{model: "m"}}
	report, got = slashReporter()
	handleSlash(sess, "/models", report)
	if !strings.Contains(lastReport(t, got), "unavailable") {
		t.Errorf("missing list endpoint must report: %q", lastReport(t, got))
	}

	// No credential yet: same report path, no panic.
	sess = &session{provider: "anthropic", cfg: config{model: "m"}}
	report, got = slashReporter()
	handleSlash(sess, "/models", report)
	if !strings.Contains(lastReport(t, got), "unavailable") {
		t.Errorf("missing credential must report: %q", lastReport(t, got))
	}
}

func TestProviderSwitchInvalidatesModelCache(t *testing.T) {
	t.Setenv("OPENROUTER_API_KEY", "ok")
	t.Setenv("ANTHROPIC_API_KEY", "ak")
	sess := newSlashSession(t)
	sess.modelIDs, sess.modelsLoaded = []string{"stale"}, true
	report, _ := slashReporter()
	handleSlash(sess, "/provider openrouter", report)
	if sess.modelsLoaded || sess.modelIDs != nil {
		t.Errorf("switch must drop the model cache: loaded=%v ids=%v", sess.modelsLoaded, sess.modelIDs)
	}
	// Switching also re-resolves the per-provider max-tokens default.
	sess.maxTokensFlag = 0
	handleSlash(sess, "/provider inceptron", report)
	if sess.cfg.maxTokens != 0 {
		t.Errorf("inceptron must default to an uncapped budget, got %d", sess.cfg.maxTokens)
	}
	handleSlash(sess, "/provider anthropic", report)
	if sess.cfg.maxTokens != 8192 {
		t.Errorf("anthropic must default to 8192, got %d", sess.cfg.maxTokens)
	}
	// An explicit flag rides along across switches.
	sess.maxTokensFlag = 4096
	handleSlash(sess, "/provider inceptron", report)
	if sess.cfg.maxTokens != 4096 {
		t.Errorf("explicit -max-tokens must override the provider default, got %d", sess.cfg.maxTokens)
	}
}

func TestDefaultMaxTokens(t *testing.T) {
	if got := defaultMaxTokens("inceptron"); got != 0 {
		t.Errorf("inceptron must be uncapped, got %d", got)
	}
	for _, p := range []string{"anthropic", "openrouter", "deepseek", "openai", "codex", "unknown"} {
		if got := defaultMaxTokens(p); got != 8192 {
			t.Errorf("%s must keep 8192, got %d", p, got)
		}
	}
}

func TestReplCompleter(t *testing.T) {
	sess := &session{
		sender:   &fakeListerSender{ids: []string{"zai-org/GLM-5.2", "zai-org/GLM-5.3", "moonshotai/Kimi-K2.6"}},
		provider: "inceptron",
		cfg:      config{model: "zai-org/GLM-5.3", maxIters: 1},
	}
	complete := replCompleter(context.Background(), sess)

	if got := complete("/mo"); len(got) != 2 || got[0] != "/model" || got[1] != "/models" {
		t.Errorf("command completion wrong: %v", got)
	}
	if got := complete("plain text"); got != nil {
		t.Errorf("plain prompts must not complete: %v", got)
	}
	if got := complete("/provider inc"); len(got) != 1 || got[0] != "inceptron" {
		t.Errorf("provider completion wrong: %v", got)
	}
	if got := complete("/effort h"); len(got) != 1 || got[0] != "high" {
		t.Errorf("effort completion wrong: %v", got)
	}
	if got := complete("/model zai"); len(got) != 2 {
		t.Errorf("model completion must use the backend list: %v", got)
	}
	if got := complete("/model moonshotai/"); len(got) != 1 || got[0] != "moonshotai/Kimi-K2.6" {
		t.Errorf("full-prefix model completion wrong: %v", got)
	}
	if got := complete("/model nope"); got != nil {
		t.Errorf("no matches must complete nothing: %v", got)
	}
	// The completion fetch populated the cache; further calls reuse it.
	if sess.sender.(*fakeListerSender).calls != 1 {
		t.Errorf("completions must share the /models cache, got %d calls", sess.sender.(*fakeListerSender).calls)
	}
}
