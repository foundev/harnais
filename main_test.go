package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync/atomic"
	"syscall"
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
		cfg:        config{model: model},
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
	sess := &session{provider: "anthropic", cfg: config{model: "m"}}
	report, _ := slashReporter()
	in := bufio.NewReader(strings.NewReader("hi\n/exit\n"))
	if code := repl(context.Background(), sess, in, report); code != 0 {
		t.Fatalf("repl should survive a credential failure, got exit %d", code)
	}
	if sess.sender != nil || len(sess.history) != 0 {
		t.Error("failed prompt must not build a sender or record history")
	}
}

func TestReplKeepsTheTurnWhenItStops(t *testing.T) {
	// A turn that stops early — here the reviewer refusing three actions in
	// a row — must leave its work in the session, so the next message
	// resumes the task instead of restarting it from nothing.
	sender := &fakeSender{t: t}
	sender.script = func(call int, req messageRequest) messageResponse {
		if len(req.Tools) == 0 {
			return reviewVerdictResponse("DENY: not this")
		}
		return messageResponse{
			ID: fmt.Sprintf("msg_%d", call),
			Content: []contentBlock{
				{Type: "tool_use", ID: fmt.Sprintf("toolu_%d", call), Name: "write",
					Input: json.RawMessage(`{"path":"notes.txt","content":"halfway"}`)},
			},
			StopReason: "tool_use",
		}
	}
	sess := &session{sender: sender, provider: "anthropic", cfg: config{model: "m"}}
	report, got := slashReporter()
	in := bufio.NewReader(strings.NewReader("write the notes\n/exit\n"))
	if code := repl(context.Background(), sess, in, report); code != 0 {
		t.Fatalf("repl should survive a stopped turn, got exit %d", code)
	}
	if joined := strings.Join(*got, "\n"); !strings.Contains(joined, "review: denied") {
		t.Errorf("the turn must show why it stopped: %v", *got)
	}
	if len(sess.history) < 3 {
		t.Fatalf("a stopped turn must keep its history, got %d messages", len(sess.history))
	}
	if blob := requestJSON(t, messageRequest{Messages: sess.history}); !strings.Contains(blob, "denied") {
		t.Errorf("the stopped turn must keep its tool results: %s", blob)
	}
}

func TestReplFreshInterruptContextEachTurn(t *testing.T) {
	// Ctrl+C stops a turn by cancelling the turn's context. signal.Notify
	// is one-shot — once fired, that context stays cancelled — so the repl
	// must arm a fresh one per turn. The regression this pins: with a
	// single shared context, the first ^C kills the turn and every later
	// turn fails instantly with "context canceled".
	if runtime.GOOS == "windows" {
		t.Skip("no syscall.Kill on windows")
	}
	started := make(chan struct{}, 1)
	calls := make(chan ctxErr, 8)
	sender := &interruptSender{started: started, calls: calls}
	sess := &session{sender: sender, provider: "anthropic", cfg: config{model: "m"}}
	report, got := slashReporter()
	in := bufio.NewReader(strings.NewReader("first\nsecond\n/exit\n"))
	replDone := make(chan int, 1)
	go func() { replDone <- repl(context.Background(), sess, in, report) }()

	// Turn 1: wait until the model call (and its signal handler) is live,
	// then deliver SIGINT the way the terminal does during a turn.
	<-started
	if err := syscall.Kill(os.Getpid(), syscall.SIGINT); err != nil {
		t.Fatalf("send SIGINT: %v", err)
	}
	ctx1 := <-calls
	if ctx1.errAtStart == nil {
		// The kill lands asynchronously; give it a moment.
		select {
		case <-ctx1.ctx.Done():
		case <-time.After(2 * time.Second):
			t.Error("interrupt did not cancel the first turn's context")
		}
	}

	// Turn 2 must start on a live context and reach its answer. Check the
	// error captured when the call began: stopInterrupt cancels the turn
	// context again once the turn is over, so a later read lies.
	ctx2 := <-calls
	if ctx2.errAtStart != nil {
		t.Fatalf("second turn got a dead context: %v (repl is stuck)", ctx2.errAtStart)
	}
	select {
	case code := <-replDone:
		if code != 0 {
			t.Fatalf("repl exit = %d, want 0", code)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("repl did not finish; a later turn likely failed on the cancelled context")
	}
	joined := strings.Join(*got, "\n")
	if !strings.Contains(joined, "^C") {
		t.Errorf("an interrupted turn must be reported, got: %v", *got)
	}
	if strings.Contains(joined, "context canceled") {
		t.Errorf("the interrupt must not surface as an error: %v", *got)
	}
	// Turn 1 left its user message; turn 2 added its own plus the answer.
	if blob := requestJSON(t, messageRequest{Messages: sess.history}); !strings.Contains(blob, `"first"`) || !strings.Contains(blob, `"second"`) || !strings.Contains(blob, "done") {
		t.Errorf("both turns should be in the history: %s", blob)
	}
}

// ctxErr pairs a model call's context with its error state at call time,
// because a turn context is cancelled again (deliberately) once the turn
// ends and any later read would misreport it as dead.
type ctxErr struct {
	ctx        context.Context
	errAtStart error
}

// interruptSender scripts two turns: the first blocks until its context is
// cancelled (what an interrupted API call does) and reports the error; the
// second succeeds. It hands each call's context to the test over calls.
type interruptSender struct {
	started chan struct{}
	calls   chan ctxErr
	n       atomic.Int32
}

func (s *interruptSender) createMessage(ctx context.Context, _ messageRequest) (*messageResponse, error) {
	s.calls <- ctxErr{ctx, ctx.Err()}
	if s.n.Add(1) > 1 {
		// The second turn must run to completion on its fresh context.
		return &messageResponse{
			ID:         "msg_2",
			Content:    []contentBlock{{Type: "text", Text: "done"}},
			StopReason: "end_turn",
		}, nil
	}
	s.started <- struct{}{}
	<-ctx.Done() // first call: sit until the interrupt lands
	return nil, ctx.Err()
}

func TestReplCompleter(t *testing.T) {
	sess := &session{
		provider: "inceptron",
		cfg:      config{model: "zai-org/GLM-5.3"},
	}
	// The list arrives from the background fetch; seed the cache directly
	// so this test never touches the network.
	sess.cacheModels(sess.modelsGen, sess.provider,
		[]string{"zai-org/GLM-5.2", "zai-org/GLM-5.3", "moonshotai/Kimi-K2.6"})
	complete := replCompleter(context.Background(), sess)

	if got := complete("/mo"); len(got) != 1 || got[0].Value != "/model" {
		t.Errorf("command completion wrong: %v", got)
	}
	if got := complete("plain text"); got != nil {
		t.Errorf("plain prompts must not complete: %v", got)
	}
	if got := complete("/provider inc"); len(got) != 1 || got[0].Value != "inceptron" {
		t.Errorf("provider completion wrong: %v", got)
	}
	if got := complete("/effort h"); len(got) != 1 || got[0].Value != "high" {
		t.Errorf("effort completion wrong: %v", got)
	}
	// Every model in the list is offered the moment /model is typed, each
	// labelled with the backend that serves it.
	all := complete("/model ")
	if len(all) != 3 {
		t.Fatalf("typing /model must list the whole backend list: %v", all)
	}
	if all[0].Label != "zai-org/GLM-5.2  (inceptron)" {
		t.Errorf("model rows must name the provider: %q", all[0].Label)
	}
	// Matching is a case-insensitive substring search, so a fragment in
	// the middle of an id still finds it.
	if got := complete("/model glm"); len(got) != 2 {
		t.Errorf("model completion must match loosely: %v", got)
	}
	if got := complete("/model moonshotai/"); len(got) != 1 || got[0].Value != "moonshotai/Kimi-K2.6" {
		t.Errorf("full-id model completion wrong: %v", got)
	}
	if got := complete("/model nope"); got != nil {
		t.Errorf("no matches must complete nothing: %v", got)
	}
}

// waitForModelFetch waits for the background prefetch to land.
func waitForModelFetch(t *testing.T, sess *session) {
	t.Helper()
	for i := 0; i < 400; i++ {
		sess.modelsMu.Lock()
		done := sess.modelsDone
		sess.modelsMu.Unlock()
		if done {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatal("model list was never fetched")
}

func TestStartModelFetchFillsCompletionCache(t *testing.T) {
	var asked []string
	sess := &session{
		provider: "anthropic",
		cfg:      config{model: "m"},
		modelFetch: func(_ context.Context, provider, keyFlag string) ([]string, error) {
			asked = append(asked, provider)
			// The backend's own list order is what it is; each backend
			// sorts before returning.
			return []string{"claude-opus-4-1", "claude-sonnet-4-20250514"}, nil
		},
	}
	startModelFetch(context.Background(), sess)
	waitForModelFetch(t, sess)

	ids := modelSnapshot(sess)
	if len(ids) != 2 || ids[0] != "claude-opus-4-1" || ids[1] != "claude-sonnet-4-20250514" {
		t.Fatalf("prefetch must cache the whole sorted list: %v", ids)
	}
	if len(asked) != 1 || asked[0] != "anthropic" {
		t.Errorf("prefetch must ask the session's backend: %v", asked)
	}
	// Completion then reads the cache only: no TAB, no network.
	rows := replCompleter(context.Background(), sess)("/model claude-")
	if len(rows) != 2 {
		t.Fatalf("typing must offer the cached models: %+v", rows)
	}
	if rows[0].Label != "claude-opus-4-1  (anthropic)" {
		t.Errorf("model rows must name the provider: %q", rows[0].Label)
	}
	if rows[1].Value != "claude-sonnet-4-20250514" {
		t.Errorf("row value must be the bare model id: %+v", rows[1])
	}
}

func TestStartModelFetchCachesFailure(t *testing.T) {
	calls := 0
	sess := &session{
		provider: "anthropic",
		cfg:      config{model: "m"},
		modelFetch: func(context.Context, string, string) ([]string, error) {
			calls++
			return nil, fmt.Errorf("missing API key")
		},
	}
	startModelFetch(context.Background(), sess)
	waitForModelFetch(t, sess)

	if ids := modelSnapshot(sess); ids != nil {
		t.Fatalf("a failed fetch must leave no models, got %v", ids)
	}
	// The cached failure keeps a keystroke from re-asking a dead backend.
	startModelFetch(context.Background(), sess)
	sess.modelsMu.Lock()
	inflight := sess.modelsInflight
	sess.modelsMu.Unlock()
	if inflight {
		t.Error("a cached failure must not be refetched by the prefetch")
	}
	// Typing on a /model line asks for the fetch again, but the cache is
	// done, so the backend is never hit a second time.
	replCompleter(context.Background(), sess)("/model ")
	if calls != 1 {
		t.Errorf("a cached failure must not be refetched, got %d calls", calls)
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

func TestListModelsWiring(t *testing.T) {
	sender := &fakeListerSender{ids: []string{"zai-org/GLM-5.3"}}
	ids, err := listModels(context.Background(), sender, "inceptron")
	if err != nil || len(ids) != 1 || ids[0] != "zai-org/GLM-5.3" {
		t.Fatalf("a backend with a list endpoint must report it: %v %v", ids, err)
	}
	if _, err := listModels(context.Background(), &fakeSender{}, "codex"); err == nil {
		t.Error("a backend without a list endpoint must report that")
	}
}

func TestProviderSwitchRetiresInflightFetch(t *testing.T) {
	stale := make(chan struct{})
	sess := &session{
		provider: "anthropic",
		cfg:      config{model: "m"},
		modelFetch: func(context.Context, string, string) ([]string, error) {
			<-stale
			return []string{"stale-model"}, nil
		},
	}
	startModelFetch(context.Background(), sess)
	// Switch backends while the old fetch is still in flight: its list
	// must never be served for the new provider.
	sess.setProvider("openrouter")
	close(stale)
	for i := 0; i < 400; i++ {
		sess.modelsMu.Lock()
		inflight := sess.modelsInflight
		sess.modelsMu.Unlock()
		if !inflight {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if ids := modelSnapshot(sess); ids != nil {
		t.Errorf("a fetch from the previous provider must not land: %v", ids)
	}
}

func TestSlashGoal(t *testing.T) {
	sess := newSlashSession(t)
	report, got := slashReporter()

	// Showing before anything is set.
	if handled, _ := handleSlash(sess, "/goal", report); !handled {
		t.Fatal("/goal not handled")
	}
	if !strings.Contains(lastReport(t, got), "no goal set") {
		t.Errorf("empty /goal should show the unset state: %q", lastReport(t, got))
	}

	// Setting pins it for the reviewer.
	if handled, _ := handleSlash(sess, "/goal fix the login bug", report); !handled {
		t.Fatal("/goal not handled")
	}
	if sess.goal == nil || sess.goal.Objective != "fix the login bug" || sess.cfg.goal != "fix the login bug" {
		t.Fatalf("goal not set: %+v cfg %+v", sess.goal, sess.cfg.goal)
	}

	// Changing it appends the supersede notice so the model pivots too.
	sess.history = []message{textMessage("user", "start on the login bug")}
	if handled, _ := handleSlash(sess, "/goal also update the docs", report); !handled {
		t.Fatal("/goal not handled")
	}
	if sess.goal.Objective != "also update the docs" || sess.cfg.goal != "also update the docs" {
		t.Fatalf("goal not updated: %+v", sess.goal)
	}
	last := sess.history[len(sess.history)-1]
	var text string
	if err := json.Unmarshal(last.Content, &text); err != nil {
		t.Fatalf("supersede notice is not plain text: %v", err)
	}
	for _, want := range []string{"supersedes", "also update the docs", "previous goal"} {
		if !strings.Contains(text, want) {
			t.Errorf("supersede notice missing %q:\n%s", want, text)
		}
	}

	// Same objective is a no-op.
	before := len(sess.history)
	handleSlash(sess, "/goal also update the docs", report)
	if len(sess.history) != before {
		t.Error("setting the same goal must not append another notice")
	}

	// Clear drops it; /reset also clears.
	if handled, _ := handleSlash(sess, "/goal clear", report); !handled {
		t.Fatal("/goal not handled")
	}
	if sess.goal != nil || sess.cfg.goal != "" {
		t.Fatalf("goal not cleared: %+v", sess.goal)
	}
	handleSlash(sess, "/goal again", report)
	handleSlash(sess, "/reset", report)
	if sess.goal != nil || sess.cfg.goal != "" {
		t.Fatalf("/reset must clear the goal: %+v", sess.goal)
	}
	if len(sess.history) != 0 {
		t.Fatal("/reset must clear history")
	}
}
