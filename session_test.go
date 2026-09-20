package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestSessionRoundTrip checks that a saved session carries the whole
// conversation — user prompts, assistant tool calls, tool results — back
// into a new session byte-for-byte at the wire level.
func TestSessionRoundTrip(t *testing.T) {
	cfgDir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", cfgDir)
	path := filepath.Join(cfgDir, "harnais", "sessions", "20240101-000000.json")

	sess := &session{
		provider: "openrouter",
		cfg:      config{model: "openai/gpt-5-mini", effort: "low", goal: "fix the failing test"},
		savePath: path,
		goal:     &threadGoal{Objective: "fix the failing test"},
		history: []message{
			textMessage("user", "fix the failing test"),
			blocksMessage("assistant", []contentBlock{
				{Type: "text", Text: "looking…"},
				{Type: "tool_use", ID: "toolu_1", Name: "bash", Input: json.RawMessage(`{"command":"go test ./..."}`)},
			}),
			blocksMessage("user", []contentBlock{
				{Type: "tool_result", ToolUseID: "toolu_1", Content: "ok"},
			}),
		},
	}
	if err := saveSession(sess); err != nil {
		t.Fatalf("save: %v", err)
	}

	loaded := &session{provider: "anthropic", cfg: config{model: "m"}}
	st, gotPath, err := latestSession("")
	if err != nil {
		t.Fatalf("latestSession: %v", err)
	}
	if gotPath != path {
		t.Errorf("latest session path = %s, want %s", gotPath, path)
	}
	applySessionState(loaded, st, gotPath)
	if loaded.provider != "openrouter" || loaded.cfg.model != "openai/gpt-5-mini" || loaded.cfg.effort != "low" {
		t.Errorf("backend not restored: %s %+v", loaded.provider, loaded.cfg)
	}
	if loaded.goal == nil || loaded.goal.Objective != "fix the failing test" || loaded.cfg.goal != "fix the failing test" {
		t.Errorf("goal not restored: %+v cfg %+v", loaded.goal, loaded.cfg.goal)
	}
	if got, _ := json.Marshal(sess.history); string(got) == "null" {
		t.Fatal("history must not be empty after save")
	}
	if raw, _ := json.Marshal(loaded.history); string(raw) == "null" || len(loaded.history) != len(sess.history) {
		t.Errorf("history not restored: %d messages", len(loaded.history))
	}
	// The restored history must be wire-identical to the original.
	want, _ := json.Marshal(sess.history)
	got, _ := json.Marshal(loaded.history)
	if string(want) != string(got) {
		t.Errorf("history changed across save/load:\nwant %s\ngot  %s", want, got)
	}
}

// TestLatestSessionSkipsUnusable checks that -resume lands on the newest
// session with a real conversation, skipping empty and unreadable files.
func TestLatestSessionSkipsUnusable(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir)
	sessDir := filepath.Join(dir, "harnais", "sessions")
	os.MkdirAll(sessDir, 0o755)

	write := func(name, body string) {
		if err := os.WriteFile(filepath.Join(sessDir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	// Older: a session that never sent a prompt. Broken: unparsable.
	write("20240101-000000.json", `{"version":1,"provider":"anthropic","history":[]}`)
	write("20240102-000000.json", `{not json`)
	// Newest: one real message.
	write("20240103-000000.json",
		`{"version":1,"provider":"anthropic","history":[{"role":"user","content":"hi"}]}`)

	st, path, err := latestSession("")
	if err != nil {
		t.Fatalf("latestSession: %v", err)
	}
	if filepath.Base(path) != "20240103-000000.json" {
		t.Errorf("picked %s, want the newest usable session", path)
	}
	if len(st.History) != 1 || st.Provider != "anthropic" {
		t.Errorf("state wrong: %+v", st)
	}

	// Nothing usable at all is a clean, explained error — not a crash.
	write("20240103-000000.json", `{"version":999,"history":[]}`)
	if _, _, err := latestSession(""); err == nil {
		t.Error("expected an error when no usable session exists")
	}
}

// TestLatestSessionExclude checks that /resume's exclusion works at the
// file level: the excluded path is skipped even when it is the newest.
func TestLatestSessionExclude(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir)
	sessDir := filepath.Join(dir, "harnais", "sessions")
	os.MkdirAll(sessDir, 0o755)
	hist := `{"version":1,"provider":"anthropic","history":[{"role":"user","content":"hi"}]}`
	newest := filepath.Join(sessDir, "20240102-000000.json")
	older := filepath.Join(sessDir, "20240101-000000.json")
	// Older timestamp name, but written last so mtime ordering would be
	// wrong — only the name sort must matter.
	os.WriteFile(older, []byte(hist), 0o644)
	os.WriteFile(newest, []byte(hist), 0o644)

	if _, path, err := latestSession(""); err != nil || path != newest {
		t.Fatalf("without exclusion: path=%s err=%v", path, err)
	}
	st, path, err := latestSession(newest)
	if err != nil {
		t.Fatalf("exclude: %v", err)
	}
	if path != older {
		t.Errorf("excluded session must be skipped, got %s", path)
	}
	if len(st.History) != 1 {
		t.Errorf("state wrong: %+v", st)
	}
	// Excluding the only usable session is a clean error.
	if _, _, err := latestSession(newest, older); err == nil {
		t.Error("expected an error when every usable session is excluded")
	}
}

// TestSameFile checks the two cases /resume relies on: an existing file is
// matched by identity, a not-yet-written one by cleaned path.
func TestSameFile(t *testing.T) {
	dir := t.TempDir()
	a := filepath.Join(dir, "a.json")
	os.WriteFile(a, []byte("x"), 0o644)
	if !sameFile(a, a) {
		t.Error("a file must match itself")
	}
	if !sameFile(a, filepath.Join(dir, "sub", "..", "a.json")) {
		t.Error("cleaned equal paths must match")
	}
	if sameFile(a, "") || sameFile("", a) {
		t.Error("empty paths match nothing")
	}
	if sameFile(a, filepath.Join(dir, "missing.json")) {
		t.Error("existing vs missing must not match")
	}
}

// TestSlashResume checks the command end to end: it saves the live
// conversation first, loads the most recent other session, and keeps the
// sender so the next prompt goes to the resumed backend.
func TestSlashResume(t *testing.T) {
	cfgDir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", cfgDir)
	sessDir := filepath.Join(cfgDir, "harnais", "sessions")
	os.MkdirAll(sessDir, 0o755)
	os.WriteFile(filepath.Join(sessDir, "20240101-000000.json"), []byte(
		`{"version":1,"provider":"openrouter","model":"openai/gpt-5-mini","effort":"low",
		  "history":[{"role":"user","content":"earlier work"},{"role":"assistant","content":"done"}]}`), 0o644)

	sender := &fakeSender{t: t}
	sender.script = func(call int, req messageRequest) messageResponse {
		return messageResponse{
			ID:         fmt.Sprintf("msg_%d", call),
			Content:    []contentBlock{{Type: "text", Text: "ok."}},
			StopReason: "end_turn",
		}
	}
	sess := &session{sender: sender, provider: "anthropic", cfg: config{model: "m"}}
	report, got := slashReporter()

	// Turn one fills the live session; /resume then switches away from it.
	in := bufio.NewReader(strings.NewReader("hello\n/resume\n/exit\n"))
	if code := repl(context.Background(), sess, in, report); code != 0 {
		t.Fatalf("repl exited %d", code)
	}
	if sess.provider != "openrouter" || sess.cfg.model != "openai/gpt-5-mini" || sess.cfg.effort != "low" {
		t.Errorf("resumed backend wrong: %s %+v", sess.provider, sess.cfg)
	}
	if len(sess.history) != 2 {
		t.Fatalf("resumed history wrong: %d messages", len(sess.history))
	}
	if blob := requestJSON(t, messageRequest{Messages: sess.history}); !strings.Contains(blob, "earlier work") {
		t.Errorf("resumed history must be the saved one: %s", blob)
	}
	joined := strings.Join(*got, "\n")
	if !strings.Contains(joined, "resumed 20240101-000000.json") {
		t.Errorf("must say which session resumed: %v", *got)
	}
	// The live session was saved before switching, so it stays resumable:
	// its file holds the "hello" turn, unlike the file now in savePath.
	var liveFiles []string
	entries, _ := os.ReadDir(sessDir)
	for _, e := range entries {
		liveFiles = append(liveFiles, filepath.Join(sessDir, e.Name()))
	}
	found := false
	for _, f := range liveFiles {
		raw, err := os.ReadFile(f)
		if err == nil && strings.Contains(string(raw), "hello") {
			found = true
		}
	}
	if !found {
		t.Errorf("the pre-switch conversation must be saved in some session file: %v", liveFiles)
	}
}

// TestSlashResumeWithoutSavedSessions checks the empty case reads as a
// normal report, not an error crash.
func TestSlashResumeWithoutSavedSessions(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	sess := &session{provider: "anthropic", cfg: config{model: "m"}}
	report, got := slashReporter()
	handled, exit := handleSlash(sess, "/resume", report)
	if !handled || exit {
		t.Errorf("/resume must be handled and not exit: %v %v", handled, exit)
	}
	if last := lastReport(t, got); !strings.Contains(last, "cannot resume") {
		t.Errorf("expected a cannot-resume report, got %q", last)
	}
}

// TestNewSessionPathSuffixesCollisions checks that a second session born
// in the same second gets its own file instead of clobbering the first.
func TestNewSessionPathSuffixesCollisions(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir)
	p1, err := newSessionPath()
	if err != nil {
		t.Fatalf("first path: %v", err)
	}
	// Pre-create both the plain name and the first suffix, so the "next"
	// path must skip past them.
	os.MkdirAll(filepath.Dir(p1), 0o755)
	if err := os.WriteFile(p1, []byte("{}"), 0o644); err != nil {
		t.Fatal(err)
	}
	suffixed := strings.TrimSuffix(p1, ".json") + "-2.json"
	if err := os.WriteFile(suffixed, []byte("{}"), 0o644); err != nil {
		t.Fatal(err)
	}
	p3, err := newSessionPath()
	if err != nil {
		t.Fatalf("third path: %v", err)
	}
	if p3 != strings.TrimSuffix(p1, ".json")+"-3.json" {
		t.Errorf("expected the -3 suffix after both names are taken, got %s", p3)
	}
	// Same second means the same base name; the suffix is what separates.
	if p1 == p3 {
		t.Error("colliding names must be distinguished")
	}
}

// TestReplayTranscript checks the transcript matches the live progress
// shape: prompts and answers in full, tool calls as "● name(input)" and
// results as one indented line. Thinking blocks and empty content stay
// silent, and unparsable content is skipped rather than crashing.
func TestReplayTranscript(t *testing.T) {
	history := []message{
		textMessage("user", "fix the failing test"),
		blocksMessage("assistant", []contentBlock{
			{Type: "thinking", Thinking: "internal reasoning"},
			{Type: "text", Text: "looking at the suite…"},
			{Type: "tool_use", ID: "toolu_1", Name: "bash", Input: json.RawMessage(`{"command":"go test ./...","timeout":30}`)},
		}),
		blocksMessage("user", []contentBlock{
			{Type: "tool_result", ToolUseID: "toolu_1", Content: "ok\nFAIL detail"},
		}),
		textMessage("user", "now fix it"),
	}
	report, got := slashReporter()
	replayTranscript(history, report)
	joined := strings.Join(*got, "\n")
	want := []string{
		"replaying transcript",
		"· fix the failing test",
		"looking at the suite…",
		"● bash(go test ./...)",
		"  ok",
		"· now fix it",
	}
	for _, w := range want {
		if !strings.Contains(joined, w) {
			t.Errorf("transcript missing %q:\n%s", w, joined)
		}
	}
	if strings.Contains(joined, "internal reasoning") {
		t.Error("thinking blocks must not be replayed")
	}
	if strings.Contains(joined, "FAIL detail") {
		t.Error("tool results must be compressed to the first line")
	}
	// Hand-edited files may carry anything; replay must never crash.
	replayTranscript([]message{{Role: "assistant", Content: json.RawMessage(`[garbage`)}}, report)
	replayTranscript(nil, report)
}

// TestResumeReplaysTranscript checks that /resume replays the loaded
// session's conversation, so the task is back on screen before the next
// prompt is typed.
func TestResumeReplaysTranscript(t *testing.T) {
	cfgDir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", cfgDir)
	sessDir := filepath.Join(cfgDir, "harnais", "sessions")
	os.MkdirAll(sessDir, 0o755)
	os.WriteFile(filepath.Join(sessDir, "20240101-000000.json"), []byte(
		`{"version":1,"provider":"anthropic","history":[{"role":"user","content":"ship the fix"}]}`), 0o644)

	sess := &session{provider: "anthropic", cfg: config{model: "m"}}
	report, got := slashReporter()
	in := bufio.NewReader(strings.NewReader("/resume\n/exit\n"))
	if code := repl(context.Background(), sess, in, report); code != 0 {
		t.Fatalf("repl exited %d", code)
	}
	joined := strings.Join(*got, "\n")
	if !strings.Contains(joined, "replaying transcript") {
		t.Errorf("/resume must replay the transcript: %v", *got)
	}
	if !strings.Contains(joined, "ship the fix") {
		t.Errorf("replayed transcript must show the restored conversation: %v", *got)
	}
}

// TestSlashNew checks that /new saves the live conversation under its own
// file, hands the next turn a fresh session path, and keeps the backend
// and effort — so the new session starts clean but on the same setup, and
// the old one stays resumable.
func TestSlashNew(t *testing.T) {
	cfgDir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", cfgDir)
	sender := &fakeSender{t: t}
	sender.script = func(call int, req messageRequest) messageResponse {
		return messageResponse{
			ID:         fmt.Sprintf("msg_%d", call),
			Content:    []contentBlock{{Type: "text", Text: "done."}},
			StopReason: "end_turn",
		}
	}
	sess := &session{sender: sender, provider: "openrouter", cfg: config{model: "openai/gpt-5-mini", effort: "low"}}
	report, got := slashReporter()

	// Turn one fills the live session; /new then archives it.
	in := bufio.NewReader(strings.NewReader("first task\n/new\nsecond task\n/exit\n"))
	if code := repl(context.Background(), sess, in, report); code != 0 {
		t.Fatalf("repl exited %d", code)
	}
	joined := strings.Join(*got, "\n")
	if !strings.Contains(joined, "new session") {
		t.Errorf("/new must report the fresh start: %v", *got)
	}
	if sess.provider != "openrouter" || sess.cfg.model != "openai/gpt-5-mini" || sess.cfg.effort != "low" {
		t.Errorf("backend must survive /new: %s %+v", sess.provider, sess.cfg)
	}
	// The pre-/new conversation is on disk somewhere.
	sessDir := filepath.Join(cfgDir, "harnais", "sessions")
	entries, err := os.ReadDir(sessDir)
	if err != nil {
		t.Fatalf("read sessions dir: %v", err)
	}
	var first, second string
	for _, e := range entries {
		raw, err := os.ReadFile(filepath.Join(sessDir, e.Name()))
		if err != nil {
			continue
		}
		switch {
		case strings.Contains(string(raw), "first task"):
			first = string(raw)
		case strings.Contains(string(raw), "second task"):
			second = string(raw)
		}
	}
	if first == "" {
		t.Errorf("the pre-/new conversation must be saved: %v", entries)
	}
	if second == "" {
		t.Errorf("the post-/new turn must be saved too: %v", entries)
	}
	// The new session must live under a fresh path — the turn after /new
	// auto-saves there (saved=true again), never into the archived file.
	if sess.savePath == "" {
		t.Fatal("after /new the session must have a fresh save path")
	}
	if raw, err := os.ReadFile(sess.savePath); err != nil || !strings.Contains(string(raw), "second task") {
		t.Errorf("the fresh session file must hold the post-/new turn: %v %s", err, raw)
	}
	if first != "" && second != "" && strings.Contains(first, "second task") {
		t.Error("the archived file must not hold post-/new turns")
	}
	// And the archived session is actually resumable via the normal path.
	st, _, err := latestSession(sess.savePath)
	if err != nil {
		t.Fatalf("archived session not resumable: %v", err)
	}
	blob, _ := json.Marshal(st.History)
	if !strings.Contains(string(blob), "first task") {
		t.Errorf("resuming the archive must bring back the old conversation: %s", blob)
	}
}

// TestApplySessionStateFallsBackFieldByField checks that a saved backend
// harnais no longer knows (or a bad effort) cannot poison the resumed one.
func TestApplySessionStateFallsBackFieldByField(t *testing.T) {
	sess := &session{provider: "anthropic", cfg: config{model: "m"}}
	applySessionState(sess, sessionState{
		Provider: "no-such-backend",
		Effort:   "turbo",
		History:  []message{textMessage("user", "hi")},
	}, "/tmp/s.json")
	if sess.provider != "anthropic" {
		t.Errorf("unknown provider must fall back, got %s", sess.provider)
	}
	if sess.cfg.effort != "" {
		t.Errorf("invalid effort must fall back, got %q", sess.cfg.effort)
	}
	if len(sess.history) != 1 {
		t.Errorf("history should load regardless: %d", len(sess.history))
	}
	if sess.savePath != "/tmp/s.json" || !sess.saved {
		t.Error("resumed sessions must record their path and count as saved")
	}
}

// TestReplAutoSavesSession checks the interactive loop writes the session
// after a turn, so a later -resume continues it.
func TestReplAutoSavesSession(t *testing.T) {
	cfgDir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", cfgDir)
	sender := &fakeSender{t: t}
	sender.script = func(call int, req messageRequest) messageResponse {
		return messageResponse{
			ID:         fmt.Sprintf("msg_%d", call),
			Content:    []contentBlock{{Type: "text", Text: "done."}},
			StopReason: "end_turn",
		}
	}
	sess := &session{sender: sender, provider: "anthropic", cfg: config{model: "m"}}
	report, got := slashReporter()
	in := bufio.NewReader(strings.NewReader("hello\n/exit\n"))
	if code := repl(context.Background(), sess, in, report); code != 0 {
		t.Fatalf("repl exited %d", code)
	}
	if sess.savePath == "" {
		t.Fatal("a fresh session must get a save path")
	}
	raw, err := os.ReadFile(sess.savePath)
	if err != nil {
		t.Fatalf("session file not written: %v", err)
	}
	var st sessionState
	if err := json.Unmarshal(raw, &st); err != nil {
		t.Fatalf("session file unparsable: %v", err)
	}
	if len(st.History) != 2 { // user prompt + assistant answer
		t.Fatalf("session history wrong: %d messages", len(st.History))
	}
	// The first save announces where the session lives.
	if joined := strings.Join(*got, "\n"); !strings.Contains(joined, "-resume") {
		t.Errorf("first save should tell the user how to resume: %v", *got)
	}
}
