package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// saveChoiceSession writes one session file with the given title and
// first prompt into the test's sessions dir, returning its path. Files
// within one test share a dir (t.TempDir per call would give each its
// own XDG root, and /resume only lists one), so tests t.Setenv once and
// call this repeatedly.
func saveChoiceSession(t *testing.T, name, title, prompt string) string {
	t.Helper()
	sessDir := filepath.Join(os.Getenv("XDG_CONFIG_HOME"), "harnais", "sessions")
	os.MkdirAll(sessDir, 0o755)
	path := filepath.Join(sessDir, name)
	raw := `{"version":1,"provider":"anthropic","title":` + quoteJSON(title) +
		`,"history":[{"role":"user","content":` + quoteJSON(prompt) + `}]}`
	if err := os.WriteFile(path, []byte(raw), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

// quoteJSON renders s as a JSON string literal.
func quoteJSON(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}

// TestSessionTitleFromFirstPrompt checks saveSession names the session
// after its first user prompt, and that /new and /reset retire the title
// with the conversation so the next session is not mislabeled.
func TestSessionTitleFromFirstPrompt(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir)
	sess := &session{
		provider: "anthropic",
		cfg:      config{model: "m"},
		savePath: filepath.Join(dir, "harnais", "sessions", "20240101-000000.json"),
		history:  []message{textMessage("user", "fix the failing test\nand the lint")},
	}
	if err := saveSession(sess); err != nil {
		t.Fatal(err)
	}
	if sess.title != "fix the failing test and the lint" {
		t.Errorf("title = %q, want the first prompt on one line", sess.title)
	}
	// Long prompts cap so picker rows stay readable.
	sess.title = ""
	sess.history = []message{textMessage("user", strings.Repeat("x", 80))}
	if err := saveSession(sess); err != nil {
		t.Fatal(err)
	}
	if len([]rune(sess.title)) != 61 || !strings.HasSuffix(sess.title, "…") {
		t.Errorf("long title = %d runes, want 61 ending in …", len([]rune(sess.title)))
	}
}

// TestSlashResumePicker checks bare /resume in an interactive REPL shows
// the saved sessions and resumes the one picked, and that backing out
// keeps the live session untouched.
func TestSlashResumePicker(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	older := saveChoiceSession(t, "20240101-000000.json", "first task", "first task")
	newer := saveChoiceSession(t, "20240102-000000.json", "second task", "second task")

	var offered []resumeChoice
	sess := &session{
		provider: "anthropic",
		cfg:      config{model: "m"},
		pickResume: func(choices []resumeChoice, report progressFunc) (resumeChoice, bool) {
			offered = choices
			return choices[0], true
		},
	}
	report, got := slashReporter()
	if handled, exit := handleSlash(sess, "/resume", report); !handled || exit {
		t.Fatalf("/resume must be handled and not exit: %v %v", handled, exit)
	}
	if len(offered) != 2 || offered[0].Path != newer || offered[1].Path != older {
		t.Errorf("picker must offer both sessions, newest first: %+v", offered)
	}
	if len(sess.history) != 1 || sess.history[0].Role != "user" {
		t.Errorf("the picked session must load: %+v", sess.history)
	}
	if joined := strings.Join(*got, "\n"); !strings.Contains(joined, "resumed second task") {
		t.Errorf("resume must name the session title: %v", *got)
	}

	// Backing out leaves the live session alone.
	cancelled := &session{
		provider: "anthropic",
		cfg:      config{model: "m"},
		pickResume: func(choices []resumeChoice, report progressFunc) (resumeChoice, bool) {
			return resumeChoice{}, false
		},
	}
	report, got = slashReporter()
	handleSlash(cancelled, "/resume", report)
	if len(cancelled.history) != 0 {
		t.Errorf("a cancelled pick must not touch the live session: %+v", cancelled.history)
	}
	if joined := strings.Join(*got, "\n"); !strings.Contains(joined, "resume cancelled") {
		t.Errorf("cancelling must be reported: %v", *got)
	}
}

// TestSlashResumeByTitleWords checks /resume <words> matches session
// titles case-insensitively, reports ambiguity, and reports misses.
func TestSlashResumeByTitleWords(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	saveChoiceSession(t, "20240101-000000.json", "fix the login bug", "fix the login bug")
	saveChoiceSession(t, "20240102-000000.json", "fix the flaky test", "fix the flaky test")

	// One match loads it.
	sess := &session{provider: "anthropic", cfg: config{model: "m"}}
	report, got := slashReporter()
	handleSlash(sess, "/resume FLAKY", report)
	blob, _ := json.Marshal(sess.history)
	if len(sess.history) != 1 || !strings.Contains(string(blob), "flaky") {
		t.Errorf("one title match must load that session: %+v", sess.history)
	}
	if joined := strings.Join(*got, "\n"); !strings.Contains(joined, "resumed fix the flaky test") {
		t.Errorf("resume must name the session title: %v", *got)
	}

	// Several matches are ambiguous, not silently the newest.
	sess = &session{provider: "anthropic", cfg: config{model: "m"}}
	report, got = slashReporter()
	handleSlash(sess, "/resume fix the", report)
	if last := lastReport(t, got); !strings.Contains(last, "matches 2 sessions") {
		t.Errorf("ambiguous words must say so, got %q", last)
	}

	// No match is a clean report.
	sess = &session{provider: "anthropic", cfg: config{model: "m"}}
	report, got = slashReporter()
	handleSlash(sess, "/resume nonexistent", report)
	if last := lastReport(t, got); !strings.Contains(last, "no saved session matches") {
		t.Errorf("a miss must be reported, got %q", last)
	}
}

// TestSessionCompletions checks the /resume completer offers sessions by
// title, file name, and list number, with the number as the inserted
// value and the full choice as the label.
func TestSessionCompletions(t *testing.T) {
	choices := []resumeChoice{
		{Path: "/s/20240101-000000.json", Title: "fix the login bug", Nmsgs: 3},
		{Path: "/s/20240102-000000.json", Title: "fix the flaky test", Nmsgs: 5},
	}
	if got := sessionCompletions(choices, "/resume "); len(got) != 2 {
		t.Errorf("empty word must offer every session, got %+v", got)
	}
	if got := sessionCompletions(choices, "/resume flak"); len(got) != 1 || got[0].Value != "2" {
		t.Errorf("typing part of a title must narrow the list: %+v", got)
	}
	if got := sessionCompletions(choices, "/resume 1"); len(got) != 1 || got[0].Value != "1" || !strings.Contains(got[0].Label, "fix the login bug") {
		t.Errorf("typing a list number must offer only that session: %+v", got)
	}
	if got := sessionCompletions(choices, "/resume 20240102"); len(got) != 1 || got[0].Value != "2" {
		t.Errorf("typing a file name must offer that session: %+v", got)
	}
	if got := sessionCompletions(choices, "/resume zzz"); got != nil {
		t.Errorf("a miss must offer nothing: %+v", got)
	}
}

// TestSlashResumePickerFiltersWords checks that words typed after /resume
// pre-filter the picker's list.
func TestSlashResumePickerFiltersWords(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	saveChoiceSession(t, "20240101-000000.json", "first task", "first task")
	saveChoiceSession(t, "20240102-000000.json", "second task", "second task")

	var offered []resumeChoice
	sess := &session{
		provider: "anthropic",
		cfg:      config{model: "m"},
		pickResume: func(choices []resumeChoice, report progressFunc) (resumeChoice, bool) {
			offered = choices
			return choices[0], true
		},
	}
	report, _ := slashReporter()
	handleSlash(sess, "/resume second", report)
	if len(offered) != 1 || offered[0].Title != "second task" {
		t.Errorf("typed words must pre-filter the picker: %+v", offered)
	}
}

// TestListSessionsTitlesOldFiles checks that session files written before
// sessions had titles still list under their first prompt.
func TestListSessionsTitlesOldFiles(t *testing.T) {
	cfgDir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", cfgDir)
	sessDir := filepath.Join(cfgDir, "harnais", "sessions")
	os.MkdirAll(sessDir, 0o755)
	os.WriteFile(filepath.Join(sessDir, "20240101-000000.json"), []byte(
		`{"version":1,"provider":"anthropic","history":[{"role":"user","content":"an older session"}]}`), 0o644)

	choices := listSessions("")
	if len(choices) != 1 {
		t.Fatalf("expected one choice, got %+v", choices)
	}
	if choices[0].Title != "an older session" {
		t.Errorf("old files must title from their first prompt, got %q", choices[0].Title)
	}
	if choices[0].When.IsZero() || choices[0].Nmsgs != 1 {
		t.Errorf("choice must carry its start time and size: %+v", choices[0])
	}
}
