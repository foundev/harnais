package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

// sessionState is the on-disk shape of a resumable session: the backend it
// ran on plus the full conversation history — roles, tool calls, tool
// results — which is exactly what the next turn needs to continue the task
// instead of restarting it.
type sessionState struct {
	Version  int         `json:"version"`
	Provider string      `json:"provider"`
	Model    string      `json:"model,omitempty"`
	Effort   string      `json:"effort,omitempty"`
	Title    string      `json:"title,omitempty"`
	Goal     *threadGoal `json:"goal,omitempty"`
	History  []message   `json:"history"`
}

// sessionStateVersion bumps when the history shape changes in a way old
// files cannot satisfy; loadSession rejects other versions.
const sessionStateVersion = 1

// sessionsDir is where session files live, next to the stored config
// ($XDG_CONFIG_HOME/harnais/sessions).
func sessionsDir() (string, error) {
	cfgPath, err := configFilePath()
	if err != nil {
		return "", err
	}
	return filepath.Join(filepath.Dir(cfgPath), "sessions"), nil
}

// newSessionPath names a fresh session file after its creation time in UTC.
// The file is only written on the first save, so opening harnais and
// quitting without a prompt leaves nothing behind. A /new session started
// in the same second as the last one would collide on the timestamped
// name, so taken names get a numeric suffix (-2, -3, …) instead of
// silently overwriting.
func newSessionPath() (string, error) {
	dir, err := sessionsDir()
	if err != nil {
		return "", err
	}
	base := filepath.Join(dir, time.Now().UTC().Format("20060102-150405")+".json")
	path := base
	for i := 2; ; i++ {
		if _, err := os.Stat(path); os.IsNotExist(err) {
			return path, nil
		} else if err != nil {
			return "", err
		}
		path = strings.TrimSuffix(base, ".json") + fmt.Sprintf("-%d.json", i)
	}
}

// saveSession atomically writes the session's current state, so a crash
// mid-write can never leave a half-readable file behind.
func saveSession(sess *session) error {
	if sess.savePath == "" {
		return fmt.Errorf("no session path available")
	}
	if sess.title == "" {
		// The title is the session's first prompt, fixed at first save so
		// later turns never rename it.
		sess.title = sessionTitle(sess.history)
	}
	st := sessionState{
		Version:  sessionStateVersion,
		Provider: sess.provider,
		Model:    sess.cfg.model,
		Effort:   sess.cfg.effort,
		Title:    sess.title,
		Goal:     sess.goal,
		History:  sess.history,
	}
	if err := os.MkdirAll(filepath.Dir(sess.savePath), 0o755); err != nil {
		return fmt.Errorf("create session dir: %w", err)
	}
	raw, err := json.MarshalIndent(st, "", "  ")
	if err != nil {
		return fmt.Errorf("encode session: %w", err)
	}
	tmp, err := os.CreateTemp(filepath.Dir(sess.savePath), ".session-*.tmp")
	if err != nil {
		return fmt.Errorf("write session: %w", err)
	}
	tmpName := tmp.Name()
	if _, err := tmp.Write(append(raw, '\n')); err != nil {
		tmp.Close()
		os.Remove(tmpName)
		return fmt.Errorf("write session: %w", err)
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmpName)
		return fmt.Errorf("write session: %w", err)
	}
	if err := os.Rename(tmpName, sess.savePath); err != nil {
		os.Remove(tmpName)
		return fmt.Errorf("write session: %w", err)
	}
	return nil
}

// resumeChoice is one resumable session as /resume and its completer
// present it: where it lives, what to call it, and how big it is.
type resumeChoice struct {
	Path  string
	Title string
	When  time.Time
	Nmsgs int
}

// label renders a choice for the picker list and completion rows: the
// title first (it is the thing being matched), then when and how much,
// dimmed so the title stays the thing being read.
func (c resumeChoice) label() string {
	detail := paint(ansiDim, fmt.Sprintf("  — %s · %d messages", c.When.Format("2006-01-02 15:04"), c.Nmsgs))
	return c.Title + detail
}

// sessionTitle derives a session's name from its first user prompt:
// one line, trimmed, capped so the picker and completion rows stay
// readable. Empty history (or a history with no plain user text — a
// hand-edited file) leaves the title empty.
func sessionTitle(history []message) string {
	for _, msg := range history {
		if msg.Role != "user" {
			continue
		}
		var text string
		if err := json.Unmarshal(msg.Content, &text); err != nil || text == "" {
			continue
		}
		text = strings.Join(strings.Fields(text), " ")
		runes := []rune(text)
		if len(runes) > 60 {
			return string(runes[:60]) + "…"
		}
		return text
	}
	return ""
}

// listSessions returns every session that actually holds a conversation,
// newest first. Files with empty history (a harnais run that never sent a
// prompt) are skipped, as are unreadable ones; each exclude omits one path
// (the live session, for /resume, which must not resume itself). Titles
// come from the saved file, falling back to the first prompt for files
// written before sessions had titles.
func listSessions(exclude ...string) []resumeChoice {
	dir, err := sessionsDir()
	if err != nil {
		return nil
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	names := make([]string, 0, len(entries))
	for _, e := range entries {
		if !e.IsDir() && filepath.Ext(e.Name()) == ".json" {
			names = append(names, e.Name())
		}
	}
	// Timestamped names sort chronologically; newest first.
	sort.Sort(sort.Reverse(sort.StringSlice(names)))
	var out []resumeChoice
	for _, name := range names {
		path := filepath.Join(dir, name)
		excluded := false
		for _, x := range exclude {
			if sameFile(path, x) {
				excluded = true
				break
			}
		}
		if excluded {
			continue
		}
		st, err := loadSessionState(path)
		if err != nil {
			continue
		}
		if len(st.History) == 0 {
			continue
		}
		title := st.Title
		if title == "" {
			title = sessionTitle(st.History)
		}
		if title == "" {
			title = strings.TrimSuffix(name, ".json")
		}
		out = append(out, resumeChoice{
			Path:  path,
			Title: title,
			When:  sessionFileTime(name, path),
			Nmsgs: len(st.History),
		})
	}
	return out
}

// loadSessionState reads and validates one session file.
func loadSessionState(path string) (sessionState, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return sessionState{}, err
	}
	var st sessionState
	if err := json.Unmarshal(raw, &st); err != nil {
		return sessionState{}, err
	}
	if st.Version != sessionStateVersion {
		return sessionState{}, fmt.Errorf("%s: session version %d", filepath.Base(path), st.Version)
	}
	return st, nil
}

// sessionFileTime recovers a session's start time from its timestamped
// name (suffixes like -2 included), falling back to the file's mtime for
// names that do not parse.
func sessionFileTime(name, path string) time.Time {
	if t, err := time.Parse("20060102-150405", strings.TrimSuffix(name, ".json")); err == nil {
		return t
	}
	base := filepath.Base(path)
	if i := strings.IndexByte(base, '-'); i >= 0 {
		if t, err := time.Parse("20060102-150405", base[:i]); err == nil {
			return t
		}
	}
	if fi, err := os.Stat(path); err == nil {
		return fi.ModTime()
	}
	return time.Time{}
}

// filterSessions keeps the choices whose title, file name, or list
// number matches the typed words, case-insensitively — the same matching
// the /resume completer offers. Digits alone are ambiguous: file names
// are dates, so digit words match a list number first, and only widen to
// file-name matching when no number starts with them.
func filterSessions(choices []resumeChoice, words string) []resumeChoice {
	needle := strings.ToLower(strings.TrimSpace(words))
	if needle == "" {
		return choices
	}
	numeric := strings.IndexFunc(needle, func(r rune) bool { return r < '0' || r > '9' }) < 0
	if numeric {
		var numbered []resumeChoice
		for i, c := range choices {
			if strings.HasPrefix(strconv.Itoa(i+1), needle) {
				numbered = append(numbered, c)
			}
		}
		if len(numbered) > 0 {
			return numbered
		}
	}
	var out []resumeChoice
	for _, c := range choices {
		if strings.Contains(strings.ToLower(c.Title), needle) ||
			strings.Contains(strings.ToLower(filepath.Base(c.Path)), needle) {
			out = append(out, c)
		}
	}
	return out
}

// latestSession returns the most recent session that actually holds a
// conversation (see listSessions for what qualifies).
func latestSession(exclude ...string) (sessionState, string, error) {
	choices := listSessions(exclude...)
	if len(choices) == 0 {
		return sessionState{}, "", fmt.Errorf("no saved session with a conversation — pass -resume after a first session exists")
	}
	st, err := loadSessionState(choices[0].Path)
	if err != nil {
		return sessionState{}, "", err
	}
	return st, choices[0].Path, nil
}

// replayTranscript prints a compact transcript of a resumed session, so
// the conversation reappears on screen instead of the REPL starting from
// a silent history. User prompts and assistant text print in full — the
// text through paintAnswer, so answers keep their markdown highlighting;
// tool calls and results compress to one line each, mirroring the live
// progress output including its colors. Thinking blocks and empty
// content stay silent.
func replayTranscript(history []message, report progressFunc) {
	report("%s", paint(ansiDim, "—— replaying transcript ———"))
	for _, msg := range history {
		// Plain-string content is always a spoken turn: a user prompt,
		// or (from hand-edited files) an assistant reply.
		var text string
		if err := json.Unmarshal(msg.Content, &text); err == nil {
			if text != "" {
				report("· %s", text)
			}
			continue
		}
		var blocks []contentBlock
		if err := json.Unmarshal(msg.Content, &blocks); err != nil {
			continue
		}
		for _, b := range blocks {
			switch b.Type {
			case "text":
				if b.Text != "" {
					report("%s", paintAnswer(b.Text))
				}
			case "tool_use":
				var input map[string]any
				if len(b.Input) > 0 {
					json.Unmarshal(b.Input, &input)
				}
				report("%s", paint(ansiCyan, fmt.Sprintf("● %s(%s)", b.Name, summarizeInput(b.Name, input))))
			case "tool_result":
				report("  %s", firstLine(b.Content))
			}
		}
	}
}

// sameFile reports whether two paths name the same file: true when both
// exist and stat to the same inode, or when their cleaned paths are equal
// (covers not-yet-written sessions). Either path may be empty.
func sameFile(a, b string) bool {
	if a == "" || b == "" {
		return false
	}
	if fa, err := os.Stat(a); err == nil {
		if fb, err := os.Stat(b); err == nil {
			return os.SameFile(fa, fb)
		}
	}
	return filepath.Clean(a) == filepath.Clean(b)
}

// applySessionState loads a saved session into the live one: history plus
// the backend, model, and effort it ran on. Anything unusable falls back
// field-by-field so an old or hand-edited file still opens.
func applySessionState(sess *session, st sessionState, path string) {
	sess.savePath = path
	sess.title = st.Title
	if _, err := defaultModelFor(st.Provider); err == nil {
		sess.provider = st.Provider
	}
	if st.Model != "" {
		sess.cfg.model = st.Model
	}
	if validEffort(st.Effort) {
		sess.cfg.effort = st.Effort
	}
	if st.Goal != nil && st.Goal.Objective != "" {
		sess.goal = st.Goal
		sess.cfg.goal = st.Goal.Objective
	} else {
		sess.goal = nil
		sess.cfg.goal = ""
	}
	sess.history = st.History
	sess.saved = true
}
