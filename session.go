package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// sessionState is the on-disk shape of a resumable session: the backend it
// ran on plus the full conversation history — roles, tool calls, tool
// results — which is exactly what the next turn needs to continue the task
// instead of restarting it.
type sessionState struct {
	Version  int       `json:"version"`
	Provider string    `json:"provider"`
	Model    string    `json:"model,omitempty"`
	Effort   string    `json:"effort,omitempty"`
	History  []message `json:"history"`
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
	st := sessionState{
		Version:  sessionStateVersion,
		Provider: sess.provider,
		Model:    sess.cfg.model,
		Effort:   sess.cfg.effort,
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

// latestSession returns the most recent session that actually holds a
// conversation. Files with empty history (a harnais run that never sent a
// prompt) are skipped, as are unreadable ones; each exclude omits one path
// (the live session, for /resume, which must not resume itself).
func latestSession(exclude ...string) (sessionState, string, error) {
	dir, err := sessionsDir()
	if err != nil {
		return sessionState{}, "", err
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return sessionState{}, "", fmt.Errorf("no saved sessions yet — pass -resume after a first session exists")
	}
	names := make([]string, 0, len(entries))
	for _, e := range entries {
		if !e.IsDir() && filepath.Ext(e.Name()) == ".json" {
			names = append(names, e.Name())
		}
	}
	// Timestamped names sort chronologically; newest first.
	sort.Sort(sort.Reverse(sort.StringSlice(names)))
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
		raw, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		var st sessionState
		if err := json.Unmarshal(raw, &st); err != nil || st.Version != sessionStateVersion {
			continue
		}
		if len(st.History) == 0 {
			continue
		}
		return st, path, nil
	}
	return sessionState{}, "", fmt.Errorf("no saved session with a conversation — pass -resume after a first session exists")
}

// replayTranscript prints a compact transcript of a resumed session, so
// the conversation reappears on screen instead of the REPL starting from
// a silent history. User prompts and assistant text print in full; tool
// calls and results compress to one line each, mirroring the live
// progress output. Thinking blocks and empty content stay silent.
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
					report("%s", b.Text)
				}
			case "tool_use":
				var input map[string]any
				if len(b.Input) > 0 {
					json.Unmarshal(b.Input, &input)
				}
				report("● %s(%s)", b.Name, summarizeInput(b.Name, input))
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
	if _, err := defaultModelFor(st.Provider); err == nil {
		sess.provider = st.Provider
	}
	if st.Model != "" {
		sess.cfg.model = st.Model
	}
	if validEffort(st.Effort) {
		sess.cfg.effort = st.Effort
	}
	sess.history = st.History
	sess.saved = true
}
