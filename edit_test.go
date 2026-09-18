package main

import (
	"bytes"
	"strings"
	"testing"
)

func readEditor(t *testing.T, input string, complete func(string) []string) (string, bool, string) {
	t.Helper()
	var out bytes.Buffer
	line, exit := readLineRaw(strings.NewReader(input), &out, "> ", complete)
	return line, exit, out.String()
}

func TestReadLineRawTypingAndSubmit(t *testing.T) {
	line, exit, out := readEditor(t, "hi there\r", nil)
	if exit || line != "hi there" {
		t.Fatalf("line=%q exit=%v", line, exit)
	}
	if !strings.Contains(out, "hi there") {
		t.Errorf("typed text must be echoed: %q", out)
	}
	line, _, _ = readEditor(t, "hi there\n", nil)
	if line != "hi there" {
		t.Fatalf("\\n must submit too, got %q", line)
	}
}

func TestReadLineRawBackspace(t *testing.T) {
	line, _, _ := readEditor(t, "abc\x7f\x7fd\r", nil)
	if line != "ad" {
		t.Fatalf("DEL must drop one rune per press, got %q", line)
	}
	// UTF-8: one backspace removes the whole rune.
	line, _, _ = readEditor(t, "aé\x7f\r", nil)
	if line != "a" {
		t.Fatalf("DEL must remove a full UTF-8 rune, got %q", line)
	}
	// Backspace on an empty line is a no-op.
	line, _, _ = readEditor(t, "\x7f\r", nil)
	if line != "" {
		t.Fatalf("DEL on empty line, got %q", line)
	}
}

func TestReadLineRawCtrlUClearsAndCtrlWDeletesWord(t *testing.T) {
	line, _, _ := readEditor(t, "one two\x15three\r", nil)
	if line != "three" {
		t.Fatalf("^U must clear, got %q", line)
	}
	line, _, _ = readEditor(t, "one two  \x17\r", nil)
	if line != "one " {
		t.Fatalf("^W must drop the trailing word and spaces, got %q", line)
	}
}

func TestReadLineRawCtrlCAndCtrlD(t *testing.T) {
	line, exit, out := readEditor(t, "ab\x03hi\r", nil)
	if exit || line != "hi" {
		t.Fatalf("^C on text must clear the line and keep reading, line=%q exit=%v", line, exit)
	}
	if !strings.Contains(out, "^C") {
		t.Errorf("^C must be shown, got %q", out)
	}
	_, exit, _ = readEditor(t, "\x03", nil)
	if !exit {
		t.Fatal("^C on an empty line must exit")
	}
	_, exit, _ = readEditor(t, "\x04", nil)
	if !exit {
		t.Fatal("^D on an empty line must exit")
	}
	// ^D on text is ignored (there is no cursor to delete at).
	line, exit, _ = readEditor(t, "ab\x04\r", nil)
	if exit || line != "ab" {
		t.Fatalf("^D on text must be ignored, line=%q exit=%v", line, exit)
	}
}

func TestReadLineRawEscSequencesIgnored(t *testing.T) {
	// Left arrow, then a typed letter, then submit.
	line, exit, out := readEditor(t, "ab\x1b[Dc\r", nil)
	if exit || line != "abc" {
		t.Fatalf("escape sequences must be swallowed, line=%q exit=%v", line, exit)
	}
	if strings.Contains(out, "\x1b[D") {
		t.Errorf("escape sequence must never be echoed: %q", out)
	}
}

func TestReadLineRawTabCompletes(t *testing.T) {
	complete := func(line string) []string {
		return filterPrefix([]string{"zai-org/GLM-5.2", "zai-org/GLM-5.3", "deepseek-ai/DeepSeek-V4-Flash-0731"}, lastWord(line))
	}
	// Single candidate: insert with a trailing space.
	line, exit, _ := readEditor(t, "/model deep\x09\r", complete)
	if exit || line != "/model deepseek-ai/DeepSeek-V4-Flash-0731 " {
		t.Fatalf("single candidate must insert, got %q", line)
	}
	// Several candidates: extend by the longest common prefix.
	line, _, _ = readEditor(t, "/model z\x09\r", complete)
	if line != "/model zai-org/GLM-5." {
		t.Fatalf("common prefix must extend, got %q", line)
	}
	// Stuck TAB then TAB again: list the candidates.
	line, _, out := readEditor(t, "/model zai-org/GLM-5.\x09\x09\r", complete)
	if line != "/model zai-org/GLM-5." {
		t.Fatalf("listing must not change the line, got %q", line)
	}
	if !strings.Contains(out, "zai-org/GLM-5.2  zai-org/GLM-5.3") {
		t.Errorf("second stuck TAB must list candidates, got %q", out)
	}
	// No candidates: TAB does nothing.
	line, _, _ = readEditor(t, "/model nope\x09\r", complete)
	if line != "/model nope" {
		t.Fatalf("TAB without candidates must be a no-op, got %q", line)
	}
}

func TestLongestCommonPrefix(t *testing.T) {
	cases := []struct {
		cands []string
		want  string
	}{
		{nil, ""},
		{[]string{"one"}, "one"},
		{[]string{"ab", "ac", "ad"}, "a"},
		{[]string{"same"}, "same"},
		{[]string{"x", "y"}, ""},
	}
	for _, c := range cases {
		if got := longestCommonPrefix(c.cands); got != c.want {
			t.Errorf("longestCommonPrefix(%v) = %q, want %q", c.cands, got, c.want)
		}
	}
}

func TestTabEditWordExtraction(t *testing.T) {
	// The completed word starts after the last space, so a second word
	// never gets the first one re-inserted.
	complete := func(line string) []string {
		return filterPrefix([]string{"model-b"}, lastWord(line))
	}
	line, _, _ := readEditor(t, "/provider x\x09 \x09\r", complete)
	// First TAB completes the empty word to "model-b "; the second TAB
	// then sees word "" again and lists (no further change).
	if line != "/provider x model-b " {
		t.Fatalf("word extraction wrong, got %q", line)
	}
}

func TestTrimLastWord(t *testing.T) {
	cases := []struct{ in, want string }{
		{"one two", "one "},
		{"one two  ", "one "},
		{"  ", ""},
		{"solo", ""},
		{"", ""},
	}
	for _, c := range cases {
		if got := string(trimLastWord([]byte(c.in))); got != c.want {
			t.Errorf("trimLastWord(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}
