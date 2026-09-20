package main

import (
	"bytes"
	"fmt"
	"strings"
	"testing"
)

func readEditor(t *testing.T, input string, complete completer) (string, bool, string) {
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

func TestReadLineRawListsCompletionsWhileTyping(t *testing.T) {
	complete := func(line string) []completion {
		return prefixCompletions([]string{"zai-org/GLM-5.2", "zai-org/GLM-5.3"}, lastWord(line))
	}
	// No TAB anywhere: the candidates show up under the prompt as the
	// word is typed, and the submitted line is exactly what was typed.
	line, exit, out := readEditor(t, "/model z\r", complete)
	if exit || line != "/model z" {
		t.Fatalf("live suggestions must not change the line, line=%q exit=%v", line, exit)
	}
	if !strings.Contains(out, "zai-org/GLM-5.2") || !strings.Contains(out, "zai-org/GLM-5.3") {
		t.Errorf("typing must list the matches, got %q", out)
	}
	// A word nothing matches lists nothing.
	_, _, out = readEditor(t, "/model zzz\r", complete)
	// Only the final frame matters: the earlier ones may have shown
	// matches that the editor then erased.
	if last := out[strings.LastIndex(out, "\r\033[J"):]; strings.Contains(last, "GLM") {
		t.Errorf("no matches must suggest nothing, got %q", out)
	}
}

func TestReadLineRawArrowSelectsAndEnterTakes(t *testing.T) {
	complete := func(line string) []completion {
		return prefixCompletions([]string{"zai-org/GLM-5.2", "zai-org/GLM-5.3", "deepseek-ai/DeepSeek-V4-Flash-0731"}, lastWord(line))
	}
	// Typing alone never changes the line: Enter submits what was typed.
	line, exit, _ := readEditor(t, "/model z\r", complete)
	if exit || line != "/model z" {
		t.Fatalf("Enter without a highlight must submit the typed line, got %q", line)
	}
	// ↓ highlights the first match and Enter takes it. \x1b[B is the down
	// arrow (ESC [ B).
	line, _, out := readEditor(t, "/model z\x1b[B\r", complete)
	if line != "/model zai-org/GLM-5.2" {
		t.Fatalf("down arrow plus Enter must take the first match, got %q", line)
	}
	if !strings.Contains(out, "\x1b[7m  zai-org/GLM-5.2\x1b[0m") {
		t.Errorf("the highlighted row must be drawn in reverse video: %q", out)
	}
	// A second ↓ walks to the next match, and ↑ back up again.
	line, _, _ = readEditor(t, "/model z\x1b[B\x1b[B\r", complete)
	if line != "/model zai-org/GLM-5.3" {
		t.Fatalf("the highlight must walk down the list, got %q", line)
	}
	line, _, _ = readEditor(t, "/model z\x1b[B\x1b[B\x1b[A\r", complete)
	if line != "/model zai-org/GLM-5.2" {
		t.Fatalf("the up arrow must walk back, got %q", line)
	}
	// ↑ from nothing highlighted takes the last match.
	line, _, _ = readEditor(t, "/model z\x1b[A\r", complete)
	if line != "/model zai-org/GLM-5.3" {
		t.Fatalf("up from nothing must take the last match, got %q", line)
	}
	// Without the arrow, Enter submits the raw line even with matches on
	// screen.
	line, _, _ = readEditor(t, "/model nope\x1b[B\r", complete)
	if line != "/model nope" {
		t.Fatalf("Enter with nothing highlighted must not substitute, got %q", line)
	}
	// TAB is not a completion key any more: it does nothing.
	line, _, _ = readEditor(t, "/model deep\x09\r", complete)
	if line != "/model deep" {
		t.Fatalf("TAB must be ignored, got %q", line)
	}
}

func TestReadLineRawTypingDropsTheHighlight(t *testing.T) {
	complete := func(line string) []completion {
		return prefixCompletions([]string{"model-a", "model-b"}, lastWord(line))
	}
	// Highlight the first match, then keep typing: the highlight goes
	// away with the edit, so Enter submits the line as typed.
	line, _, _ := readEditor(t, "m\x1b[B-x\r", complete)
	if line != "m-x" {
		t.Fatalf("typing must drop the highlight, got %q", line)
	}
}

func TestReadLineRawClearsSuggestionBlock(t *testing.T) {
	var out bytes.Buffer
	e := &lineEditor{
		out:    &out,
		prompt: "> ",
		complete: func(line string) []completion {
			return prefixCompletions([]string{"one", "only"}, lastWord(line))
		},
	}
	e.line = []byte("o")
	e.draw()
	if got := out.String(); !strings.Contains(got, "only") || !strings.Contains(got, completionHint) {
		t.Fatalf("typing must draw the matches and the hint: %q", got)
	}
	// The next paint erases downwards from the prompt line before drawing
	// the new block, so lines never stack up.
	out.Reset()
	e.line = []byte("zz")
	e.draw()
	if got := out.String(); !strings.HasPrefix(got, "\r\x1b[J> zz") {
		t.Errorf("repainting must erase downwards and redraw the line: %q", got)
	}
	// Submitting drops the block without disturbing the typed line.
	out.Reset()
	e.line = []byte("o")
	e.draw()
	out.Reset()
	e.finishLine()
	if got := out.String(); got != "\x1b[J\r\n" {
		t.Errorf("submit must clear the block first, got %q", got)
	}
}

func TestTranscriptSurvivesTyping(t *testing.T) {
	// Regression: engaging the completion list used to walk the cursor up
	// past the prompt and erase the transcript under it. The editor may
	// only ever erase downwards, so history stays exactly where the user
	// left it — even when the block pushes the prompt up the screen.
	for _, history := range []int{20, 23} {
		var out bytes.Buffer
		for i := 0; i < history; i++ {
			fmt.Fprintf(&out, "history %02d\r\n", i)
		}
		complete := func(line string) []completion {
			return prefixCompletions([]string{"/model", "/provider", "/quit"}, lastWord(line))
		}
		readLineRaw(strings.NewReader("/mo\x1b[B\x1b[B\x1b[A\x1b[B\r"), &out, "deepseek> ", complete)

		term := newScreen(24, 80)
		term.feed(out.String())
		text := term.text()
		for i := 0; i < history; i++ {
			if !strings.Contains(text, fmt.Sprintf("history %02d", i)) {
				t.Fatalf("%d transcript lines, line %d erased by the completion list:\n%s", history, i, text)
			}
		}
	}
}

func TestPromptColumnsSkipsColors(t *testing.T) {
	if got := promptColumns("anthropic> "); got != 11 {
		t.Errorf("plain prompt width = %d, want 11", got)
	}
	if got := promptColumns("\033[1;36manthropic> \033[0m"); got != 11 {
		t.Errorf("colored prompt width = %d, want 11", got)
	}
}

func TestSuggestionRowsCapLongLists(t *testing.T) {
	all := make([]completion, 0, 12)
	for i := 1; i <= 12; i++ {
		all = append(all, completion{Value: fmt.Sprintf("model-%02d", i), Label: fmt.Sprintf("model-%02d", i)})
	}
	e := &lineEditor{prompt: "> ", complete: func(string) []completion { return all }}
	rows := suggestionRows(e.candidates())
	if len(rows) != maxSuggestions+2 {
		t.Fatalf("a long list must be capped, got %d rows", len(rows))
	}
	if !strings.Contains(rows[maxSuggestions], "4 more") {
		t.Errorf("the cap must say how many are hidden: %q", rows[maxSuggestions])
	}
	if !strings.Contains(rows[len(rows)-1], completionHint) {
		t.Errorf("the hint must sit under the matches: %q", rows[len(rows)-1])
	}
	if strings.Contains(strings.Join(rows, "\n"), "model-12") {
		t.Error("rows past the cap must not be drawn")
	}
}

func TestCompletionHintNamesTheKeys(t *testing.T) {
	// The hint is the whole tutorial: nobody should have to guess that the
	// list is driven by the arrows and Enter.
	for _, want := range []string{"↓/↑", "Enter"} {
		if !strings.Contains(completionHint, want) {
			t.Errorf("the completion hint must spell it out (%q missing): %q", want, completionHint)
		}
	}
}

func TestApplySelectionReplacesTheWord(t *testing.T) {
	// Only the word being typed is replaced: the command in front of it,
	// and a second word, survive.
	complete := func(line string) []completion {
		return prefixCompletions([]string{"model-b"}, lastWord(line))
	}
	line, _, _ := readEditor(t, "/provider x model-\x1b[B\r", complete)
	if line != "/provider x model-b" {
		t.Fatalf("selection must replace the word being typed, got %q", line)
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
