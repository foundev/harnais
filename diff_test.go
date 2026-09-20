package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"
)

// plain renders without color; colored rendering is checked separately via
// the escape sequences the pure renderer emits.
func plain(path, old, new string) []string {
	return renderFileDiff(path, old, new, false)
}

func TestRenderFileDiffNoChange(t *testing.T) {
	if got := plain("a.go", "x\n", "x\n"); got != nil {
		t.Errorf("identical content must render nothing, got %q", got)
	}
}

func TestRenderFileDiffHeaderCounts(t *testing.T) {
	got := plain("a.go", "a\nb\nc\n", "a\nB\nc\nd\n")
	if len(got) == 0 {
		t.Fatal("expected a diff")
	}
	for _, want := range []string{"── a.go", "+2", "-1"} {
		if !strings.Contains(got[0], want) {
			t.Errorf("header %q missing %q", got[0], want)
		}
	}
}

func TestRenderFileDiffSignsAndGutters(t *testing.T) {
	got := plain("a.txt", "one\ntwo\nthree\n", "one\nTWO\nthree")
	joined := strings.Join(got, "\n")
	// The shared last line is a common suffix: context, not a deletion.
	for _, want := range []string{"- two", "+ TWO", "3 3   three", "@@ -1,3 +1,3 @@"} {
		if !strings.Contains(joined, want) {
			t.Errorf("diff missing %q:\n%s", want, joined)
		}
	}
	// Line numbers sit in aligned gutters before the sign column: both
	// numbers, a sign, then the content.
	if !strings.Contains(joined, "1 1   one") {
		t.Errorf("gutter numbers missing:\n%s", joined)
	}
}

func TestRenderFileDiffNewFile(t *testing.T) {
	got := plain("new.txt", "", "hello\n")
	joined := strings.Join(got, "\n")
	if !strings.Contains(joined, "+ hello") || !strings.Contains(got[0], "+1") {
		t.Errorf("new-file diff wrong:\n%s", joined)
	}
}

func TestRenderFileDiffContextMerges(t *testing.T) {
	// Two changes within 2*diffContext+1 lines merge into one hunk.
	var old, new strings.Builder
	for i := 1; i <= 10; i++ {
		fmt.Fprintf(&old, "line %d\n", i)
		if i == 2 || i == 9 {
			fmt.Fprintf(&new, "LINE %d\n", i)
		} else {
			fmt.Fprintf(&new, "line %d\n", i)
		}
	}
	got := plain("a.txt", old.String(), new.String())
	hunks := 0
	for _, l := range got {
		if strings.HasPrefix(l, "@@") {
			hunks++
		}
	}
	if hunks != 1 {
		t.Errorf("close changes must merge into one hunk, got %d:\n%s", hunks, strings.Join(got, "\n"))
	}
}

func TestRenderFileDiffTrailingNewlineNotAChange(t *testing.T) {
	if got := plain("a.txt", "a\nb\n", "a\nb"); got != nil {
		t.Errorf("trailing-newline-only difference must be ignored, got %q", got)
	}
}

func TestRenderFileDiffColored(t *testing.T) {
	got := renderFileDiff("a.txt", "a\n", "b\n", true)
	joined := strings.Join(got, "\n")
	if !strings.Contains(joined, "\033[32m+\033[0m \033[32mb\033[0m") {
		t.Errorf("addition must be green:\n%q", joined)
	}
	if !strings.Contains(joined, "\033[31m-\033[0m \033[31ma\033[0m") {
		t.Errorf("deletion must be red:\n%q", joined)
	}
}

func TestRenderFileDiffLineCap(t *testing.T) {
	var old, new strings.Builder
	for i := 0; i < maxDiffLines+50; i++ {
		fmt.Fprintf(&old, "old %d\n", i)
		fmt.Fprintf(&new, "new %d\n", i)
	}
	got := plain("a.txt", old.String(), new.String())
	if len(got) != maxDiffLines {
		t.Fatalf("capped diff must be %d lines, got %d", maxDiffLines, len(got))
	}
	if last := got[len(got)-1]; !strings.Contains(last, "more lines") {
		t.Errorf("cap must be announced, last line %q", last)
	}
}

func TestRenderFileDiffTruncatesLongLines(t *testing.T) {
	long := strings.Repeat("x", maxDiffCols+40)
	got := plain("a.txt", "", long+"\n")
	joined := strings.Join(got, "\n")
	if !strings.Contains(joined, "…") {
		t.Errorf("overlong line must be truncated:\n%q", joined)
	}
	for _, l := range got {
		if strings.HasPrefix(l, "──") || strings.HasPrefix(l, "@@") {
			continue
		}
		// Strip the two gutters and sign to recover the content width.
		rest := strings.SplitN(l, " ", 4)[3]
		if n := len([]rune(rest)) - 2; n > maxDiffCols {
			t.Errorf("content exceeds the column cap: %d runes", n)
		}
	}
}

func TestComputeDiffShuffleMiddle(t *testing.T) {
	// Prefix/suffix trimming must not mislabel lines when the change is in
	// the middle: only the middle lines differ.
	got := computeDiff("a\nb\nc\n", "a\nX\nc\n")
	var kinds []byte
	for _, op := range got {
		kinds = append(kinds, op.kind)
	}
	if string(kinds) != " -+ " {
		t.Errorf("unexpected diff ops %q: %+v", string(kinds), got)
	}
}

// TestAgentLoopShowsFileDiff verifies the REPL wiring: an edit tool call in
// the loop reports a diff of the change to the user while the model keeps
// its plain result.
func TestAgentLoopShowsFileDiff(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/note.txt"
	if err := os.WriteFile(path, []byte("hello\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	sender := &fakeSender{t: t}
	sender.script = func(call int, req messageRequest) messageResponse {
		if len(req.Tools) == 0 {
			return reviewVerdictResponse("APPROVE")
		}
		if call == 1 {
			input := fmt.Sprintf(`{"path":%q,"find":"hello","replace":"hi"}`, path)
			return messageResponse{
				ID: "msg_1",
				Content: []contentBlock{
					{Type: "tool_use", ID: "toolu_1", Name: "edit", Input: json.RawMessage(input)},
				},
				StopReason: "tool_use",
			}
		}
		return messageResponse{
			ID:         "msg_2",
			Content:    []contentBlock{{Type: "text", Text: "done"}},
			StopReason: "end_turn",
		}
	}

	var reports []string
	report := func(format string, args ...any) {
		reports = append(reports, fmt.Sprintf(format, args...))
	}
	cfg := config{model: "test-model"}
	if _, _, err := runPrompt(context.Background(), sender, cfg, nil, "fix it", report); err != nil {
		t.Fatalf("runPrompt: %v", err)
	}
	var diffBlock string
	for _, r := range reports {
		if strings.Contains(r, "── "+path) {
			diffBlock = r
		}
	}
	if diffBlock == "" {
		t.Fatalf("no diff reported:\n%s", strings.Join(reports, "\n"))
	}
	// Reports are one formatted call per report; the diff lines follow the
	// header line, so check the whole transcript.
	blob := strings.Join(reports, "\n")
	for _, want := range []string{"- hello", "+ hi"} {
		if !strings.Contains(blob, want) {
			t.Errorf("diff missing %q:\n%s", want, blob)
		}
	}
	// Colors stay off off-TTY (tests are not terminals): no escapes.
	if strings.Contains(blob, "\033[") {
		t.Errorf("diff must be plain without a TTY:\n%s", blob)
	}
}
