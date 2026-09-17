package main

import (
	"strings"
	"testing"
)

func TestHighlightMarkdown(t *testing.T) {
	if got := highlightMarkdown("# Hi"); got != "\033[1m# Hi\033[0m" {
		t.Errorf("heading: %q", got)
	}
	if got := highlightMarkdown("run `go test` now"); !strings.Contains(got, "\033[33mgo test\033[0m") {
		t.Errorf("inline code: %q", got)
	}
	if got := highlightMarkdown("a **bold** move"); got != "a \033[1mbold\033[0m move" {
		t.Errorf("bold markers should hide: %q", got)
	}
	if got := highlightMarkdown("> quoted"); !strings.HasPrefix(got, "\033[32m>\033[0m") {
		t.Errorf("quote: %q", got)
	}
	if got := highlightMarkdown("- item"); !strings.HasPrefix(got, "\033[36m-\033[0m") {
		t.Errorf("list: %q", got)
	}
	if got := highlightMarkdown("1. first"); !strings.HasPrefix(got, "\033[36m1.\033[0m") {
		t.Errorf("ordered list: %q", got)
	}
}

func TestHighlightFence(t *testing.T) {
	in := "```go\nx := `y`\n```"
	got := highlightMarkdown(in)
	lines := strings.Split(got, "\n")
	if len(lines) != 3 {
		t.Fatalf("lines: %q", got)
	}
	if lines[0] != "\033[2m```go\033[0m" || lines[2] != "\033[2m```\033[0m" {
		t.Errorf("fences should be dim: %q", got)
	}
	if lines[1] != "x := `y`" {
		t.Errorf("fence content must stay plain (unmatched-marker risk): %q", lines[1])
	}
}

func TestHighlightUnmatchedMarkersPassThrough(t *testing.T) {
	for _, in := range []string{"a `b", "a **b", "plain #hashtag and 1.5 numbers"} {
		if got := highlightMarkdown(in); got != in {
			t.Errorf("unmatched markers must pass through: %q -> %q", in, got)
		}
	}
}

func TestPaintAnswerRespectsNoColor(t *testing.T) {
	t.Setenv("NO_COLOR", "1")
	if got := paintAnswer("# Hi `x`"); got != "# Hi `x`" {
		t.Errorf("NO_COLOR must leave answers plain: %q", got)
	}
}
