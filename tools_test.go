package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestWriteReadRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "sub", "note.txt")
	out, err := executeTool(context.Background(), "write", map[string]any{"path": path, "content": "hello\nworld\n"})
	if err != nil {
		t.Fatalf("write: %v", err)
	}
	got, err := executeTool(context.Background(), "read", map[string]any{"path": path})
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if !strings.Contains(got, "1|hello") || !strings.Contains(got, "2|world") {
		t.Fatalf("unexpected read output: %q (write said %q)", got, out)
	}
}

func TestEditRequiresUniqueMatch(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "f.txt")
	if _, err := executeTool(context.Background(), "write", map[string]any{"path": path, "content": "a\nb\na\n"}); err != nil {
		t.Fatal(err)
	}
	if _, err := executeTool(context.Background(), "edit", map[string]any{"path": path, "find": "a", "replace": "z"}); err == nil {
		t.Fatal("expected ambiguity error for repeated match")
	}
	if _, err := executeTool(context.Background(), "edit", map[string]any{"path": path, "find": "b", "replace": "B"}); err != nil {
		t.Fatalf("unique edit failed: %v", err)
	}
	raw, _ := os.ReadFile(path)
	if string(raw) != "a\nB\na\n" {
		t.Fatalf("unexpected content: %q", raw)
	}
}

func TestReadWindow(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "f.txt")
	if _, err := executeTool(context.Background(), "write", map[string]any{"path": path, "content": "1\n2\n3\n4\n"}); err != nil {
		t.Fatal(err)
	}
	got, err := executeTool(context.Background(), "read", map[string]any{"path": path, "offset": 2.0, "limit": 2.0})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(got, "2|2") || !strings.Contains(got, "3|3") || strings.Contains(got, "4|4") {
		t.Fatalf("unexpected window: %q", got)
	}
}

func TestBashEchoAndFailure(t *testing.T) {
	out, err := executeTool(context.Background(), "bash", map[string]any{"command": "echo hello"})
	if err != nil || !strings.Contains(out, "hello") {
		t.Fatalf("echo failed: out=%q err=%v", out, err)
	}
	if _, err := executeTool(context.Background(), "bash", map[string]any{"command": "exit 3"}); err == nil {
		t.Fatal("expected error for non-zero exit")
	}
}

func TestUnknownTool(t *testing.T) {
	if _, err := executeTool(context.Background(), "browse", map[string]any{}); err == nil {
		t.Fatal("expected error for unknown tool")
	}
}
