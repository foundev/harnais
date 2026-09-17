package main

import (
	"bytes"
	"strings"
	"testing"
	"time"
)

func TestSpinnerNoTTYIsSilent(t *testing.T) {
	var buf bytes.Buffer
	done := make(chan struct{})
	go func() {
		startSpinnerTo(&buf, "thinking").finish()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("finish on a non-TTY spinner must return immediately")
	}
	if buf.Len() != 0 {
		t.Errorf("non-TTY spinner must not write, wrote %q", buf.String())
	}
}

func TestSpinnerAnimatesAndClears(t *testing.T) {
	var buf bytes.Buffer
	s := &spinner{stop: make(chan struct{}), done: make(chan struct{})}
	go s.run(&buf, "thinking")
	deadline := time.Now().Add(2 * time.Second)
	for buf.Len() == 0 && time.Now().Before(deadline) {
		time.Sleep(20 * time.Millisecond)
	}
	s.finish()
	out := buf.String()
	if !strings.Contains(out, "\r") {
		t.Errorf("spinner should carriage-return its line: %q", out)
	}
	frameSeen := false
	for _, f := range spinnerFrames {
		if strings.Contains(out, f) {
			frameSeen = true
			break
		}
	}
	if !frameSeen {
		t.Errorf("no spinner frame rendered: %q", out)
	}
	if !strings.HasSuffix(out, "\r\033[K") {
		t.Errorf("spinner should clear its line on finish: %q", out)
	}
}
