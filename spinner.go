package main

import (
	"fmt"
	"io"
	"os"
	"time"
)

// Spinner shown while blocked on a model response. TTY-gated: under pipes,
// CI, or tests there is no animation and start/finish are a no-op, so logs
// stay clean and one-shot stdout is never polluted (we only ever write to
// the given writer — os.Stderr in production).

var spinnerFrames = []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}

type spinner struct {
	stop chan struct{}
	done chan struct{}
}

func startSpinner(label string) *spinner {
	return startSpinnerTo(os.Stderr, label)
}

func startSpinnerTo(w io.Writer, label string) *spinner {
	s := &spinner{stop: make(chan struct{}), done: make(chan struct{})}
	if !isTerminal(w) {
		close(s.done)
		return s
	}
	go s.run(w, label)
	return s
}

// finish stops the animation and clears its line. Safe to call on a
// never-started spinner; each spinner must be finished exactly once.
func (s *spinner) finish() {
	select {
	case <-s.done:
	default:
		close(s.stop)
		<-s.done
	}
}

func (s *spinner) run(w io.Writer, label string) {
	defer close(s.done)
	start := time.Now()
	tick := time.NewTicker(100 * time.Millisecond)
	defer tick.Stop()
	i := 0
	for {
		select {
		case <-s.stop:
			fmt.Fprint(w, "\r\033[K")
			return
		case <-tick.C:
			fmt.Fprintf(w, "\r%s %s (%.0fs)", spinnerFrames[i%len(spinnerFrames)], label, time.Since(start).Seconds())
			i++
		}
	}
}

func isTerminal(w io.Writer) bool {
	f, ok := w.(*os.File)
	if !ok {
		return false
	}
	st, err := f.Stat()
	if err != nil {
		return false
	}
	return st.Mode()&os.ModeCharDevice != 0
}
