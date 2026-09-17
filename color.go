package main

import (
	"io"
	"os"
)

// Terminal color for interactive chrome only: prompts, spinner, progress,
// and errors on stderr. Never applied to model-facing text (tool results),
// stdout answers, or --help, so pipes, logs, and files stay clean.
//
// Honors NO_COLOR (https://no-color.org), CLICOLOR=0, and TERM=dumb, and
// stays off whenever stderr is not a TTY.

const (
	ansiRed      = "31"
	ansiGreen    = "32"
	ansiYellow   = "33"
	ansiCyan     = "36"
	ansiBold     = "1"
	ansiDim      = "2"
	ansiBoldCyan = "1;36"
)

func colorsOn() bool {
	return colorsOnFor(os.Stderr)
}

func colorsOnFor(w io.Writer) bool {
	if os.Getenv("NO_COLOR") != "" {
		return false
	}
	if os.Getenv("CLICOLOR") == "0" {
		return false
	}
	if os.Getenv("TERM") == "dumb" {
		return false
	}
	return isTerminal(w)
}

// colorize wraps unconditionally; paint wraps only when color is on.
func colorize(code, s string) string {
	return "\033[" + code + "m" + s + "\033[0m"
}

func paint(code, s string) string {
	if !colorsOn() {
		return s
	}
	return colorize(code, s)
}
