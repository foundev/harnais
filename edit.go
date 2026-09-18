package main

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
)

// Line editing with TAB completion for the interactive REPL. The Go
// standard library does not expose termios, so the editor shells out to
// stty(1) — present on every Unix harnais targets — to switch stdin to
// raw mode, and only ever when both stdin and stderr are terminals (the
// REPL checks before calling in). Raw mode is held per line, not per
// session, so model output, progress, and errors keep their normal
// cooked-mode rendering. The byte-dispatch core (readLineRaw) is pure
// I/O and unit-tested without a terminal.

// readLineEdited reads one line from the terminal with editing and TAB
// completion. exit=true means the user asked to leave (^C or ^D on an
// empty line, or EOF). On any stty failure it degrades to a plain read:
// losing the editor must not lose the session.
func readLineEdited(prompt string, complete func(string) []string) (string, bool) {
	saved, err := stty("-g")
	if err != nil {
		return readLinePlain(), false
	}
	if _, err := stty("raw", "-echo"); err != nil {
		return readLinePlain(), false
	}
	defer stty(strings.TrimSpace(saved))
	return readLineRaw(os.Stdin, os.Stderr, prompt, complete)
}

// readLinePlain reads a line without any terminal setup (stty missing,
// unusual environments). Byte-at-a-time so no buffered reader swallows
// input the REPL still needs.
func readLinePlain() string {
	var buf [1]byte
	var line []byte
	for {
		n, err := os.Stdin.Read(buf[:])
		if n > 0 {
			if buf[0] == '\n' || buf[0] == '\r' {
				return string(line)
			}
			line = append(line, buf[0])
		}
		if err != nil {
			return string(line)
		}
	}
}

// stty runs stty(1) against the process terminal, returning its stdout
// (only -g prints anything).
func stty(args ...string) (string, error) {
	cmd := exec.Command("stty", args...)
	cmd.Stdin = os.Stdin
	var out strings.Builder
	cmd.Stdout = &out
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("stty %s: %w", strings.Join(args, " "), err)
	}
	return out.String(), nil
}

// readLineRaw is the editor core: it echoes to out, edits the line on
// backspace (^H, DEL), ^U (clear), ^W (delete word), submits on CR/LF,
// exits on ^C or ^D on an empty line (EOF included), ignores other
// control bytes and escape sequences (arrow keys), and completes on TAB:
// one match inserts it with a trailing space, several extend to the
// longest common prefix, and a TAB that extends nothing lists them.
func readLineRaw(in io.Reader, out io.Writer, prompt string, complete func(string) []string) (string, bool) {
	var line []byte
	redraw := func() {
		fmt.Fprintf(out, "\r\033[K%s%s", prompt, line)
	}
	fmt.Fprint(out, prompt)
	tabStuck := false // last TAB made no progress → the next one lists
	esc := 0          // escape-sequence state: 0 none, 1 after ESC, 2 inside ESC [
	buf := make([]byte, 256)
	for {
		n, err := in.Read(buf)
		for i := 0; i < n; i++ {
			b := buf[i]
			if esc > 0 {
				// Consume CSI/SS3 sequences (arrow keys and friends)
				// one final byte at a time; editing has no cursor
				// movement, so they are ignored, never echoed.
				if esc == 1 {
					esc = 0
					if b == '[' || b == 'O' {
						esc = 2
					}
				} else if b >= 0x40 && b <= 0x7e {
					esc = 0
				}
				continue
			}
			switch {
			case b == '\r' || b == '\n':
				fmt.Fprint(out, "\r\n")
				return string(line), false
			case b == '\t' && complete != nil:
				edited, stuck := tabEdit(string(line), tabStuck, complete, out)
				line, tabStuck = []byte(edited), stuck
				redraw()
			case b == 0x7f || b == 0x08:
				if len(line) > 0 {
					line = line[:len(line)-runeTailLen(line)]
					redraw()
				}
			case b == 0x03: // ^C: clear the line, or leave on empty
				fmt.Fprint(out, "^C\r\n")
				if len(line) == 0 {
					return "", true
				}
				line = line[:0]
				redraw()
			case b == 0x04: // ^D: leave on an empty line
				if len(line) == 0 {
					fmt.Fprint(out, "\r\n")
					return "", true
				}
			case b == 0x15: // ^U: clear the line
				line = line[:0]
				redraw()
			case b == 0x17: // ^W: delete the trailing word
				line = trimLastWord(line)
				redraw()
			case b == 0x1b:
				esc = 1
			case b < 0x20:
				// Other control bytes: ignored.
			default:
				line = append(line, b)
				redraw()
			}
		}
		if err != nil {
			return "", true
		}
	}
}

// tabEdit applies one TAB press to the line, printing the candidate
// listing itself when a stuck TAB asks for it.
func tabEdit(line string, tabStuck bool, complete func(string) []string, out io.Writer) (string, bool) {
	cands := complete(line)
	if len(cands) == 0 {
		return line, false
	}
	if len(cands) == 1 {
		word := lastWord(line)
		return line + cands[0][len(word):] + " ", false
	}
	word := lastWord(line)
	if lcp := longestCommonPrefix(cands); len(lcp) > len(word) {
		return line + lcp[len(word):], false
	}
	if tabStuck {
		// A second stuck TAB lists the candidates, readline-style; the
		// caller redraws the prompt line afterwards.
		fmt.Fprintf(out, "\r\n%s\r\n", strings.Join(cands, "  "))
	}
	return line, true
}

// lastWord is the text after the final space of the line.
func lastWord(line string) string {
	return line[strings.LastIndexByte(line, ' ')+1:]
}

// longestCommonPrefix is the shared start of all candidates.
func longestCommonPrefix(cands []string) string {
	if len(cands) == 0 {
		return ""
	}
	prefix := cands[0]
	for _, c := range cands[1:] {
		for !strings.HasPrefix(c, prefix) {
			prefix = prefix[:len(prefix)-1]
		}
	}
	return prefix
}

// trimLastWord drops trailing spaces and the word before them (^W).
func trimLastWord(line []byte) []byte {
	end := len(line)
	for end > 0 && line[end-1] == ' ' {
		end--
	}
	for end > 0 && line[end-1] != ' ' {
		end--
	}
	return line[:end]
}

// runeTailLen is how many bytes to drop to remove the last rune —
// UTF-8 continuation bytes plus the lead byte. Display width is not
// tracked (model ids are ASCII).
func runeTailLen(line []byte) int {
	n := 1
	for i := len(line) - 1; i > 0 && line[i]&0xc0 == 0x80; i-- {
		n++
	}
	return n
}
