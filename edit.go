package main

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"unicode/utf8"
)

// Line editing with live completion for the interactive REPL. The Go
// standard library does not expose termios, so the editor shells out to
// stty(1) — present on every Unix harnais targets — to switch stdin to
// raw mode, and only ever when both stdin and stderr are terminals (the
// REPL checks before calling in). Raw mode is held per line, not per
// session, so model output, progress, and errors keep their normal
// cooked-mode rendering. The byte-dispatch core (readLineRaw) is pure
// I/O and unit-tested without a terminal.

// completion is one candidate the editor offers: Value is what Enter
// inserts for the highlighted row, Label is what the live list under the
// prompt shows (usually Value plus an annotation, such as the backend a
// model id belongs to).
type completion struct {
	Value string
	Label string
}

// completer returns the candidates for the line typed so far: nil when the
// line cannot be completed (a plain prompt) or nothing matches.
type completer func(line string) []completion

// maxSuggestions caps the live list so it stays short enough to keep the
// prompt on screen without the terminal scrolling it away.
const maxSuggestions = 8

// completionHint is the dim footer the list carries while it is on screen:
// walking the highlight with the arrows and taking a row with Enter is
// obvious if you have used a search box and invisible if you have not.
const completionHint = "↓/↑ to move · Enter to select"

// readLineEdited reads one line from the terminal with editing and live
// completion. exit=true means the user asked to leave (^C or ^D on an
// empty line, or EOF). On any stty failure it degrades to a plain read:
// losing the editor must not lose the session.
func readLineEdited(prompt string, complete completer) (string, bool) {
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

// lineEditor is the raw-mode editing state. Every change to the line
// repaints the prompt plus the candidates that match it, so completion
// needs no TAB: matches appear as the words are typed, search-box style,
// the arrow keys move a highlight down and up the list, and Enter takes
// the highlighted row — or submits the line as typed.
type lineEditor struct {
	out      io.Writer
	prompt   string
	complete completer
	line     []byte
	sel      int // highlighted candidate; -1 means none
	top      int // first candidate shown when the list exceeds maxSuggestions
}

// readLineRaw is the editor core: it echoes to out, edits the line on
// backspace (^H, DEL), ^U (clear), ^W (delete word), submits on CR/LF,
// exits on ^C or ^D on an empty line (EOF included), moves the completion
// highlight on the arrow keys, ignores every other control byte and escape
// sequence, and keeps the list under the prompt up to date.
func readLineRaw(in io.Reader, out io.Writer, prompt string, complete completer) (string, bool) {
	e := &lineEditor{out: out, prompt: prompt, complete: complete, sel: -1}
	e.draw()
	esc := 0 // escape-sequence state: 0 none, 1 after ESC, 2 inside ESC [
	buf := make([]byte, 256)
	for {
		n, err := in.Read(buf)
		for i := 0; i < n; i++ {
			b := buf[i]
			if esc > 0 {
				// CSI/SS3 sequences: the arrows walk the completion
				// highlight, everything else (home, end, delete, …) is
				// swallowed, never echoed — the line has no cursor
				// movement to apply them to.
				if esc == 1 {
					esc = 0
					if b == '[' || b == 'O' {
						esc = 2
					}
					continue
				}
				if b >= 0x40 && b <= 0x7e {
					esc = 0
					switch b {
					case 'A': // up
						e.moveSelection(-1)
						e.draw()
					case 'B': // down
						e.moveSelection(1)
						e.draw()
					}
				}
				continue
			}
			switch {
			case b == '\r' || b == '\n':
				// Enter picks the highlighted suggestion the way a
				// search box does, then submits the line.
				if e.applySelection() {
					e.draw()
				}
				e.finishLine()
				return string(e.line), false
			case b == '\t':
				// TAB is deliberately not a completion key: the list
				// below the prompt is driven by typing and the arrows.
			case b == 0x7f || b == 0x08:
				if len(e.line) > 0 {
					e.line = e.line[:len(e.line)-runeTailLen(e.line)]
					e.sel = -1
					e.draw()
				}
			case b == 0x03: // ^C: clear the line, or leave on empty
				e.clear()
				fmt.Fprint(out, "^C\r\n")
				if len(e.line) == 0 {
					return "", true
				}
				e.line = e.line[:0]
				e.sel = -1
				e.draw()
			case b == 0x04: // ^D: leave on an empty line
				if len(e.line) == 0 {
					e.clear()
					fmt.Fprint(out, "\r\n")
					return "", true
				}
			case b == 0x15: // ^U: clear the line
				e.line = e.line[:0]
				e.sel = -1
				e.draw()
			case b == 0x17: // ^W: delete the trailing word
				e.line = trimLastWord(e.line)
				e.sel = -1
				e.draw()
			case b == 0x1b:
				esc = 1
			case b < 0x20:
				// Other control bytes: ignored.
			default:
				e.line = append(e.line, b)
				e.sel = -1
				e.draw()
			}
		}
		if err != nil {
			e.clear()
			return "", true
		}
	}
}

// draw repaints the prompt line and the matches under it, and parks the
// cursor at the end of the typed line so the next keystroke lands where the
// user is typing and the candidates stay visible below.
//
// Everything is erased downwards from the cursor, never by stepping the
// cursor up: the transcript above the prompt is the user's, and a cursor
// that walks over it takes the history with it.
func (e *lineEditor) draw() {
	cands := e.candidates()
	e.clampTop(cands)
	rows := suggestionRows(cands, e.top)
	if e.sel >= len(cands) {
		e.sel = -1
	}
	// Home + erase-down drops the prompt line's old text and the block
	// from the previous keystroke; both are redrawn below.
	fmt.Fprint(e.out, "\r\033[J")
	fmt.Fprint(e.out, e.prompt, string(e.line))
	for i, row := range rows {
		if i == e.sel-e.top {
			row = colorize(ansiReverse, row)
		}
		fmt.Fprintf(e.out, "\r\n\033[K%s", row)
	}
	if len(rows) > 0 {
		fmt.Fprintf(e.out, "\033[%dA\r", len(rows))
		if col := promptColumns(e.prompt) + utf8.RuneCount(e.line); col > 0 {
			fmt.Fprintf(e.out, "\033[%dC", col)
		}
	}
}

// clear drops the suggestion block. The cursor sits at the end of the typed
// line, so erasing from here can only reach the block underneath it.
func (e *lineEditor) clear() {
	fmt.Fprint(e.out, "\033[J")
}

// finishLine clears the suggestions and steps below the submitted line.
func (e *lineEditor) finishLine() {
	e.clear()
	fmt.Fprint(e.out, "\r\n")
}

// candidates asks the completer about the line as it stands.
func (e *lineEditor) candidates() []completion {
	if e.complete == nil {
		return nil
	}
	return e.complete(string(e.line))
}

// suggestionRows renders the visible window of matches — a scrolling
// slice of the full list when it exceeds maxSuggestions, so the arrows
// reach every row — plus the hint line.
func suggestionRows(cands []completion, top int) []string {
	if len(cands) == 0 {
		return nil
	}
	rows := make([]string, 0, maxSuggestions+2)
	for i := top; i < len(cands) && i < top+maxSuggestions; i++ {
		rows = append(rows, "  "+cands[i].Label)
	}
	return append(rows, paint(ansiDim, "  "+completionHint))
}

// moveSelection walks the highlight down (delta 1) or up (-1) through
// every match — including ones past the visible window, which scrolls to
// keep the highlight on screen — wrapping at both ends like a search box.
// Nothing highlighted yet: down takes the first row, up takes the last.
func (e *lineEditor) moveSelection(delta int) {
	cands := e.candidates()
	if len(cands) == 0 {
		e.sel = -1
		return
	}
	if e.sel < 0 {
		if delta > 0 {
			e.sel = 0
		} else {
			e.sel = len(cands) - 1
		}
	} else {
		e.sel = (e.sel + delta + len(cands)) % len(cands)
	}
	e.scrollIntoView()
}

// scrollIntoView shifts the top of the visible window so the highlighted
// candidate is among the maxSuggestions rows on screen.
func (e *lineEditor) scrollIntoView() {
	if e.sel < e.top {
		e.top = e.sel
	}
	if e.sel >= e.top+maxSuggestions {
		e.top = e.sel - maxSuggestions + 1
	}
}

// clampTop keeps the window's top inside the current candidate list —
// typing narrows the list after the window has scrolled down.
func (e *lineEditor) clampTop(cands []completion) {
	if limit := len(cands) - maxSuggestions; e.top > limit {
		if limit < 0 {
			limit = 0
		}
		e.top = limit
	}
	if e.top < 0 {
		e.top = 0
	}
}

// applySelection replaces the word being typed with the highlighted
// candidate, reporting whether there was one.
func (e *lineEditor) applySelection() bool {
	cands := e.candidates()
	if e.sel < 0 || e.sel >= len(cands) {
		return false
	}
	word := lastWord(string(e.line))
	e.line = []byte(string(e.line[:len(e.line)-len(word)]) + cands[e.sel].Value)
	e.sel = -1
	return true
}

// promptColumns is the printed width of a prompt: ANSI colors take no
// space, so they must not count toward the cursor column.
func promptColumns(s string) int {
	width := 0
	for i := 0; i < len(s); {
		if s[i] == 0x1b {
			i++
			if i < len(s) && s[i] == '[' {
				i++
				for i < len(s) && (s[i] < 0x40 || s[i] > 0x7e) {
					i++
				}
				i++ // the sequence's final byte
			}
			continue
		}
		_, size := utf8.DecodeRuneInString(s[i:])
		i += size
		width++
	}
	return width
}

// lastWord is the text after the final space of the line.
func lastWord(line string) string {
	return line[strings.LastIndexByte(line, ' ')+1:]
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
