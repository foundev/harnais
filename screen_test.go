package main

import (
	"strconv"
	"strings"
	"unicode/utf8"
)

// screen replays a terminal byte stream so a test can see what the user
// would end up looking at. It is deliberately tiny: rows and columns,
// CR/LF with scrolling into scrollback, the cursor moves and erases the
// editor emits, and nothing else.
type screen struct {
	rows, cols int
	lines      [][]rune
	history    []string
	row, col   int
}

func newScreen(rows, cols int) *screen {
	s := &screen{rows: rows, cols: cols, lines: make([][]rune, rows)}
	for r := range s.lines {
		s.lines[r] = blankLine(cols)
	}
	return s
}

func blankLine(cols int) []rune {
	line := make([]rune, cols)
	for i := range line {
		line[i] = ' '
	}
	return line
}

// feed applies a stream of output bytes to the screen.
func (s *screen) feed(stream string) {
	for i := 0; i < len(stream); {
		if stream[i] == '\x1b' {
			i += s.escape(stream[i:])
			continue
		}
		switch stream[i] {
		case '\r':
			s.col = 0
			i++
		case '\n':
			s.newline()
			i++
		default:
			r, size := utf8.DecodeRuneInString(stream[i:])
			i += size
			s.lines[s.row][s.col] = r
			if s.col++; s.col >= s.cols {
				s.col = 0
				s.newline()
			}
		}
	}
}

// escape applies one escape sequence, returning how many bytes it used.
func (s *screen) escape(seq string) int {
	if len(seq) < 2 || seq[1] != '[' {
		return 1
	}
	end := 2
	for end < len(seq) && (seq[end] < 0x40 || seq[end] > 0x7e) {
		end++
	}
	if end >= len(seq) {
		return len(seq)
	}
	n := 1
	if params := seq[2:end]; params != "" {
		if v, err := strconv.Atoi(params); err == nil {
			n = v
		}
	}
	switch seq[end] {
	case 'A':
		s.row = max(0, s.row-n)
	case 'B':
		s.row = min(s.rows-1, s.row+n)
	case 'C':
		s.col = min(s.cols-1, s.col+n)
	case 'D':
		s.col = max(0, s.col-n)
	case 'J': // erase from the cursor to the end of the screen
		if n == 0 {
			for c := s.col; c < s.cols; c++ {
				s.lines[s.row][c] = ' '
			}
			for r := s.row + 1; r < s.rows; r++ {
				s.lines[r] = blankLine(s.cols)
			}
		}
	case 'K': // erase from the cursor to the end of the line
		for c := s.col; c < s.cols; c++ {
			s.lines[s.row][c] = ' '
		}
	}
	return end + 1
}

// newline drops to the next row, scrolling one line into history when the
// bottom of the screen is reached.
func (s *screen) newline() {
	if s.row++; s.row >= s.rows {
		s.history = append(s.history, string(s.lines[0]))
		s.lines = append(s.lines[1:], blankLine(s.cols))
		s.row = s.rows - 1
	}
}

// text is everything the user could still read, scrollback included.
func (s *screen) text() string {
	var b strings.Builder
	for _, line := range s.history {
		b.WriteString(strings.TrimRight(line, " "))
		b.WriteByte('\n')
	}
	for _, line := range s.lines {
		b.WriteString(strings.TrimRight(string(line), " "))
		b.WriteByte('\n')
	}
	return b.String()
}
