package main

import (
	"fmt"
	"strings"
	"unicode/utf8"
)

// Colored file diffs: when the agent edits or writes a file, the user sees
// a unified diff of the change on stderr. The appearance follows codex's
// diff renderer — right-aligned line-number gutters, a `+`/`-`/space gutter
// sign per line, green content for additions, red for deletions, dim
// context and hunk headers, and a file header carrying per-file change
// counts. Colors are terminal chrome only: the model still receives the
// plain "edited /path" tool result, and NO_COLOR / CLICOLOR=0 / TERM=dumb /
// a non-TTY stderr render the same diff without escapes.

const (
	// diffContext is how many unchanged lines surround each change block.
	diffContext = 3
	// maxDiffLines caps a rendered diff so one giant write cannot flood
	// the transcript.
	maxDiffLines = 200
	// maxDiffCols caps one rendered line; longer lines are truncated with
	// an ellipsis instead of wrapping, which would scramble the gutter.
	maxDiffCols = 200
	// maxDPCells bounds the edit-distance DP table. Above it the middle
	// section (after common prefix/suffix trimming) is shown as one
	// wholesale replace block rather than a fine-grained diff.
	maxDPCells = 16 << 20
)

// diffLine is one line of a computed diff: kind is '+', '-', or ' '
// (context); oldNo/newNo are 1-based line numbers in the old/new file (0
// when the line has no counterpart there); text is the line content.
type diffLine struct {
	kind  byte
	oldNo int
	newNo int
	text  string
}

// splitLines splits file content into lines, dropping the empty element a
// trailing newline produces so "a\n" and "a" diff the same.
func splitLines(s string) []string {
	s = strings.TrimSuffix(s, "\n")
	if s == "" {
		return nil
	}
	return strings.Split(s, "\n")
}

// computeDiff diffs two file contents line by line. Common prefix and
// suffix are trimmed first — cheap, and it keeps the DP table small for
// the edits tools actually make.
func computeDiff(old, new string) []diffLine {
	a, b := splitLines(old), splitLines(new)
	if linesEqual(a, b) {
		return nil
	}
	p := 0
	for p < len(a) && p < len(b) && a[p] == b[p] {
		p++
	}
	s := 0
	for s < len(a)-p && s < len(b)-p && a[len(a)-1-s] == b[len(b)-1-s] {
		s++
	}
	var ops []diffLine
	for i := 0; i < p; i++ {
		ops = append(ops, diffLine{' ', i + 1, i + 1, a[i]})
	}
	ops = append(ops, diffMiddle(a[p:len(a)-s], b[p:len(b)-s], p)...)
	base := len(a) - s
	for i := 0; i < s; i++ {
		ops = append(ops, diffLine{' ', base + i + 1, len(b) - s + i + 1, a[len(a)-s+i]})
	}
	return ops
}

// linesEqual compares line slices cheaply.
func linesEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// diffMiddle diffs the trimmed middle sections (0-based into nothing, line
// numbers offset by base, the number of common prefix lines already
// emitted). Uses LCS DP; oversized inputs degrade to one replace block.
func diffMiddle(a, b []string, base int) []diffLine {
	if len(a) == 0 && len(b) == 0 {
		return nil
	}
	if (len(a)+1)*(len(b)+1) > maxDPCells {
		var out []diffLine
		for i, l := range a {
			out = append(out, diffLine{'-', base + i + 1, 0, l})
		}
		for i, l := range b {
			out = append(out, diffLine{'+', 0, base + i + 1, l})
		}
		return out
	}
	m, n := len(a), len(b)
	lcs := make([]int32, (m+1)*(n+1))
	at := func(i, j int) int { return i*(n+1) + j }
	for i := m - 1; i >= 0; i-- {
		for j := n - 1; j >= 0; j-- {
			switch {
			case a[i] == b[j]:
				lcs[at(i, j)] = lcs[at(i+1, j+1)] + 1
			case lcs[at(i+1, j)] >= lcs[at(i, j+1)]:
				lcs[at(i, j)] = lcs[at(i+1, j)]
			default:
				lcs[at(i, j)] = lcs[at(i, j+1)]
			}
		}
	}
	var out []diffLine
	i, j := 0, 0
	for i < m && j < n {
		switch {
		case a[i] == b[j]:
			out = append(out, diffLine{' ', base + i + 1, base + j + 1, a[i]})
			i++
			j++
		case lcs[at(i+1, j)] >= lcs[at(i, j+1)]:
			out = append(out, diffLine{'-', base + i + 1, 0, a[i]})
			i++
		default:
			out = append(out, diffLine{'+', 0, base + j + 1, b[j]})
			j++
		}
	}
	for ; i < m; i++ {
		out = append(out, diffLine{'-', base + i + 1, 0, a[i]})
	}
	for ; j < n; j++ {
		out = append(out, diffLine{'+', 0, base + j + 1, b[j]})
	}
	return out
}

// renderFileDiff renders the change from old to new as display lines:
// a file header with change counts, then hunks with context. Identical
// inputs render nothing. color=false produces the same lines without ANSI
// escapes.
func renderFileDiff(path, old, new string, color bool) []string {
	ops := computeDiff(old, new)
	if len(ops) == 0 {
		return nil
	}
	paintIf := func(code, s string) string {
		if color {
			return colorize(code, s)
		}
		return s
	}

	adds, dels := 0, 0
	for _, op := range ops {
		switch op.kind {
		case '+':
			adds++
		case '-':
			dels++
		}
	}
	header := paintIf(ansiBoldCyan, "── "+path) + "  " +
		paintIf(ansiGreen, fmt.Sprintf("+%d", adds)) + " " +
		paintIf(ansiRed, fmt.Sprintf("-%d", dels))

	// Hunk ranges: every change expands by diffContext lines on each side,
	// adjacent (or overlapping) windows merge into one hunk.
	var hunks [][2]int // inclusive op-index ranges
	for idx, op := range ops {
		if op.kind == ' ' {
			continue
		}
		lo, hi := idx-diffContext, idx+diffContext
		if lo < 0 {
			lo = 0
		}
		if hi > len(ops)-1 {
			hi = len(ops) - 1
		}
		if n := len(hunks); n > 0 && lo <= hunks[n-1][1]+1 {
			if hi > hunks[n-1][1] {
				hunks[n-1][1] = hi
			}
			continue
		}
		hunks = append(hunks, [2]int{lo, hi})
	}

	// One gutter width across the whole diff keeps the sign column
	// aligned, as codex's renderer does.
	width := 1
	for _, op := range ops {
		for _, num := range [...]int{op.oldNo, op.newNo} {
			if num > 0 {
				if w := len(fmt.Sprint(num)); w > width {
					width = w
				}
			}
		}
	}
	gutter := func(num int) string {
		if num <= 0 {
			return paintIf(ansiDim, strings.Repeat(" ", width))
		}
		return paintIf(ansiDim, fmt.Sprintf("%*d", width, num))
	}
	truncate := func(s string) string {
		if utf8.RuneCountInString(s) <= maxDiffCols {
			return s
		}
		r := []rune(s)
		return string(r[:maxDiffCols-1]) + "…"
	}

	out := []string{header}
	for _, h := range hunks {
		lo, hi := h[0], h[1]
		oldStart, oldCount, newStart, newCount := 0, 0, 0, 0
		for k := lo; k <= hi; k++ {
			if ops[k].oldNo != 0 {
				if oldCount == 0 {
					oldStart = ops[k].oldNo
				}
				oldCount++
			}
			if ops[k].newNo != 0 {
				if newCount == 0 {
					newStart = ops[k].newNo
				}
				newCount++
			}
		}
		out = append(out, paintIf(ansiDim,
			fmt.Sprintf("@@ -%d,%d +%d,%d @@", oldStart, oldCount, newStart, newCount)))
		for k := lo; k <= hi; k++ {
			op := ops[k]
			sign := " "
			body := truncate(op.text)
			switch op.kind {
			case '+':
				sign, body = paintIf(ansiGreen, "+"), paintIf(ansiGreen, body)
			case '-':
				sign, body = paintIf(ansiRed, "-"), paintIf(ansiRed, body)
			}
			out = append(out, gutter(op.oldNo)+" "+gutter(op.newNo)+" "+sign+" "+body)
		}
	}

	if len(out) > maxDiffLines {
		extra := len(out) - (maxDiffLines - 1)
		out = append(out[:maxDiffLines-1], paintIf(ansiDim, fmt.Sprintf("… %d more lines", extra)))
	}
	return out
}
