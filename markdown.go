package main

import (
	"os"
	"strings"
)

// Minimal markdown highlighting for model answers on terminals: headings,
// bold, inline code, fenced blocks, quotes, and list markers. Deliberately
// line-oriented and naive — unmatched markers pass through untouched so a
// code sample can never leak color into the rest of the output.
//
// Gated by paintAnswer on stdout being a TTY (plus the usual kill
// switches), so piped answers stay byte-identical to plain text.

// paintAnswer highlights a model answer for terminal display.
func paintAnswer(s string) string {
	if !colorsOnFor(os.Stdout) {
		return s
	}
	return highlightMarkdown(s)
}

func highlightMarkdown(s string) string {
	lines := strings.Split(s, "\n")
	var sb strings.Builder
	inFence := false
	for i, line := range lines {
		if i > 0 {
			sb.WriteByte('\n')
		}
		trimmed := strings.TrimLeft(line, " \t")
		if strings.HasPrefix(trimmed, "```") {
			inFence = !inFence
			sb.WriteString(colorize(ansiDim, line))
			continue
		}
		if inFence {
			sb.WriteString(line)
			continue
		}
		sb.WriteString(highlightLine(line, trimmed))
	}
	return sb.String()
}

func highlightLine(line, trimmed string) string {
	indent := line[:len(line)-len(trimmed)]
	if isHeading(trimmed) {
		return indent + colorize(ansiBold, trimmed)
	}
	if strings.HasPrefix(trimmed, ">") && (trimmed == ">" || strings.HasPrefix(trimmed, "> ")) {
		return indent + colorize(ansiGreen, ">") + highlightInline(trimmed[1:])
	}
	if marker, rest, ok := splitListMarker(trimmed); ok {
		return indent + colorize(ansiCyan, marker) + highlightInline(rest)
	}
	return indent + highlightInline(trimmed)
}

func isHeading(s string) bool {
	n := 0
	for n < len(s) && s[n] == '#' {
		n++
	}
	return n >= 1 && n <= 6 && (n == len(s) || s[n] == ' ')
}

// splitListMarker splits "- ", "* ", "+ ", or "1. " style markers.
func splitListMarker(s string) (marker, rest string, ok bool) {
	for _, m := range []string{"- ", "* ", "+ "} {
		if strings.HasPrefix(s, m) {
			return m[:1], s[1:], true
		}
	}
	i := 0
	for i < len(s) && s[i] >= '0' && s[i] <= '9' {
		i++
	}
	if i > 0 && strings.HasPrefix(s[i:], ". ") {
		return s[:i+1], s[i+1:], true
	}
	return "", s, false
}

// highlightInline colors `code` and **bold** spans, hiding the markers the
// way chat renderers do. Unmatched markers stay literal so color can never
// leak past its line.
func highlightInline(s string) string {
	var sb strings.Builder
	i := 0
	for i < len(s) {
		if s[i] == '`' {
			j := strings.IndexByte(s[i+1:], '`')
			if j < 0 {
				sb.WriteString(s[i:])
				break
			}
			sb.WriteString(colorize(ansiYellow, s[i+1:i+1+j]))
			i += 2 + j
			continue
		}
		if strings.HasPrefix(s[i:], "**") {
			j := strings.Index(s[i+2:], "**")
			if j < 0 {
				sb.WriteString(s[i:])
				break
			}
			sb.WriteString(colorize(ansiBold, s[i+2:i+2+j]))
			i += 4 + j
			continue
		}
		sb.WriteByte(s[i])
		i++
	}
	return sb.String()
}
