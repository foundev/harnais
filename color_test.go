package main

import (
	"testing"
)

func TestColorizeWraps(t *testing.T) {
	if got := colorize(ansiRed, "x"); got != "\033[31mx\033[0m" {
		t.Errorf("bad wrap: %q", got)
	}
}

func TestPaintRespectsKillSwitches(t *testing.T) {
	for _, env := range [][2]string{{"NO_COLOR", "1"}, {"CLICOLOR", "0"}, {"TERM", "dumb"}} {
		t.Setenv(env[0], env[1])
		if colorsOn() {
			t.Errorf("colorsOn with %s=%s", env[0], env[1])
		}
		if got := paint(ansiRed, "x"); got != "x" {
			t.Errorf("paint with %s=%s should pass through: %q", env[0], env[1], got)
		}
	}
}
